# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 12:40:56 2021

@author: tang_
"""
import tensorflow as tf
from tensorflow import keras

def get_decoder_mask(self_attn_inputs):
  """Returns causal mask to apply for self-attention layer.

  Args:
    self_attn_inputs: Inputs to self attention layer to determine mask shape
  """
  len_s = tf.shape(self_attn_inputs)[1]
  bs = tf.shape(self_attn_inputs)[:1]
  mask = keras.backend.cumsum(tf.eye(len_s, batch_shape=bs), 1)
  return mask

def tf_stack(x, axis=0):
    if not isinstance(x, list):
        # when loading, tensorflow sometimes forgets...
        x = [x]
    return tf.keras.backend.stack(x, axis=axis)

class Linear_layer(keras.layers.Layer):
    """Returns simple Keras linear layer.
  
    Args:
      size: Output size
      activation: Activation function to apply if required
      use_time_distributed: Whether to apply layer across time
      use_bias: Whether bias should be included in layer
    """
    def __init__(self, size, activation=None, use_time_distributed=False, 
                 use_bias=True):
        super(Linear_layer, self).__init__()
        linear = keras.layers.Dense(size, activation=activation, 
                                    use_bias=use_bias)
        if use_time_distributed:
            self.linear = keras.layers.TimeDistributed(linear)
        else:
            self.linear = linear
    def call(self, inputs):
        return self.linear(inputs)

class Apply_gating_layer(keras.layers.Layer):
    """Applies a Gated Linear Unit (GLU) to an input.

    Args:
      x: Input to gating layer
      hidden_layer_size: Dimension of GLU
      dropout_rate: Dropout rate to apply if any
      use_time_distributed: Whether to apply across time
      activation: Activation function to apply to the linear feature transform if
        necessary
  
    Returns:
      Tuple of tensors for: (GLU output, gate)
    """
    def __init__(self, hidden_layer_size, dropout_rate=None,
                 use_time_distributed=True, activation=None):
        super(Apply_gating_layer, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.multiply = keras.layers.Multiply()
        dense_1 = keras.layers.Dense(hidden_layer_size, activation=activation)
        dense_2 = keras.layers.Dense(hidden_layer_size, activation='sigmoid')
        if use_time_distributed:
            self.activation_layer = keras.layers.TimeDistributed(dense_1)
            self.gated_layer = keras.layers.TimeDistributed(dense_2)
        else:
            self.activation_layer = dense_1
            self.gated_layer = dense_2
    
    def call(self, x, training):
        if self.dropout_rate is not None:
            x = self.dropout(x, training=training)
        activation = self.activation_layer(x)
        gated = self.gated_layer(x)
        outputs_1 = self.multiply([activation,gated])
        return outputs_1, gated

class Add_and_norm(keras.layers.Layer):
    def __init__(self):
        super(Add_and_norm, self).__init__()
        self.add = keras.layers.Add()
        self.layernorm = keras.layers.LayerNormalization()
    def call(self, x):
        tmp = self.add(x)
        tmp = self.layernorm(tmp)
        return tmp

class Gated_residual_network(keras.layers.Layer):
    """Applies the gated residual network (GRN) as defined in paper.
    
    Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes
    
    Returns:
        Tuple of tensors for: (GRN output, GLU gate)
    """
    def __init__(self, hidden_layer_size, output_size=None,
                 dropout_rate=None, use_time_distributed=True,
                 return_gate=False):
        super(Gated_residual_network, self).__init__()
        self.liner_layer_1 = Linear_layer(hidden_layer_size, activation=None,
                                          use_time_distributed=use_time_distributed)
        self.liner_layer_2 = Linear_layer(hidden_layer_size, activation=None,
                                          use_time_distributed=use_time_distributed,
                                          use_bias=False)
        self.liner_layer_3 = Linear_layer(hidden_layer_size, activation=None,
                                          use_time_distributed=use_time_distributed)
        self.elu = tf.keras.layers.Activation('elu')
        self.hidden_layer_size = hidden_layer_size
        self.output_flag = output_size
        self.add_and_norm = Add_and_norm()
        self.return_gate = return_gate
        if output_size is None:
            self.output_size = hidden_layer_size
        else:
            self.output_size = output_size
            self.linear = keras.layers.Dense(self.output_size)
            if use_time_distributed:
                self.linear = keras.layers.TimeDistributed(self.linear)
        self.apply_gating_layer = Apply_gating_layer(hidden_layer_size=self.output_size,
                                                     dropout_rate=dropout_rate,
                                                     use_time_distributed=use_time_distributed,
                                                     activation=None)
    def call(self, x, training, additional_context=None):
        # Setup skip connection
        if self.output_flag is None:
            skip = x
        else:
            skip = self.linear(x)
        
        # Apply feedforward network
        hidden = self.liner_layer_1(x)
        if additional_context is not None:
            hidden = hidden + self.liner_layer_2(additional_context)
        hidden = self.elu(hidden)
        hidden = self.liner_layer_3(hidden)
        
        gating_layer, gate = self.apply_gating_layer(hidden, training)
        
        if self.return_gate:
            return self.add_and_norm([skip, gating_layer]), gate
        else:
            return self.add_and_norm([skip, gating_layer])
        
    
def tempering_batchdot(input_list):
    d, k = input_list
    temper = tf.sqrt(tf.cast(k.shape[-1], dtype="float32"))
    return keras.backend.batch_dot(d, k, axes=[2, 2]) / temper

class ScaledDotProductAttention(keras.layers.Layer):
    """Defines scaled dot product attention layer.

    Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g. softmax by default)
    """

    def __init__(self, attn_dropout: float = 0.0, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(attn_dropout)
        self.activation = tf.keras.layers.Activation("softmax")

    def call(self, q, k, v, training, mask):
        """Applies scaled dot product attention.

        Args:
            q: Queries
            k: Keys
            v: Values
            mask: Masking if required -- sets softmax to very large value

        Returns:
            Tuple of (layer outputs, attention weights)
        """
        attn = keras.layers.Lambda(tempering_batchdot)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = keras.layers.Lambda(lambda x: (-1e9) * (1.0 - tf.cast(x, "float32")))(mask)  # setting to infinity
            attn = tf.keras.layers.add([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn, training=training)
        output = keras.layers.Lambda(lambda x: tf.keras.backend.batch_dot(x[0], x[1]))([attn, v])
        return output, attn
    

class InterpretableMultiHeadAttention(keras.layers.Layer):
    """Defines interpretable multi-head attention layer.

    Attributes:
        n_head: Number of heads
        d_k: Key/query dimensionality per head
        d_v: Value dimensionality
        dropout: Dropout rate to apply
        qs_layers: List of queries across heads
        ks_layers: List of keys across heads
        vs_layers: List of values across heads
        attention: Scaled dot product attention layer
        w_o: Output weight matrix to project internal state to the original TFT state size
    """

    def __init__(self, n_head: int, d_model: int, dropout: float, **kwargs):
        """Initialises layer.

        Args:
            n_head: Number of heads
            d_model: TFT state dimensionality
            dropout: Dropout discard rate
        """

        super(InterpretableMultiHeadAttention, self).__init__(**kwargs)
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # Use same value layer to facilitate interp
        vs_layer = tf.keras.layers.Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(tf.keras.layers.Dense(d_k, use_bias=False))
            self.ks_layers.append(tf.keras.layers.Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, q, k, v, training, mask=None):
        """Applies interpretable multihead attention.

        Using T to denote the number of time steps fed into the transformer.

        Args:
            q: Query tensor of shape=(?, T, d_model)
            k: Key of shape=(?, T, d_model)
            v: Values of shape=(?, T, d_model)
            mask: Masking if required with shape=(?, T, T)

        Returns:
            Tuple of (layer outputs, attention weights)
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, training, mask)

            head_dropout = tf.keras.layers.Dropout(self.dropout)(head, 
                                                                 training=training)
            heads.append(head_dropout)
            attns.append(attn)
        head = keras.layers.Lambda(tf_stack)(heads) if n_head > 1 else heads[0]
        attn = keras.layers.Lambda(tf_stack)(attns)

        outputs = keras.layers.Lambda(tf.keras.backend.mean, 
                                      arguments={"axis": 0})(head) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = tf.keras.layers.Dropout(self.dropout)(outputs, 
                                                        training=training)  # output dropout

        return outputs, attn

class Static_combine_and_mask(keras.layers.Layer):
    """Applies variable selection network to static inputs.
    
    Args:
      embedding: Transformed static inputs
    
    Returns:
      Tensor output for variable selection network
    """
    def __init__(self, hidden_layer_size, static_num, dropout_rate):
        super(Static_combine_and_mask, self).__init__()
        self._hidden_layer_size = hidden_layer_size
        self._static_num = static_num
        self._dropout_rate = dropout_rate
        self.flatten = tf.keras.layers.Flatten()
        self.gated_1 = Gated_residual_network(self._hidden_layer_size,
                                              output_size=self._static_num,
                                              dropout_rate=self._dropout_rate,
                                              use_time_distributed=False)
        self.softmax = tf.keras.layers.Activation('softmax')
        self.gated_list = []
        for _ in range(self._static_num):
            self.gated_list.append(Gated_residual_network(self._hidden_layer_size,
                                                          dropout_rate=self._dropout_rate,
                                                          use_time_distributed=False))
        self.concatenate = keras.layers.Concatenate(axis=1)
        self.multiply = tf.keras.layers.Multiply()
        
    def call(self, embedding, training):
        # Add temporal features
        flatten = self.flatten(embedding)
        # Nonlinear transformation with gated residual network.
        mlp_outputs = self.gated_1(flatten, training)
        sparse_weights = self.softmax(mlp_outputs)
        sparse_weights = tf.expand_dims(sparse_weights, axis=-1)

        trans_emb_list = []
        for i in range(self._static_num):
            e = self.gated_list[i](embedding[:, i:i + 1, :],training)
            trans_emb_list.append(e)
        
        transformed_embedding = self.concatenate(trans_emb_list)
        combined = self.multiply([sparse_weights, transformed_embedding])
        static_vec = tf.reduce_sum(combined, axis=1)
        static_vec = tf.expand_dims(static_vec, axis=1)
        return static_vec

class Static_handle(keras.layers.Layer):
    def __init__(self, static_num, static_category_counts, hidden_layer_size,
                 dropout_rate):
        super(Static_handle, self).__init__()
        self._static_num = static_num
        self._static_category_counts = static_category_counts
        self._hidden_layer_size = hidden_layer_size
        self.embedding_list = []
        self.concate = keras.layers.Concatenate(axis=1)
        for i in range(self._static_num):
            self.embedding_list.append(keras.layers.Embedding(self._static_category_counts[i],
                                                              self._hidden_layer_size,
                                                              input_length = 1))
        self.static_combine_and_mask = Static_combine_and_mask(self._hidden_layer_size, 
                                                               static_num, 
                                                               dropout_rate)
        self.gated_variable_selection = Gated_residual_network(self._hidden_layer_size,
                                                               dropout_rate=dropout_rate,
                                                               use_time_distributed=False)
        self.gated_enrichment = Gated_residual_network(self._hidden_layer_size,
                                                       dropout_rate=dropout_rate,
                                                       use_time_distributed=False)
        self.state_h = Gated_residual_network(self._hidden_layer_size,
                                              dropout_rate=dropout_rate,
                                              use_time_distributed=False)
        self.state_c = Gated_residual_network(self._hidden_layer_size,
                                              dropout_rate=dropout_rate,
                                              use_time_distributed=False)

    
    def call(self, static_inputs, training):
        embedings_static = []
        for i in range(self._static_num):
            embeding = self.embedding_list[i](static_inputs[:,:,i])
            embedings_static.append(embeding)  
        embedings_static = self.concate(embedings_static)

        weighted_static = self.static_combine_and_mask(embedings_static,
                                                       training=training)
        
        static_context_variable_selection = self.gated_variable_selection(weighted_static,
                                                                          training)        
        static_context_enrichment = self.gated_enrichment(weighted_static,
                                                          training)        
        static_context_state_h = self.state_h(weighted_static,training)
        static_context_state_c = self.state_c(weighted_static,training)
        output = [static_context_variable_selection,static_context_enrichment,
                  static_context_state_h,static_context_state_c]
        return  output
    
class H_F_data_handle(keras.layers.Layer):
    def __init__(self, hidden_layer_size, h_data_feature_num, 
                 f_data_feature_num, dropout_rate):
        super(H_F_data_handle, self).__init__()
        self._hidden_layer_size = hidden_layer_size
        self._h_data_feature_num = h_data_feature_num
        self._f_data_feature_num = f_data_feature_num
        self._dropout_rate = dropout_rate
        
        self.h_dense_list =[]
        self.h_grn_list =[]
        for _ in range(self._h_data_feature_num):
            tmp_dense = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self._hidden_layer_size))
            self.h_dense_list.append(tmp_dense)
            temp_grn = Gated_residual_network(self._hidden_layer_size,
                                              dropout_rate=self._dropout_rate,
                                              use_time_distributed=True)
            self.h_grn_list.append(temp_grn)
        
        self.f_dense_list =[]
        self.f_grn_list =[]
        for _ in range(self._f_data_feature_num):
            tmp_dense = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(self._hidden_layer_size))
            self.f_dense_list.append(tmp_dense)
            temp_grn = Gated_residual_network(self._hidden_layer_size,
                                              dropout_rate=self._dropout_rate,
                                              use_time_distributed=True)
            self.f_grn_list.append(temp_grn)
        
        self.concate = keras.layers.Concatenate(axis=2)
        self.grn_h = Gated_residual_network(self._hidden_layer_size,
                                            output_size=self._h_data_feature_num,
                                            dropout_rate=self._dropout_rate,
                                            use_time_distributed=True,
                                            return_gate=True)
        self.grn_f = Gated_residual_network(self._hidden_layer_size,
                                            output_size=self._f_data_feature_num,
                                            dropout_rate=self._dropout_rate,
                                            use_time_distributed=True,
                                            return_gate=True)
        self.softmax = tf.keras.layers.Activation('softmax')
        self.multiply = tf.keras.layers.Multiply()
        
    
    def call(self, h_data, f_data, static_vector, training):
        h_data_linear_list = []
        f_data_linear_list = []
        h_data_grn_list = []
        f_data_grn_list = []
        h_data_weighted_list = []
        f_data_weighted_list = []
        
        for i in range(0, self._h_data_feature_num):
            temp_vector = tf.expand_dims(h_data[:,:,i], axis=2)
            temp_vector = self.h_dense_list[i](temp_vector)
            h_data_linear_list.append(temp_vector)
            h_data_grn_list.append(self.h_grn_list[i](temp_vector,training))
        
        for i in range(0, self._f_data_feature_num):
            temp_vector = tf.expand_dims(f_data[:,:,i], axis=2)
            temp_vector = self.f_dense_list[i](temp_vector)
            f_data_linear_list.append(temp_vector)
            f_data_grn_list.append(self.f_grn_list[i](temp_vector,training))
        
        flatten_h_data = self.concate(h_data_linear_list)
        flatten_f_data = self.concate(f_data_linear_list)
        
        mlp_outputs_h, static_gate_h = self.grn_h(flatten_h_data,training,
                                                  additional_context=static_vector)
        sparse_weights_h = self.softmax(mlp_outputs_h)
        
        mlp_outputs_f, static_gate_f = self.grn_f(flatten_f_data,training,
                                                  additional_context=static_vector)
        sparse_weights_f = self.softmax(mlp_outputs_f)
        
        for i in range(0, self._h_data_feature_num):
            temp_weight = tf.expand_dims(sparse_weights_h[:,:,i], axis=2)
            temp_vector = self.multiply([h_data_grn_list[i], temp_weight])  
            h_data_weighted_list.append(temp_vector)
            
        h_data_weighted = tf.reduce_sum(h_data_weighted_list, axis=0)
        
        for i in range(0, self._f_data_feature_num ):
            temp_weight = tf.expand_dims(sparse_weights_f[:,:,i], axis=2)
            temp_vector = self.multiply([f_data_grn_list[i],
                                                      temp_weight])           
            f_data_weighted_list.append(temp_vector)
            
        f_data_weighted = tf.reduce_sum(f_data_weighted_list, axis=0)
        return h_data_weighted, f_data_weighted

class LSTM_layer(keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate):
        super(LSTM_layer, self).__init__()
        self.h_lstm = tf.keras.layers.LSTM(hidden_layer_size,
                                           return_sequences=True,
                                           return_state=True)
        self.f_lstm = tf.keras.layers.LSTM(hidden_layer_size,
                                           return_sequences=True,
                                           return_state=False)
        self.concate = keras.layers.Concatenate(axis=1)
        self.glu = Apply_gating_layer(hidden_layer_size, 
                                      dropout_rate, 
                                      activation=None)
        self.add_and_norm = Add_and_norm()
        
    def call(self, h_data, f_data, static_h, static_c, training):
        static_h = tf.squeeze(static_h, axis=1) # Removes dimensions of size 1 from the shape of a tensor.
        static_c = tf.squeeze(static_c, axis=1)
        history_lstm, state_h, state_c = self.h_lstm(h_data, 
                                                     initial_state=[static_h,
                                                                    static_c])
        future_lstm = self.f_lstm(f_data, 
                                  initial_state=[state_h, state_c])
        lstm_layer = self.concate([history_lstm, future_lstm])
        input_embeddings = self.concate([h_data, f_data])
        lstm_layer, _ = self.glu(lstm_layer, training)
        temporal_feature_layer = self.add_and_norm([lstm_layer, input_embeddings])
        return temporal_feature_layer

class Temporal_fusion_decoder(keras.layers.Layer):
    def __init__(self, hidden_layer_size, dropout_rate, num_heads):
        super(Temporal_fusion_decoder, self).__init__()
        self.grn_1 = Gated_residual_network(hidden_layer_size,
                                            dropout_rate=dropout_rate,
                                            use_time_distributed=True,
                                            return_gate=True)
        self.grn_2 = Gated_residual_network(hidden_layer_size,
                                            dropout_rate=dropout_rate,
                                            use_time_distributed=True)
        self.attn_layer = InterpretableMultiHeadAttention(num_heads, 
                                                          hidden_layer_size, 
                                                          dropout=dropout_rate)
        self.glu_1 = Apply_gating_layer(hidden_layer_size,
                                        dropout_rate=dropout_rate,
                                        activation=None)
        self.glu_2 = Apply_gating_layer(hidden_layer_size,
                                        dropout_rate=dropout_rate,
                                        activation=None)
        self.add_and_norm_1 = Add_and_norm()
        self.add_and_norm_2 = Add_and_norm()
        
    def call(self, temporal_feature, static_context_enrichment, training):
        enriched, _ = self.grn_1(temporal_feature,training,
                                 additional_context=static_context_enrichment)
        mask = get_decoder_mask(enriched)
        x, self_att = self.attn_layer(enriched, enriched, enriched, training,mask=mask)
        
        x, _ = self.glu_1(x, training)
        x = self.add_and_norm_1([x, enriched])
        decoder = self.grn_2(x,training)

        # Final skip connection
        decoder, _ = self.glu_2(decoder, training)
        transformer_layer = self.add_and_norm_2([decoder, temporal_feature])
        return transformer_layer
        
class TFT(tf.keras.Model):
    def __init__(self, structure, hyperparameters=None, name="encoder", **kwargs):
        super(TFT, self).__init__(name=name, **kwargs)
        # defining model structure and hyperparameters========================
        self._static_num = structure['static_num']
        self._static_category_counts = structure['static_category_counts']
        self._h_length = structure['h_length']
        self._f_length = structure['f_length']
        self._h_feature = structure['h_num']
        self._f_feature = structure['f_num']
        self._output_size = structure['label_num']
        self._pet_shape = structure['pet_shape']
        self._action_length = structure['action_length']

        if hyperparameters is not None:
            self._hidden_layer_size = hyperparameters['hidden_layer_size']
            self._dropout_rate = hyperparameters['dropout_rate']                        
            self._num_heads = hyperparameters['num_heads']     
            self._learning_rate = hyperparameters['learning_rate']
            self._max_gradient_norm = hyperparameters['max_gradient_norm']
            self.quantiles  = hyperparameters['quantiles']
        else:
            self._hidden_layer_size = 128
            self._dropout_rate = 0.1                      
            self._num_heads = 12    
            self._learning_rate = 0.001            
            self._max_gradient_norm = 1.0
            self.quantiles = [0.1, 0.5, 0.9]
        # ========================================================================
        # defining the layer
        self.static_handle = Static_handle(self._static_num, 
                                           self._static_category_counts, 
                                           self._hidden_layer_size,
                                           self._dropout_rate)
        self.h_f_data_handle = H_F_data_handle(self._hidden_layer_size, 
                                               self._h_feature, 
                                               self._f_feature, 
                                               self._dropout_rate)
        self.lstm_layer = LSTM_layer(self._hidden_layer_size, 
                                     self._dropout_rate)
        self.temporal_fusion_decoder = Temporal_fusion_decoder(self._hidden_layer_size, 
                                                               self._dropout_rate, 
                                                               self._num_heads)
        self.ouput_layer = keras.layers.TimeDistributed(
            tf.keras.layers.Dense(len(self.quantiles)*self._output_size))
    def call(self, inputs, training):
        h_data_input = inputs[0]
        f_data_input = inputs[1]
        static_data_input = inputs[2]
        static_vector = self.static_handle(static_data_input, training=training)
        h_data, f_data = self.h_f_data_handle(h_data_input, f_data_input, 
                                              static_vector[0],
                                              training=training)
        lstm = self.lstm_layer(h_data, f_data, static_vector[2], 
                               static_vector[3],
                               training=training)
        decoder = self.temporal_fusion_decoder(lstm, static_vector[1],
                                               training=training)
        outputs = self.ouput_layer(decoder[:, self._h_length:, :])
        return outputs
    

