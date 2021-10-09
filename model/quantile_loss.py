#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 19:58:02 2021

@author: weihao-tang
"""
import tensorflow as tf
import numpy as np
class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles = [0.1, 0.5, 0.9], output_size = 1):
        self.quantiles = np.array(quantiles)
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        super().__init__()

    def call(self, y_true, y_pred):
        losses = []
        for i, q in enumerate(self.quantiles):
            error = tf.subtract(
                y_true[..., self.output_size * i : self.output_size * (i + 1)],
                y_pred[..., self.output_size * i : self.output_size * (i + 1)],
            )
            loss = tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)
            losses.append(loss)

        combined_loss = tf.reduce_mean(tf.add_n(losses))
        return combined_loss
    
class Normalized_QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles = 0.5):
        self.quantiles = quantiles
        super().__init__()

    def call(self, y_true, y_pred):
        error = tf.subtract(y_true, y_pred)
        loss = tf.reduce_mean(tf.maximum(self.quantiles * error, 
                                            (self.quantiles - 1) * error), 
                                 axis=-1)

        combined_loss = loss
        # / tf.reduce_sum(tf.math.abs(y_true))
        return combined_loss