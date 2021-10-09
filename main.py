#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 15:24:54 2021

@author: weihao-tang
"""
import tensorflow as tf
from opt_action_TFT.model.tft import TFT
from opt_action_TFT.model.quantile_loss import QuantileLoss, Normalized_QuantileLoss
import numpy as np
import time
from create_dataset.csv_to_tfdataset import WindowGenerator
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
from indicator import Flood_Indicators
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def get_traindata(dataset, batch_size):
    train_dataset = dataset.train
    val_dataset = dataset.val
    test_dataset = dataset.test
    y_train_data = np.concatenate([train_dataset['data_y'], 
                                   train_dataset['data_y'],
                                   train_dataset['data_y']], axis=-1)
    train_data = tf.data.Dataset.from_tensor_slices({'x':(train_dataset['data_h'],
                                                          train_dataset['data_f'],
                                                          train_dataset['static'],
                                                          train_dataset['pet']),
                                                     'y':y_train_data})
    y_val_data = np.concatenate([val_dataset['data_y'], 
                                 val_dataset['data_y'],
                                 val_dataset['data_y']], axis=-1)
    val_data = tf.data.Dataset.from_tensor_slices({'x':(val_dataset['data_h'],
                                                        val_dataset['data_f'],
                                                        val_dataset['static'],
                                                        val_dataset['pet']),
                                                   'y':y_val_data})
    y_test_data = np.concatenate([test_dataset['data_y'], 
                                  test_dataset['data_y'],
                                  test_dataset['data_y']], axis=-1)
    test_data = tf.data.Dataset.from_tensor_slices({'x':(test_dataset['data_h'], 
                                                         test_dataset['data_f'],
                                                         test_dataset['static'], 
                                                         test_dataset['pet']),
                                                    'y':y_test_data})
    train_data = train_data.shuffle(buffer_size=1024).batch(batch_size)
    val_data = val_data.batch(batch_size)
    test_tensor = test_data.batch(batch_size)
    test_train = test_data.shuffle(buffer_size=1024).batch(batch_size)
    return train_data, val_data, test_tensor, test_train

class Predict_Model():
    def __init__(self, structure, name='TFT', hyperparameters=None):
        self.predict_optimizer = tf.keras.optimizers.Adam()
        if hyperparameters is None:
            self.quantiles = [0.1, 0.5, 0.9]
        else:
            self.quantiles = hyperparameters['quantiles']
        self._output_size = structure['label_num']
        self.loss_function = QuantileLoss(quantiles=self.quantiles, 
                                          output_size=self._output_size)
        self.structure = structure
        self.hyperparameters = hyperparameters
        self.name = name
    
    def build_model(self):
        self.model = TFT(self.structure, self.hyperparameters)
    
    def build_tensor(self, dataset, batch_size=64):
        self.train_tensor, self.val_tensor, \
        self.test_tensor, self.test_train = get_traindata(dataset, batch_size)
    
    @tf.function
    def training(self, data):
        with tf.GradientTape() as tape:
            y_prime = self.model(data['x'], True)
            loss_value = self.loss_function(data['y'], y_prime)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.predict_optimizer.apply_gradients(zip(
            grads, self.model.trainable_weights))
        return loss_value
    
    @tf.function
    def evaluate(self, data):
        y_prime = self.model(data['x'], False)
        loss_value = self.loss_function(data['y'], y_prime)
        return loss_value
    
    def save_model(self, path='./saved_model/'):
        self.model.save_weights(path+f'{self.name}.h5')
    
    def load_model(self, path='./saved_model/'):
        for step_train, data in enumerate(self.train_tensor):
            _ = self.model(data['x'])
            break
        self.model.load_weights(path+f'{self.name}.h5')
        
    def train_predict_model(self, epochs=50, early_stop=True, patience=5, 
                            save_model=True):
        val_loss_list = []
        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0
            start_time = time.time()
            for step_train, data in enumerate(self.train_tensor):
                total_train_loss += self.training(data)  
            
            for step_val, data in enumerate(self.val_tensor):
                total_val_loss += self.evaluate(data)
            time_spent = int(time.time() - start_time)
            print(f"Epoch at {epoch}/{epochs}, Training loss %.4f, "
                  f"Validation loss %.4f, "
                  f"Time spent {time_spent}s"%(float(total_train_loss/(step_train+1)),
                                               float(total_val_loss/(step_val+1))))
            val_loss_list.append(total_val_loss/(step_val+1))
            min_val_step = np.argmin(val_loss_list)
            if early_stop:
                if epoch - min_val_step >= patience:
                    break
        if save_model:
            self.save_model(path='./saved_model/')
    
    def train_predict_model_test(self, epochs=50, early_stop=True, patience=5,
                                 save_model=True):
        val_loss_list = []
        for epoch in range(epochs):
            total_train_loss = 0
            total_val_loss = 0
            start_time = time.time()
            for step_train, data in enumerate(self.test_train):
                total_train_loss += self.training(data)  
            
            for step_val, data in enumerate(self.test_train):
                total_val_loss += self.evaluate(data)
            time_spent = int(time.time() - start_time)
            print(f"Epoch at {epoch}/{epochs}, Training loss %.4f, "
                  f"Validation loss %.4f, "
                  f"Time spent {time_spent}s"%(float(total_train_loss/(step_train+1)),
                                               float(total_val_loss/(step_val+1))))
            val_loss_list.append(total_val_loss/(step_val+1))
            min_val_step = np.argmin(val_loss_list)
            if early_stop:
                if epoch - min_val_step >= patience:
                    break
        if save_model:
            self.save_model(path='./saved_model/')

    def evaluate_valdata(self):
        total_val_loss = 0
        for step, data in enumerate(self.val_tensor):
            total_val_loss += self.evaluate(data)
        val_loss = total_val_loss/(step+1)
        print('validation loss %.4f'%(float(val_loss)))
        return val_loss
    
    def evaluate_testdata(self):
        result = []
        total_loss = 0
        for step, data in enumerate(self.test_tensor):
            y_prime = self.model(data['x'], False)
            result.append(y_prime.numpy())
            total_loss += self.loss_function(data['y'], y_prime)
        loss_value = total_loss/(step+1)
        print('test loss %.4f'%(float(loss_value)))
        return np.concatenate(result)
    
    def __call__(self, data_x):
        y_prime = self.model(data_x, False)
        return y_prime
    
def plot_continuous(continuous_result, river_site, aim_hourly, aim_index_range):
    site_ylim_dict_6 = {'3721-W':(1.0, 5.5),'3722-W':(1.0, 7.0),
                        '5921-W':(1.0, 5.0),'5173-W':(1.0, 3.5)}
    site_ylim_dict_7 = {'3721-W':(1.5, 6.0),'3722-W':(1.5, 7.0),
                        '5921-W':(1.5, 5.0),'5173-W':(1.5, 3.5)}
    site_ylim_dict = {'19-06-19':site_ylim_dict_6, '19-07-09':site_ylim_dict_7}
    
    index = continuous_result.index.strftime("%Y-%m-%d %H:%M:%S")
    num_index = list(range(len(index)))
    plt.plot(num_index,continuous_result['q50'],color='black',label='50%分位数', 
             linewidth=1.5)
    plt.fill_between(num_index, continuous_result['q10'], continuous_result['q90'],
                     facecolor='black', alpha=0.2,label='80%置信区间')
    plt.plot(num_index, continuous_result['true'],linestyle=':'
             ,color='black',label='真实值')
    plt.xlabel('时间/h', fontsize=14)
    plt.ylabel('水位高度/m', fontsize=14)
    plt.xlim((num_index[0], num_index[-1]))
    plt.legend(fontsize=12, frameon=False, loc=1)
    plt.xticks([num_index[k] for k in range(0, num_index[-1]+24, 24)],
       [num_index[k] for k in range(0, num_index[-1]+24, 24)])
    plt.ylim(site_ylim_dict[aim_index_range][river_site])
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    plt.savefig(f'./fig/{model_name} {aim_index_range} {river_site} {aim_hourly}.png', 
                dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_result_continuous(dataset, result):
    index_range_dict = {'19-06-19':('2019-06-19 18:00:00','2019-06-27 18:00:00'),
                        '19-07-09':('2019-07-09 00:00:00','2019-07-21 00:00:00')}
    test_data = dataset.test
    site_list = ['3721-W','3722-W','5921-W','5173-W']
    result_dict = {}
    aim_hourly = [6, 12]
    for aim_index_range in index_range_dict.keys():
        # writer = pd.ExcelWriter(f'{aim_index_range}.xlsx')
        for i, site in enumerate(site_list):
            result_dict[site] = result[:,:,[i,i+6,i+12]]
            predict_data = result_dict[site]*dataset.std_label[i]+dataset.mean_label[i]
            
            for col in range(3):
                predict_data[:,0,col] = predict_data[:,0,col] + test_data['label_point'][:,i]
                for row in range(1,12):
                    predict_data[:,row,col] = predict_data[:,row,col] + predict_data[:,row-1,col]
            dict_index = pd.to_datetime(test_data['timeindex_point'])
            index = pd.date_range(start=index_range_dict[aim_index_range][0], 
                                  end=index_range_dict[aim_index_range][1], freq='H')
            predict_result = dict(zip(dict_index, predict_data))
            y_true = dict(zip(dict_index, test_data['label'][:,:,i]))
            flood_result = {}
            flood_true = {}
            for idx in index:
                flood_result[idx] = predict_result[idx]
                flood_true[idx] = y_true[idx]
            for horizen in aim_hourly:
                continuous_result = {}
                for time_index in index:
                    tempindex = time_index+datetime.timedelta(hours=horizen)
                    continuous_result[tempindex] = [*list(predict_result[time_index][horizen-1,:]),
                                                    y_true[time_index][horizen-1]]
                continuous_result = pd.DataFrame(continuous_result).T
                continuous_result.columns = ['q10','q50','q90','true']
                plot_continuous(continuous_result, site, horizen, aim_index_range)
                # continuous_result.to_excel(writer, f'{site}+{horizen}')
        # writer.save()
        
def plot_result_continuous_excel(dataset, result):
    index_range_dict = {'19-06-19':('2019-06-19 18:00:00','2019-06-27 18:00:00'),
                        '19-07-09':('2019-07-09 00:00:00','2019-07-21 00:00:00'),
                        '19-05-26':('2019-05-26 06:00:00','2019-05-30 12:00:00'),
                        '19-06-30':('2019-06-30 00:00:00','2019-07-07 00:00:00'),
                        '19-02-12':('2019-02-12 18:00:00','2019-03-10 00:00:00')}
    test_data = dataset.test
    site_list = ['3721-W','3722-W','5921-W','5173-W']
    result_dict = {}
    aim_hourly = [6, 12]
    indicator = Flood_Indicators(site_list, index_range_dict.keys(), aim_hourly)
    for aim_index_range in index_range_dict.keys():
        writer = pd.ExcelWriter(f'./result_excel/{model_name}-{aim_index_range}.xlsx')
        for i, site in enumerate(site_list):
            result_dict[site] = result[:,:,[i,i+6,i+12]]
            predict_data = result_dict[site]*dataset.std_label[i]+dataset.mean_label[i]
            
            for col in range(3):
                predict_data[:,0,col] = predict_data[:,0,col] + test_data['label_point'][:,i]
                for row in range(1,12):
                    predict_data[:,row,col] = predict_data[:,row,col] + predict_data[:,row-1,col]
            dict_index = pd.to_datetime(test_data['timeindex_point'])
            index = pd.date_range(start=index_range_dict[aim_index_range][0], 
                                  end=index_range_dict[aim_index_range][1], freq='H')
            predict_result = dict(zip(dict_index, predict_data))
            y_true = dict(zip(dict_index, test_data['label'][:,:,i]))
            flood_result = {}
            flood_true = {}
            for idx in index:
                flood_result[idx] = predict_result[idx]
                flood_true[idx] = y_true[idx]
            for horizen in aim_hourly:
                continuous_result = {}
                for time_index in index:
                    tempindex = time_index+datetime.timedelta(hours=horizen)
                    continuous_result[tempindex] = [*list(predict_result[time_index][horizen-1,:]),
                                                    y_true[time_index][horizen-1]]
                continuous_result = pd.DataFrame(continuous_result).T
                continuous_result.columns = ['q10','q50','q90','true']
                # plot_continuous(continuous_result, site, horizen, aim_index_range)
                continuous_result.to_excel(writer, f'{model_name}-{site}-{horizen}')
                indicator(continuous_result, site, aim_index_range, horizen)
        writer.save()
    indicator.save(model_name)
    
def plot_baseline(dataset, result):
    aim_site = 0
    aim_index = pd.to_datetime('2019-06-20 19:00:00')
    time_index = dataset.test['timeindex_point']
    num_index = list(range(len(time_index)))
    index_dict = dict(zip(time_index,num_index))
    h_data = dataset.test['data_h'][index_dict[aim_index]][:,6+aim_site]
    h_data = h_data*dataset._std_h[6+aim_site]+dataset._mean_h[6+aim_site]
    
    y_true = dataset.test['label'][index_dict[aim_index]][:,aim_site]
    
    temp = dataset.test['label_point'][index_dict[aim_index]][aim_site]
    baseline = [temp for i in range(12)]
    
    rain_h = dataset.test['data_h'][index_dict[aim_index]][:,16]
    rain_h = rain_h*dataset._std_h[16]+dataset._mean_h[16]
    
    rain_f = dataset.test['data_f'][index_dict[aim_index]][:,4]
    rain_f = rain_f*dataset._std_f[4]+dataset._mean_f[4]
    
    y_pred_q10 = result[index_dict[aim_index]][:,aim_site]
    y_pred_q10 = y_pred_q10*dataset.std_label[aim_site]+dataset.mean_label[aim_site]
    y_pred_q10[0] = y_pred_q10[0]+dataset.test['label_point'][index_dict[
        aim_index]][aim_site]
    for i in range(1, len(y_pred_q10)):
        y_pred_q10[i] = y_pred_q10[i]+y_pred_q10[i-1]
        
    y_pred_q50 = result[index_dict[aim_index]][:,6+aim_site]
    y_pred_q50 = y_pred_q50*dataset.std_label[aim_site]+dataset.mean_label[aim_site]
    y_pred_q50[0] = y_pred_q50[0]+dataset.test['label_point'][index_dict[
        aim_index]][aim_site]
    for i in range(1, len(y_pred_q50)):
        y_pred_q50[i] = y_pred_q50[i]+y_pred_q50[i-1]
        
    y_pred_q90 = result[index_dict[aim_index]][:,12+aim_site]
    y_pred_q90 = y_pred_q90*dataset.std_label[aim_site]+dataset.mean_label[aim_site]
    y_pred_q90[0] = y_pred_q90[0]+dataset.test['label_point'][index_dict[
        aim_index]][aim_site]
    for i in range(1, len(y_pred_q90)):
        y_pred_q90[i] = y_pred_q90[i]+y_pred_q90[i-1]
        
    # index = pd.date_range(start=aim_index-datetime.timedelta(hours=47), 
    #                       end=aim_index+datetime.timedelta(hours=12), freq='H')
    
    # fig, ax = plt.subplots()
    # fig.subplots_adjust(right=0.75)
    
    # twin1 = ax.twinx()
    # twin2 = ax.twinx()
    
    # # Offset the right spine of twin2.  The ticks and label have already been
    # # placed on the right by twinx above.
    # twin2.spines.right.set_position(("axes", 1.2))
    
    # p1, = ax.plot(range(0,48),h_data,color='black',label='历史水位', 
    #               linewidth=1.5, linestyle='-.')
    # p2, = ax.plot(range(48,60),y_true,color='black',label='预测真实水位', 
    #               linewidth=1.5)
    # p3, = ax.plot(range(48,60),baseline,color='black',label='预测基准', 
    #               linewidth=1.5, linestyle=':')
    # p4, = ax.plot(range(48,60),y_pred_q50,color='black',label='水位预测0.5分位数', 
    #           linewidth=1.5, linestyle='--')
    # p5 = ax.fill_between(range(48,60), y_pred_q10, y_pred_q90,
    #                   facecolor='black', alpha=0.2,label='水位预测80%置信区间')
    
    # p6 = twin1.bar(range(0,48), rain_h, label='历史降雨量',
    #                 color='black', hatch="///", fill=False)
    # p7 = twin2.bar(range(48,60), rain_f, label='未来降雨量',
    #                 color='black', hatch="xxx", fill=False)
    
    # ax.set_xlim((0, 59))
    # ax.set_ylim((1.25, 3.25))
    # ax.set_xticks([0,10,20,30,40,50,59])
    # twin1.set_ylim(200, 0)
    # twin2.set_ylim(0, 200)
    
    # ax.set_xlabel('时间/h', fontsize=14)
    # ax.set_ylabel('水位高度/m', fontsize=14)
    # twin1.set_ylabel('历史小时累计降雨量/mm', fontsize=14)
    # twin2.set_ylabel('未来小时累计降雨量/mm', fontsize=14)

    # tkw = dict(size=4, width=1.5)
    # ax.tick_params(axis='y', **tkw)
    # twin1.tick_params(axis='y', **tkw)
    # twin2.tick_params(axis='y', **tkw)
    # ax.tick_params(axis='x', **tkw)
    
    # ax.legend(handles=[p1, p2, p3, p4, p5, p6, p7], frameon=False, 
    #           loc='center left')
    
    # plt.savefig('./fig/baseline.png', dpi=600, bbox_inches='tight')
    # plt.show()
    
    fig, ax = plt.subplots()
    fig.subplots_adjust(right=0.75)
    
    twin1 = ax.twinx()
    
    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    
    p1, = ax.plot(range(0,48),h_data,color='black',label='历史水位', 
                  linewidth=1.5, linestyle='-.')
    p2, = ax.plot(range(48,60),y_true,color='black',label='预测真实水位', 
                  linewidth=1.5)
    p3, = ax.plot(range(48,60),baseline,color='black',label='预测基准', 
                  linewidth=1.5, linestyle=':')
    p4, = ax.plot(range(48,60),y_pred_q50,color='black',label='水位预测0.5分位数', 
              linewidth=1.5, linestyle='--')
    p5 = ax.fill_between(range(48,60), y_pred_q10, y_pred_q90,
                      facecolor='black', alpha=0.2,label='水位预测80%置信区间')
    
    p6 = twin1.bar(range(0,48), rain_h, color='black')
    p7 = twin1.bar(range(48,60), rain_f, label='累计降雨量', color='black')
    
    ax.set_xlim((0, 59))
    ax.set_ylim((1.25, 4.25))
    ax.set_yticks([1.25,1.50,1.75,2.00,2.25,2.50,2.75,3.00,3.25,3.50,3.75,4.00,4.25])
    ax.set_xticks([0,10,20,30,40,50,59])
    twin1.set_ylim(250, 0)
    
    ax.set_xlabel('时间/h', fontsize=14)
    ax.set_ylabel('水位高度/m', fontsize=14)
    twin1.set_ylabel('小时累计降雨量/mm', fontsize=14)

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis='y', **tkw)
    twin1.tick_params(axis='y', **tkw)
    ax.tick_params(axis='x', **tkw)
    
    ax.legend(handles=[p1, p2, p3, p4, p5, p7], frameon=False, 
              loc='center left')
    
    plt.savefig('./fig/baseline.png', dpi=600, bbox_inches='tight')
    plt.show()
    
def eval_baseline(dataset):
    y = dataset.test['data_y']
    risk = []
    q_risk = Normalized_QuantileLoss()
    for i in range(0,4):
        risk.append(q_risk(y[:,:,i], 0.0))
    return risk
    
if __name__ == '__main__':
    model_name = 'cnn-seq2seq'
    batch_size = 64
    dataset = WindowGenerator(input_width=48, f_includeaction=True)
    model = Predict_Model(dataset.structure, model_name)
    model.build_model()
    model.build_tensor(dataset, batch_size = 64)
    model.load_model(path='./saved_model/')
    # model.train_predict_model(epochs = 50)
    # model.train_predict_model_test(epochs = 2)
    test_result = model.evaluate_testdata()
    # eval_baseline(dataset)
    plot_baseline(dataset, test_result)
    # plot_result_continuous(dataset, test_result)
    # plot_result_continuous_excel(dataset, test_result)
        
        
        
        
