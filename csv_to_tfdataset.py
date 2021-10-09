# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 15:43:23 2021

@author: tang_
"""
import numpy as np
import pandas as pd

def action_onehot(action):
    if isinstance(action, pd.DataFrame):
        action = action.values
    onehot_dict = {0:[0,0,0,1],50:[0,0,1,0],100:[0,1,0,0],150:[1,0,0,0]}
    onehot_list = []
    for i in range(action.shape[0]):
        temp = []
        for j in range(action.shape[1]):
            temp.extend(onehot_dict[action[i,j]])
        onehot_list.append(temp)
    return np.array(onehot_list)

def action_onehot_reverse(action):
    reverse_dict = {3:0,2:50,1:100,0:150}
    action_list = []
    for i in range(action.shape[0]):
        temp = []
        for j in range(0, action.shape[1], 4):
            temp.append(reverse_dict[np.argmax(action[i][j:j+4])])
        action_list.append(temp)
    return np.array(action_list)
        
class WindowGenerator(): 
    def __init__(self, input_width=48, label_width=12, train_per=0.7,
                 val_per=0.15, f_includeaction=False):
        # future data and history data length
        self.input_width = input_width
        self.label_width = label_width
        # read dataset and release csv
        dataset = pd.read_csv('../dataset/dataset.csv',
                              index_col='time')
        dataset.set_index(pd.to_datetime(dataset.index), inplace=True)
        release = pd.read_csv('../dataset/release.csv',
                      index_col=0)
        release.set_index(pd.to_datetime(release.index), inplace=True)
        # generate time feature
        time = dataset.index.to_list()
        month = [time[i].month-1 for i in range(len(time))]
        day = [time[i].day-1 for i in range(len(time))]
        hour = [time[i].hour for i in range(len(time))]
        time = pd.DataFrame({'month':month, 'day':day, 'hour':hour})
        time.set_index(dataset.index, inplace=True)
        # create label
        dataset_label = dataset[['3721-W', '3722-W', '5921-W', '5173-W', 
                              '3710-W', '3716-W']].diff(1)
        dataset_label.columns = ['3721-W-label', '3722-W-label', 
                                  '5921-W-label', '5173-W-label', 
                                  '3710-W-label', '3716-W-label']
        
        dataset = pd.concat([dataset_label, dataset, release, time], axis=1)
        dataset = dataset.iloc[1:] # drop the nan row
        
        pet_dataset = np.load('../dataset/pet.npy')
        self._pet_dataset = (pet_dataset-pet_dataset.mean())/ \
                            pet_dataset.std()
        
        dataset_h = dataset[['3721-W-label', '3722-W-label', 
                             '5921-W-label', '5173-W-label', 
                             '3710-W-label', '3716-W-label',
                             '3721-W', '3722-W', '5921-W', '5173-W', 
                             '3710-W', '3716-W','ar1','ar2','ar3',
                             'ar4','ar5','ar6','ar7','ar8','ar9','ar10',
                             '老石坎放水量','赋石放水量','鸭坑坞']]
        if f_includeaction:
            dataset_f = dataset[['ar1','ar2','ar3','ar4','ar5','ar6',
                                 'ar7','ar8','ar9','ar10','老石坎放水量',
                                 '赋石放水量','鸭坑坞']]
        else:
            dataset_f = dataset[['ar1','ar2','ar3','ar4','ar5','ar6',
                                 'ar7','ar8','ar9','ar10']]
        dataset_a = dataset[['老石坎放水量','赋石放水量','鸭坑坞']]
        dataset_label = dataset[['3721-W-label', '3722-W-label', 
                                 '5921-W-label', '5173-W-label', 
                                 '3710-W-label', '3716-W-label']]
        dataset_label_raw = dataset[['3721-W', '3722-W', '5921-W', '5173-W', 
                                     '3710-W', '3716-W']]
        # ====================================================================
        self._dataset_static = dataset[['hour','day', 'month']]
        
        train_index = int(train_per*len(dataset))
        val_index = int((train_per+val_per)*len(dataset))
        # divide dataset
        self._train_h = dataset_h.iloc[:train_index]
        self._val_h = dataset_h.iloc[train_index:val_index]
        self._test_h = dataset_h.iloc[val_index:]
        
        self._train_f = dataset_f.iloc[:train_index]
        self._val_f = dataset_f.iloc[train_index:val_index]
        self._test_f = dataset_f.iloc[val_index:]
          
        self._train_label = dataset_label.iloc[:train_index]
        self._val_label = dataset_label.iloc[train_index:val_index]
        self._test_label = dataset_label.iloc[val_index:]
        
        self._train_label_raw = dataset_label_raw.iloc[:train_index]
        self._val_label_raw = dataset_label_raw.iloc[train_index:val_index]
        self._test_label_raw = dataset_label_raw.iloc[val_index:]
        
        self._train_action = dataset_a.iloc[:train_index]
        self._val_action = dataset_a.iloc[train_index:val_index]
        self._test_action = dataset_a.iloc[val_index:]
        
        self._train_timeindex = dataset.index[:train_index].to_list()
        self._val_timeindex = dataset.index[train_index:val_index].to_list()
        self._test_timeindex = dataset.index[val_index:].to_list()
        # =================
        # Data normalization
        self._mean_h = self._train_h.mean()
        self._mean_f = self._train_f.mean()
        self.mean_label = self._train_label.mean()
        
        self._std_h = self._train_h.std()
        self._std_f = self._train_f.std()
        self.std_label = self._train_label.std()
        
        self._train_h = (self._train_h-self._mean_h)/self._std_h
        self._val_h = (self._val_h-self._mean_h)/self._std_h
        self._test_h = (self._test_h-self._mean_h)/self._std_h
        
        self._train_f = (self._train_f-self._mean_f)/self._std_f
        self._val_f = (self._val_f-self._mean_f)/self._std_f
        self._test_f = (self._test_f-self._mean_f)/self._std_f
        
        self._train_label_norm = (self._train_label-self.mean_label)/self.std_label
        self._val_label_norm = (self._val_label-self.mean_label)/self.std_label
        self._test_label_norm = (self._test_label-self.mean_label)/self.std_label
        # =================
        # encode action to onehot
        self._train_action = action_onehot(self._train_action)
        self._val_action = action_onehot(self._val_action)
        self._test_action = action_onehot(self._test_action)
        # =================
        self._action_num = [len(set(dataset_a[i])) for i in dataset_a.columns]
        # ====================================================================
        self.structure = {'h_length':self.input_width,
                          'h_num':dataset_h.shape[1],
                          'f_length':self.label_width,
                          'f_num':dataset_f.shape[1],
                          'label_length':self.label_width,
                          'label_num':dataset_label.shape[1],
                          'static_num':self._dataset_static.shape[1],
                          'static_category_counts':
                              [i+1 for i in self._dataset_static.max().to_list()],
                          'pet_shape':pet_dataset.shape[-2:],
                          'action_num':self._action_num,
                          'action_length':sum(self._action_num)}
                
    def stack_data(self, data_h, data_f, label, label_norm, action, timeindex):
        data_h = data_h.values
        data_f = data_f.values
        label = label.values
        label_norm = label_norm.values
        
        data_h_stack = []
        data_f_stack = []
        label_stack = []
        label_norm_stack = []
        label_point_stack = []
        timeindex_stack = []
        timeindex_point = []
        static_stack = []
        pet_stack = []
        action_stack = []
        
        start_index = self.input_width
        end_index = len(data_h) - self.label_width
        
        for i in range(start_index, end_index):
            indices = range(i-self.input_width, i, 1)
            if np.isnan(data_h[indices]).any() or \
                np.isnan(data_f[i:i+self.label_width]).any() or \
                np.isnan(label_norm[i:i+self.label_width]).any() or \
                np.isnan(label[i:i+self.label_width]).any():
                continue
            
            data_h_stack.append(data_h[indices])
            data_f_stack.append(data_f[i:i+self.label_width])
            action_stack.append(action[i:i+self.label_width])
            label_stack.append(label[i:i+self.label_width])
            label_norm_stack.append(label_norm[i:i+self.label_width])
            label_point_stack.append(label[i-1])
            timeindex_stack.append(timeindex[i:i+self.label_width])
            timeindex_point.append(timeindex[i-1])
            static_stack.append(self._dataset_static.loc[timeindex[i-1]].values)
            pet_stack.append(self._pet_dataset[self._dataset_static.
                                        loc[timeindex[i-1]].values[-1]])
        
        data_h_stack = np.array(data_h_stack, dtype=np.float32)
        data_f_stack = np.array(data_f_stack, dtype=np.float32)
        action_stack = np.array(action_stack, dtype=np.float32)
        label_stack = np.array(label_stack, dtype=np.float32)
        label_norm_stack = np.array(label_norm_stack, dtype=np.float32)
        label_point_stack = np.array(label_point_stack, dtype=np.float32)
        static_stack = np.array(static_stack, dtype=np.int32).reshape(
            (len(static_stack),1,len(self._dataset_static.columns)))
        pet_stack = np.array(pet_stack)
        
        return {'data_h':data_h_stack,'data_f':data_f_stack,
                'label':label_stack,'data_y':label_norm_stack,
                'static':static_stack,'timeindex_stack':timeindex_stack,
                'timeindex_point':timeindex_point,'pet':pet_stack,      
                'label_point':label_point_stack,
                'action':action_stack}
    
    @property
    def train(self):
        result = self.stack_data(self._train_h, 
                                 self._train_f, 
                                 self._train_label_raw,
                                 self._train_label_norm,
                                 self._train_action,
                                 self._train_timeindex)
        return result
    
    @property
    def val(self):
        result = self.stack_data(self._val_h, 
                                 self._val_f, 
                                 self._val_label_raw,
                                 self._val_label_norm,
                                 self._val_action,
                                 self._val_timeindex)
        return result
    
    @property
    def test(self):
        result = self.stack_data(self._test_h, 
                                 self._test_f, 
                                 self._test_label_raw,
                                 self._test_label_norm,
                                 self._test_action,
                                 self._test_timeindex)
        return result
if __name__ == '__main__':
    train = WindowGenerator(f_includeaction=False).train
    data_f = train['data_f']
    action = train['action']
