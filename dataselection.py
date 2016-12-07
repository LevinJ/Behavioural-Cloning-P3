import sys
import os
from blaze.expr.expressions import label
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Bin(object):
    def __init__(self, all_samples, bin_name):
        
        min_val,max_val = bin_name.split('/')
        max_val,min_val = float(max_val),float(min_val)
        bselected = (all_samples >min_val ) & ( all_samples <=max_val)
        self.samples = np.where(bselected == True)[0]
        
        return
    def get_next_sample(self):
        #bin will be selected by turn, while samples in bin is selected randomly
        return np.random.choice(self.samples)

class DataSelection(object):
    def __init__(self):
        self.load_records()
#         bin_names = ['1.1:0.6','0.6:0.4','0.4:0.2','0.2:0.001', 
#                      '0.001:-0.001',
#                      '-0.001:-0.1','-0.1:-0.2','-0.2:-0.3' ,'-0.3:-0.4','-0.4:-0.5','-0.5:-0.6','-0.6:-1.1']
        bin_names = ['1.1','0.6','0.4','0.2', 
                     '0.001','-0.001',
                     '-0.1','-0.2' ,'-0.3','-0.4','-0.5','-0.6', '-1.1']
        temp = []
        for i in range(len(bin_names) -1):
            item = '/'.join([bin_names[i+1], bin_names[i]])
            temp.append(item)
        bin_names = temp
        self.current_bin_index = 0
        self.current_sample_index=0
        bin_dict={}
        
        for bin_name in bin_names:
            bin = Bin(self.record_df['steering_angle'].values, bin_name)
            bin_dict[bin_name] = bin
            
        self.bin_names = bin_names
        self.bin_dict = bin_dict
       
        return
    def load_records(self):
        filename = './data/simulator-linux/driving_log_center.csv'
        column_names=['center_imgage', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
        self.record_df = pd.read_csv(filename, header = None, names = column_names)
        self.record_df = self.shuffle_records(self.record_df)
        return
    def shuffle_records(self, df):
        df =  df.iloc[np.random.permutation(len(df))]
        df = df.reset_index(drop=True)
        return df 
    def get_next_sample_bybin(self):
       
        current_bin = self.bin_dict[self.bin_names[self.current_bin_index]]
        self.current_bin_index +=1
        if self.current_bin_index >= len(self.bin_names):
            self.current_bin_index = 0
        sample_index = current_bin.get_next_sample()
        sample_record = self.record_df.iloc[sample_index]
        data = sample_record['center_imgage']
        label = sample_record['steering_angle']
            
        return data, label
    def get_next_sample(self):
       
        sample_record = self.record_df.iloc[self.current_sample_index]
        
        self.current_sample_index +=1
        if self.current_sample_index >= self.record_df.shape[0]:
            self.current_sample_index = 0
        
        
        data = sample_record['center_imgage']
        label = sample_record['steering_angle']
            
        return data, label
    
    
    
    
        
    def test_select_bysample(self):
        labels = []
        
        for i in range(1500):
            _, data_label = self.get_next_sample()
            labels.append(data_label)
        return labels
    def test_select_bybin(self):
        labels = []
        
        for i in range(1500):
            _, data_label = self.get_next_sample_bybin()
            labels.append(data_label)
        return labels
    def run(self):
        
        #test bin initialization
        for bin_name in self.bin_names:
            print('{}: {}'.format(bin_name, self.bin_dict[bin_name].samples.shape[0]))
        #test next batch
        labels = self.test_select_bybin()
#         labels = self.test_select_bysample()     
        df = pd.DataFrame(labels, columns=['labels'])
        print(df.describe())
        df.hist()
        plt.show()
            
            
       
        return
    


if __name__ == "__main__":   
    obj= DataSelection()
    obj.run()