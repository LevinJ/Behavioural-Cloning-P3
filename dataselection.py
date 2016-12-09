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
    def __init__(self, record_df):
        #record_df, data frame with center_image and steering_angle column
        self.record_df = record_df

        bin_names = ['1.1',
                     '0.000001','-0.000001',
                    '-1.1']
        self.bin_probablity = [0.45, 0.1, 0.45]
        temp = []
        for i in range(len(bin_names) -1):
            item = '/'.join([bin_names[i+1], bin_names[i]])
            temp.append(item)
        bin_names = temp
#         self.current_bin_index = 0
        self.current_sample_index=0
        bin_dict={}
        
        for bin_name in bin_names:
            bin = Bin(self.record_df['steering_angle'].values, bin_name)
            bin_dict[bin_name] = bin
            
        self.bin_names = bin_names
        self.bin_dict = bin_dict
       
        return
    def get_next_batch(self, batch_size):
        num_total = len(self.record_df)

        if self.current_sample_index >= num_total:
            self.current_sample_index = 0
        if self.current_sample_index == 0:
            self.record_df=self.shuffle_records(self.record_df)
            
        
        end = self.current_sample_index + batch_size
        sample_records = self.record_df.iloc[self.current_sample_index: end]
        
        self.current_sample_index = end
        
        return sample_records['center_image'].values, sample_records['steering_angle'].values

    
    def get_next_sample_bybin(self):
       
        current_bin = self.bin_dict[np.random.choice(self.bin_names, p=self.bin_probablity)]
       
        sample_index = current_bin.get_next_sample()
        sample_record = self.record_df.iloc[sample_index]

        return sample_record['center_image'], sample_record['steering_angle']
    def get_next_sample(self):
       
        sample_record = self.record_df.iloc[self.current_sample_index]
        
        self.current_sample_index +=1
        if self.current_sample_index >= self.record_df.shape[0]:
            self.current_sample_index = 0
        
        
        data = sample_record['center_image']
        label = sample_record['steering_angle']
            
        return data, label
    
    
    
    
        
    def test_select_bysample(self):
        labels = []
        
        for i in range(1500):
            _, data_label = self.get_next_sample()
            labels.append(data_label)
        return labels
    def test_get_next_batch(self):
        labels = []
        
        for i in range(1500):
            _, data_label = self.get_next_batch(batch_size = 16)
            labels.extend(data_label)
        return labels
    def test_select_bybin(self):
        labels = []
        
        for i in range(20000):
            _, data_label = self.get_next_sample_bybin()
            labels.append(data_label)
        return labels
    def shuffle_records(self, df):
        df =  df.iloc[np.random.permutation(len(df))]
        df = df.reset_index(drop=True)
        return df 
    def run(self):
        
        #test bin initialization
        bin_sample_num = []
        for bin_name in self.bin_names:
            bin_sample_num.append(self.bin_dict[bin_name].samples.shape[0])
        for i in range(len(self.bin_names)):
            print('{}: {}, {:.2f}'.format(self.bin_names[i], bin_sample_num[i], bin_sample_num[i]/float(sum(bin_sample_num))))
        #test next batch
#         labels = self.test_select_bybin()
        labels = self.test_get_next_batch()     
        df = pd.DataFrame(labels, columns=['labels'])
        print(df.describe())
        df.hist(bins=20)
        plt.show()
            
            
       
        return
    
    
if __name__ == "__main__":   
    from myimagedatagenerator import PrepareData
    prepare_data = PrepareData()
    obj= DataSelection(prepare_data.df)
    obj.run()