import sys
import os
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
        return
    def get_next_batch(self, batch_size = 32):
        data = []
        labels = []
        for _ in range(batch_size):
            current_bin = self.bin_dict[self.bin_names[self.current_bin_index]]
            self.current_bin_index +=1
            if self.current_bin_index >= len(self.bin_names):
                self.current_bin_index = 0
            sample_index = current_bin.get_next_sample()
            sample_record = self.record_df.iloc[sample_index]
            data.append(sample_record['center_imgage'])
            labels.append(sample_record['steering_angle'])
            
        return data, labels
    
    
    
        
 
    def run(self):
        
        #test bin initialization
        for bin_name in self.bin_names:
            print('{}: {}'.format(bin_name, self.bin_dict[bin_name].samples.shape[0]))
        #test next batch
        labels = []
        for i in range(500):
            data, data_labels = self.get_next_batch()
            labels.extend(data_labels)
             
        df = pd.DataFrame(labels, columns=['labels'])
        print(df.describe())
        df.hist()
        plt.show()
            
            
       
        return
    


if __name__ == "__main__":   
    obj= DataSelection()
    obj.run()