import sys
import os
sys.path.insert(0, os.path.abspath('..'))


import matplotlib.pyplot as plt
import pandas as pd


class Visualzie(object):
    def __init__(self):
       
        return
    def load_records(self):
        filename = './data/driving_log.csv'
        column_names=['center_imgage', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
        self.record_df = pd.read_csv(filename, header = None, names = column_names)
        return
    def show_angle(self):
        self.record_df[['steering_angle', 'throttle', 'break', 'speed']].plot(subplots=True)
        plt.legend(loc='best')
        return
    
    def run(self):
        self.load_records()
        self.show_angle()
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= Visualzie()
    obj.run()