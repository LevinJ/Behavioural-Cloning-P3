import sys
import os
sys.path.insert(0, os.path.abspath('..'))


import matplotlib.pyplot as plt
import pandas as pd
from model import BCModel
from keras.models import model_from_json


class Visualzie(BCModel):
    def __init__(self):
       
        return
#     def load_records(self):
#         filename = './data/driving_log.csv'
#         column_names=['center_imgage', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
#         self.record_df = pd.read_csv(filename, header = None, names = column_names)
#         return
    def show_angle(self):
        self.record_df[['steering_angle', 'throttle', 'break', 'speed']].plot(subplots=True)
        plt.legend(loc='best')
        return
    def show_prediction(self):
        self.load_records()
        self.load_images()
        with open('model.json', 'r') as jfile:
            json_string = jfile.read()
            model = model_from_json(json_string)

        model.compile("adam", "mse")
        model.load_weights('model.h5')
        y_prd = model.predict(self.X, batch_size=32, verbose=0)
        self.record_df['steering_angle_pred'] = y_prd
        self.record_df[['steering_angle', 'steering_angle_pred']].plot()
        plt.legend(loc='best')
        
        return
    
    def run(self):
        self.load_records()
        self.show_angle()
#         self.show_prediction()
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= Visualzie()
    obj.run()