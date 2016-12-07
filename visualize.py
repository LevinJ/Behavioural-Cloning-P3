import sys
import os
sys.path.insert(0, os.path.abspath('..'))


import matplotlib.pyplot as plt
import pandas as pd
from myimagedatagenerator import PrepareData
from keras.models import model_from_json
from utility.vis_utils import vis_grid_withlabels



class Visualzie(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        return
#     def load_records(self):
#         filename = './data/driving_log.csv'
#         column_names=['center_imgage', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
#         self.record_df = pd.read_csv(filename, header = None, names = column_names)
#         return
    def show_angle(self):
        print(self.record_df[['steering_angle']].describe())
        self.record_df[['steering_angle', 'throttle', 'break', 'speed']].plot(subplots=True)
        plt.legend(loc='best')
        self.record_df[['steering_angle']].hist(bins=20)
        return
    def show_imgs_labels(self):
       
        generator = self.get_generator(self.df, select_bybin=True).generate_batch( batch_size=16, data_augmentation= False, test_gen = True)

        imgs = []
        labels = []
        for item in generator:
            imgs,labels = item
            break
        vis_grid_withlabels(imgs, labels)    
        return
    def show_prediction(self):

        with open('model.json', 'r') as jfile:
            json_string = jfile.read()
            model = model_from_json(json_string)

        model.compile("adam", "mse")
        model.load_weights('model.h5')
        
        generator = self.get_generator(self.df).generate_batch( batch_size=16)
        val_samples = self.y.shape[0]
        y_prd = model.predict_generator(generator, val_samples)
        self.record_df['steering_angle_pred'] = y_prd.reshape(-1)
        self.record_df[['steering_angle','steering_angle_pred']].plot()
        
        return
    def test_sample(self):
        self.load_records()
        self.load_images()
        with open('model.json', 'r') as jfile:
            json_string = jfile.read()
            model = model_from_json(json_string)

        model.compile("adam", "mse")
        model.load_weights('model.h5')
        
        generator = self.generate_batch(self.X_sample, self.y_sample, batch_size=32)
        val_samples = self.y_sample.shape[0]
        y_prd = model.predict_generator(generator, val_samples)
        print(self.y_sample)
        print(y_prd)
        
        return
    
    def run(self):
#         self.show_imgs_labels()

#         self.show_angle()
        self.show_prediction()
#         self.test_sample()
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= Visualzie()
    obj.run()