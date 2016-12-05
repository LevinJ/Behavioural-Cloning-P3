import sys
import os
sys.path.insert(0, os.path.abspath('..'))


import matplotlib.pyplot as plt
import pandas as pd
from myimagedatagenerator import MyImageDataGenerator
from keras.models import model_from_json
from utility.vis_utils import vis_grid_withlabels



class Visualzie(MyImageDataGenerator):
    def __init__(self):
        MyImageDataGenerator.__init__(self)
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
    def show_imgs_labels(self):
        generator = self.generate_batch(self.X, self.y, batch_size=16, horizontal_flip= True, test_gen = True)
#         generator = self.generate_batch(self.X, self.y, batch_size=16, horizontal_flip= False, test_gen = False)
        imgs = []
        labels = []
        for item in generator:
            imgs,labels = item
            break
        vis_grid_withlabels(imgs, labels)    
        return
    def show_prediction(self):
        self.load_records()
        self.load_images()
        with open('model.json', 'r') as jfile:
            json_string = jfile.read()
            model = model_from_json(json_string)

        model.compile("adam", "mse")
        model.load_weights('model.h5')
        
        generator = self.generate_batch(self.X, self.y, batch_size=32)
        val_samples = self.y.shape[0]
        y_prd = model.predict_generator(generator, val_samples)
#         y_prd = model.predict(self.X, batch_size=32, verbose=0)
        self.record_df['steering_angle_pred'] = y_prd
        self.record_df[['steering_angle','steering_angle_pred']].plot()

#         self.record_df['steering_angle_pred'] = y_prd[:,0]
#         self.record_df['throttle_pred'] = y_prd[:,1]
#         
#         _, (ax1, ax2) = plt.subplots(2, 1)
#         ax1.plot(self.record_df[['steering_angle']])
#         ax1.plot(self.record_df[['steering_angle_pred']])
#         
#         ax2.plot(self.record_df[['throttle']])
#         ax2.plot(self.record_df[['throttle_pred']])
        
#         plt.subplot(2,1,1)
#         self.record_df[['steering_angle', 'steering_angle_pred']].plot()
#         plt.legend(loc='best')
#    
#         self.record_df[['throttle', 'throttle_pred']].plot()
#         plt.legend(loc='best')
        
        return
    
    def run(self):
#         self.show_imgs_labels()

        self.show_angle()
#         self.show_prediction()
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= Visualzie()
    obj.run()