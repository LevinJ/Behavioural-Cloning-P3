import sys
import os
sys.path.insert(0, os.path.abspath('..'))


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob
import pandas as pd


class BCModel(object):
    def __init__(self):
       
        return
    def load_records(self):
        filename = './data/driving_log.csv'
        column_names=['center_imgage', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
        self.record_df = pd.read_csv(filename, header = None, names = column_names)
        return
    def load_images(self):
        image_paths = glob.glob("./data/IMG/center_*.jpg")
        image_paths.sort()
        num_sample = len(image_paths)
        imgs = []
        for i in range(num_sample):
            image_path = image_paths[i]
            image_path_center = self.record_df.iloc[i]['center_imgage']
            assert os.path.basename(image_path) in image_path_center, "Houston we've got a problem"
            img = image.load_img(image_path)
            img = image.img_to_array(img)
            imgs.append(img)
        imgs = np.array(imgs)
        preprocess_input(imgs)
        #spit train val set
        
        
        num_train = int(num_sample * 0.75)
        self.X_train = imgs[:num_train]
        self.y_train = self.record_df.iloc[:num_train]['steering_angle'].values
        
        self.X_val= imgs[num_train:]
        self.y_val = self.record_df.iloc[num_train:]['steering_angle'].values
        
        
        
        return
    def run(self):
        self.load_records()
        self.load_images()
       
        return
    


if __name__ == "__main__":   
    obj= BCModel()
    obj.run()