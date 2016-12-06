import os

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
import keras.preprocessing.image
import PIL
from keras.applications.inception_v3 import preprocess_input
import math



   


class MyImageDataGenerator(object):
    def __init__(self):
        self.check_integrity = False
        self.load_records()
        self.load_images()
        self.split_train_val()
        self.input_label_tracking = []
        return
    def load_records(self):
        filename = './data/simulator-linux/driving_log.csv'
        column_names=['center_imgage', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
        self.record_df = pd.read_csv(filename, header = None, names = column_names)
        return
    def load_images(self):
        image_paths = glob.glob("./data/simulator-linux/IMG/center_*.jpg")
        image_paths.sort()
        
        #make sure each image has the right label attached
        if self.check_integrity:
            num_sample = len(image_paths)
            for i in range(num_sample):
                image_path = image_paths[i]
                image_path_center = self.record_df.iloc[i]['center_imgage']
                assert os.path.basename(image_path) in image_path_center, "Houston we've got a problem"

        self.X = np.array(image_paths)
        self.y = self.record_df['steering_angle'].values
        
        return
    def split_train_val(self):
        imgs = self.X
        num_sample = self.X.shape[0]
        num_train = num_sample - 1000# the last lap for test dataset
        
        _positions = np.random.choice(num_sample, size=10, replace=False)
        self.X_sample = imgs[_positions]
        self.y_sample = self.record_df.iloc[_positions]['steering_angle'].values
        
        self.X_train = imgs[:num_train]
        self.y_train = self.record_df.iloc[:num_train]['steering_angle'].values
        
        self.X_val= imgs[num_train:]
        self.y_val = self.record_df.iloc[num_train:]['steering_angle'].values
        print("train/val sample number: {}/{}".format(self.y_train.shape[0], self.y_val.shape[0]))
        return
    def generate_batch(self,data, labels, batch_size=32, data_augmentation= False, test_gen = False):
        start = 0
        num_total = data.shape[0]
        data, labels = shuffle(data, labels)
        while True:
            end = start + batch_size
            yield self.preprocess_images(data[start:end], labels[start:end], data_augmentation, test_gen)
            
            start = end
            if start >= num_total:
                start = 0
                data, labels = shuffle(data, labels)
    
    def transform_image(self, img, label):
        title1 = ''
        title2 = ''
        title3 = ''
        #The distribution has zero mean, and the standard deviation is twice 
        #the standard deviation that we measured with human drivers.
#         rotation_std = 0.132052  * 180/math.pi  #
        rotation_std = 0.01  * 180/math.pi  #
#         img, label, title1 = self.flip_image(img, label)
#         img, label, title2 = self.shift_image(img, label, width_shift_range=10)
        img, label, title3 = self.rotate_image(img, label, rotation_std = rotation_std)
        title  = ",".join([title1, title2, title3])
        return img, label, title
    def flip_image(self,img,label):
        title = ""
        if np.random.random() < 0.5:
            if label != 0.0:
                img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                label= -label
                title = "f_"
        title = title  + str(label)
        return img, label, title
    def rotate_image(self,img,label, rotation_std = 15):
        # rotation unit is degree
#         rotation_degree = np.random.normal(loc=0.0, scale=rotation_std, size=None)
        rotation_degree = np.random.uniform(-rotation_std, rotation_std)
    
        img = img.rotate(rotation_degree)
        rotation_radian = math.pi *(rotation_degree/180.0)
        label -= rotation_radian  #PIL 's postive dirction is opposite to that of the game
        
        title = "r_"  + str(label)[:6]+  ":" + str(rotation_radian)[:6] + ":" + str(rotation_degree)[:6]  

        return img, label, title
    def shift_image(self,img,label, width_shift_range = 10):
        # rotation unit is degree
        shift = np.random.uniform(-width_shift_range, width_shift_range)

        a = 1
        b = 0
        c = shift #left/right (i.e. 5/-5)
        d = 0
        e = 1
        f = 0 #up/down (i.e. 5/-5)
    
        img = img.transform(img.size, PIL.Image.AFFINE, (a, b, c, d, e, f))
        
        label = label - 0.2 * abs(label)*shift/width_shift_range  #PIL 's postive dirction is opposite to that of the game
        
        title = "s_"  + str(label)[:6]+ ":" + str(shift)[:6]

        return img, label, title
    def preprocess_images(self, image_paths, labels, data_augmentation, test_gen):
        imgs=[]
        titles = []
        
        for i in range(image_paths.shape[0]):
            image_path = image_paths[i]
            #load the image in PIL format
            img = keras.preprocessing.image.load_img(image_path)
            if data_augmentation:
                img, label, title = self.transform_image(img, labels[i])  
                title = ','.join([str(labels[i]), title, os.path.basename(image_path)[-16:-4]])
                titles.append(title)
                labels[i] = label
                self.input_label_tracking.append(label) 
                
            pixels = keras.preprocessing.image.img_to_array(img)
            imgs.append(pixels)
        #output result    
        imgs = np.array(imgs)
        if test_gen:
            imgs = imgs.astype(np.uint8)
            return (imgs, titles)
        else:
            preprocess_input(imgs)
            return (imgs, labels)
   
    def show_img_compare(self, before_img, before_title, after_img, after_title):
        _,(ax1,ax2) = plt.subplots(1, 2)
        ax1.imshow(before_img)
        ax1.set_title(before_title,loc='left')
        ax1.grid(False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax2.imshow(after_img)
        ax2.set_title(after_title,loc='left')
        ax2.grid(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        return
    def test_transform(self):
        image_path = './data/simulator-linux/IMG/center_2016_12_05_20_22_21_004.jpg'
        label = 0
        
        
        before_img = keras.preprocessing.image.load_img(image_path)
        

        before_title = '[' + str(label) + ']' + os.path.basename(image_path)[-16:-4]
        

        
        after_img, _, after_title = self.transform_image(before_img, label)

        
        self.show_img_compare(before_img, before_title, after_img, after_title)
        
        return
            
    
    def run(self):
        self.test_transform()
        plt.show()

        
        
#         gen = self.generate_batch(self.X, self.y, horizontal_flip=True)
#         
#         for item in gen:
#             print(item)
        
        return
    


if __name__ == "__main__":   
    obj= MyImageDataGenerator()
    obj.run()