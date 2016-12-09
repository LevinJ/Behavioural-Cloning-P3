import os

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
# import keras.preprocessing.image
# import PIL
from keras.applications.inception_v3 import preprocess_input
import math
import cv2
from dataselection import DataSelection



class PrepareData(object):
    def __init__(self, use_recoverydata = False, use_side_images = False):
        self.use_recoverydata = use_recoverydata
        self.load_records()
        if use_side_images:
            self.add_side_images_records()
        self.split_train_val()
    
        return   
    def load_records(self):
      
        #this needs to be commted out during normal application running
#         self.record_df = self.load_record('./data/simulator-linux/driving_log.csv')
#         return self.record_df
        
        center_df = self.load_record('./data/simulator-linux/driving_log_center.csv')
        
        
        if not self.use_recoverydata:
            self.record_df = center_df
            return
        left_df = self.load_record('./data/simulator-linux/driving_log_left.csv')
        right_df = self.load_record('./data/simulator-linux/driving_log_right.csv')
        left_df = left_df[(left_df['steering_angle'] > 0) & (left_df['steering_angle'] < 0.8)]
        right_df = right_df[right_df['steering_angle'] < 0]
        
        self.record_df = pd.concat([center_df, left_df, right_df], ignore_index=True)
        
        
        return
    def get_sideimaegdf(self,df, adjust_angle):
        adjust_angle = np.random.uniform(low = adjust_angle/2, high=adjust_angle, size = len(df))
        resdf = df.copy()
        resdf['steering_angle'] = resdf['steering_angle'] + adjust_angle
        current_column_name = resdf.columns.values[0]
        resdf=resdf.rename(columns = {current_column_name:'center_image'})
        resdf['center_image'] = resdf['center_image'].str.replace(' ', '')
        return resdf
    def add_side_images_records(self):
        #ceter image with zero, left and right side images
        zero_left = self.record_df[self.record_df['steering_angle'] == 0][['left_image', 'steering_angle']]
        zero_left = self.get_sideimaegdf(zero_left, -0.08)
        zero_right = self.record_df[self.record_df['steering_angle'] == 0][['right_image', 'steering_angle']]
        zero_right = self.get_sideimaegdf(zero_right, 0.08)
        
        #cetenr image with negative steering angel, use left image
        negative_left = self.record_df[self.record_df['steering_angle'] < 0][['left_image', 'steering_angle']]
        negative_left = self.get_sideimaegdf(negative_left, -0.08)
        
        #center image with postive steering angle, use right image
        positive_right = self.record_df[self.record_df['steering_angle'] > 0][['right_image', 'steering_angle']]
        positive_right = self.get_sideimaegdf(positive_right, 0.08)
        
       
        
        self.record_df = self.record_df[['center_image', 'steering_angle']]
        #append left side images data
        self.record_df = pd.concat([self.record_df, zero_left, zero_right, negative_left, positive_right], ignore_index=True)
        self.record_df.to_csv('./data/validrecords.csv')
        return
    def load_record(self, filename):
        column_names=['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
        df = pd.read_csv(filename, header = None, names = column_names)
        return df

    def split_train_val(self):
        self.X = self.record_df['center_image'].values
        self.y = self.record_df['steering_angle'].values
        self.df = self.record_df
        

        num_sample = self.X.shape[0]
        num_test = 1000# the last lap for test dataset
        
        self.X_val= self.X[:num_test]
        self.y_val = self.record_df.iloc[:num_test]['steering_angle'].values
        self.valdf = self.record_df.iloc[:num_test]
        
        self.X_train = self.X[num_test:]
        self.y_train = self.record_df.iloc[num_test:]['steering_angle'].values
        self.traindf = self.record_df.iloc[num_test:]
        
       
        
        _positions = np.random.choice(num_sample, size=10, replace=False)
        self.X_sample = self.X[_positions]
        self.y_sample = self.record_df.iloc[_positions]['steering_angle'].values
        self.sampledf = self.record_df.iloc[_positions]
        
        print("train/val sample number: {}/{}".format(self.y_train.shape[0], self.y_val.shape[0]))
        return
    def get_generator(self, df, select_bybin=False):
        return  MyImageDataGenerator(df, select_bybin=select_bybin)

class DataAugmentation(object):
    def __init__(self):
        return
    def transform_image(self, img, label):
        title1 = ''
        title2 = ''
        title3 = ''
        #The distribution has zero mean, and the standard deviation is twice 
        #the standard deviation that we measured with human drivers.
#         rotation_range = 0.132052  * 180/math.pi  #
        rotation_range = 0.1  * 180/math.pi  #
#         img, label, title1 = self.flip_image(img, label)
#         img, label, title2 = self.shift_image(img, label, width_shift_range=10)
        img, label, title3 = self.rotate_image(img, label, rotation_range = rotation_range)
        title  = ",".join([title1, title2, title3])
        return img, label, title
    def flip_image(self,img,label):
        title = ""
#         if np.random.random() < 0.5:
#             if label != 0.0:
#                 img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#                 label= -label
#                 title = "f_"
#         title = title  + str(label)
        return img, label, title
    def rotate_image(self,img,label, rotation_range = 15):
        # rotation unit is degree
#         rotation_degree = np.random.normal(loc=0.0, scale=rotation_range, size=None)
        
        rotation_degree = np.random.uniform(-rotation_range, rotation_range)
        rotation_degree = 0.1  * 180/math.pi
    
        
        rows,cols, _ = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows),rotation_degree, 1)
        img = cv2.warpAffine(img,M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
#         img = cv2.warpAffine(img,M,(cols,rows))
        
        rotation_radian = math.pi *(rotation_degree/180.0)
        label -= rotation_radian  #PIL 's postive dirction is opposite to that of the game
        
        title = "r_"  + str(label)[:6]+  ":" + str(rotation_radian)[:6] + ":" + str(rotation_degree)[:6]  

        return img, label, title
    def shift_image(self,img,label, width_shift_range = 10):
        # rotation unit is degree
        shift = np.random.uniform(-width_shift_range, width_shift_range)

#         a = 1
#         b = 0
#         c = shift #left/right (i.e. 5/-5)
#         d = 0
#         e = 1
#         f = 0 #up/down (i.e. 5/-5)
#     
#         img = img.transform(img.size, PIL.Image.AFFINE, (a, b, c, d, e, f))
#         
#         label = label - 0.2 * abs(label)*shift/width_shift_range  #PIL 's postive dirction is opposite to that of the game
        
        title = "s_"  + str(label)[:6]+ ":" + str(shift)[:6]

        return img, label, title
    def preprocess_images(self, image_paths, labels, data_augmentation, test_gen):
        imgs=[]
        titles = []
        
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            
            #load the image in PIL format
#             img = keras.preprocessing.image.load_img(image_path)
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if data_augmentation:
                img, label, title = self.transform_image(img, labels[i])  
                title = ','.join([str(labels[i]), title, os.path.basename(image_path)[:-4]])
                labels[i] = label
            else:
                title = ','.join([str(labels[i]),os.path.basename(image_path)[:-4]])
            titles.append(title)

            img = img[...,::-1] #convert from opencv bgr to standard rgb
            imgs.append(img)
        #output result    
        imgs = np.array(imgs)
        if test_gen:
            imgs = imgs.astype(np.uint8)
            return (imgs, titles)
        else:
            imgs = imgs.astype(np.float32)
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
        image_path = './data/simulator-linux/IMG/center_2016_12_05_20_22_20_097.jpg'
        label = -0.04257631
        
        
#         before_img = keras.preprocessing.image.load_img(image_path)
        before_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        

        before_title = '[' + str(label) + ']' + os.path.basename(image_path)[-16:-4]
        

        
        after_img, _, after_title = self.transform_image(before_img, label)

        before_img = before_img[...,::-1] #convert from opencv bgr to standard rgb
        after_img = after_img[...,::-1] #convert from opencv bgr to standard rgb
        self.show_img_compare(before_img, before_title, after_img, after_title)
        
        return

    def run(self):
        self.test_transform()
        plt.show()      
        return
    
    
class MyImageDataGenerator(object):
    def __init__(self, record_df, select_bybin=False):

        self.data_selection = DataSelection(record_df)
        self.select_bybin = select_bybin
        self.input_label_tracking = []
        self.data_aug = DataAugmentation()
        return
    
    def generate_batch(self, batch_size=32, data_augmentation= False, test_gen = False):

        while True:
            img_paths = []
            labels = []
            if self.select_bybin:
                for _ in range(batch_size):            
                    img_path, label = self.data_selection.get_next_sample_bybin()
                    img_paths.append(img_path)
                    labels.append(label)
            else:
                img_paths, labels = self.data_selection.get_next_batch(batch_size)
            
            
            self.input_label_tracking.extend(labels)
            
            yield self.data_aug.preprocess_images(np.array(img_paths), np.array(labels), data_augmentation, test_gen)
            
    

if __name__ == "__main__":   
    obj= DataAugmentation()
    obj.run()