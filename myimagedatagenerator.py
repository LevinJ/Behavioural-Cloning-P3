import os

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from keras.preprocessing import image
import PIL
import matplotlib.image as mpimg
from keras.applications.vgg16 import preprocess_input



   


class MyImageDataGenerator(object):
    def __init__(self):
        self.check_integrity = False
        self.load_records()
        self.load_images()
        self.split_train_val()
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
        self.y = self.record_df[['steering_angle']].values
        
        return
    def split_train_val(self):
        imgs = self.X
        num_sample = self.X.shape[0]
        num_train = int(num_sample * 0.75)
        
        self.X_train = imgs[:num_train]
        self.y_train = self.record_df.iloc[:num_train][['steering_angle']].values
        
        self.X_val= imgs[num_train:]
        self.y_val = self.record_df.iloc[num_train:][['steering_angle']].values
        print("train/val sample number: {}/{}".format(self.y_train.shape[0], self.y_val.shape[0]))
        return
    def generate_batch(self,data, labels, batch_size=32, horizontal_flip= False, test_gen = False):
        start = 0
        num_total = data.shape[0]
        data, labels = shuffle(data, labels)
        while True:
            end = start + batch_size
            yield self.preprocess_images(data[start:end], labels[start:end], horizontal_flip, test_gen)
            
            start = end
            if start >= num_total:
                start = 0
                data, labels = shuffle(data, labels)
            
    def preprocess_images(self, image_paths, labels, horizontal_flip, test_gen):
        imgs=[]
        titles = []
        
        for i in range(image_paths.shape[0]):
            image_path = image_paths[i]
            img = image.load_img(image_path)
            flipped = ""
            if horizontal_flip:
                #flip image and label
                if np.random.random() < 0.5:
                    if labels[i] != 0.0:
                        img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
                        labels[i] = -labels[i]
                        flipped = "flipped"
            
            if test_gen:       
                titles.append(flipped+ str(labels[i]) + os.path.basename(image_path)[-16:-4])
            pixels = image.img_to_array(img)
            imgs.append(pixels)
            
        imgs = np.array(imgs)
        
        if test_gen:
            imgs = imgs.astype(np.uint8)
            return (imgs, titles)
        
        preprocess_input(imgs)
        return (imgs, labels)
    def test_flip(self):
        image_path = './data/simulator-linux/IMG/center_2016_12_03_20_31_54_279.jpg'
        img = image.load_img(image_path)
        pixels = image.img_to_array(img)
        test = mpimg.imread(image_path)
        plt.imshow(test)
#         plt.imshow(pixels.astype(np.uint8))
        plt.imshow(pixels)
#         img = PIL.Image.open('./data/simulator-linux/IMG/center_2016_12_03_20_31_54_279.jpg')
#         img2 = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
#         img2.show()
        
        return
            
    
    def run(self):
        self.test_flip()
        
        gen = self.generate_batch(self.X, self.y, horizontal_flip=True)
        
        for item in gen:
            print(item)
        
        return
    


if __name__ == "__main__":   
    obj= MyImageDataGenerator()
    obj.run()