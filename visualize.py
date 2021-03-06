import sys
import os
sys.path.insert(0, os.path.abspath('..'))


import matplotlib.pyplot as plt
import pandas as pd
from myimagedatagenerator import PrepareData
from keras.models import model_from_json
from utility.vis_utils import vis_grid_withlabels
import cv2
from sklearn.metrics import mean_absolute_error



class Visualzie(PrepareData):
    def __init__(self):
        PrepareData.__init__(self)
        return
#     def load_records(self):
#         filename = './data/driving_log.csv'
#         column_names=['center_image', 'left_image', 'right_image', 'steering_angle', 'throttle', 'break', 'speed']
#         self.record_df = pd.read_csv(filename, header = None, names = column_names)
#         return
    def show_angle(self):
        print(self.record_df[['steering_angle']].describe())
        self.record_df[['steering_angle']].plot()
        plt.legend(loc='best')
        self.record_df[['steering_angle']].hist(bins=20)
        return
    def show_imgs_labels(self):
        generator = self.get_generator(self.df, select_bybin=False)
        generator_func = generator.generate_batch( batch_size=4, data_augmentation= False, test_gen = True)


        imgs = []
        labels = []
        for item in generator_func:
            imgs,labels = item
            break
        vis_grid_withlabels(imgs, labels)    
        return
    def show_prediction(self):
        self.test_sample()

        with open('model.json', 'r') as jfile:
            json_string = jfile.read()
            model = model_from_json(json_string)

        model.compile("adam", "mse")
        model.load_weights('model.h5')
        
        generator = self.get_generator(self.df).generate_batch( batch_size=16)
        val_samples = self.y.shape[0]
        y_pred = model.predict_generator(generator, val_samples)
        print("mean absolute error: {:.3f}".format(mean_absolute_error(self.y, y_pred)))
        self.record_df['steering_angle_pred'] = y_pred.reshape(-1)
        self.record_df[['steering_angle','steering_angle_pred']].plot()
        
        return
    def test_sample(self):
#         self.load_records()
#         self.load_images()
        with open('model.json', 'r') as jfile:
            json_string = jfile.read()
            model = model_from_json(json_string)

        model.compile("adam", "mse")
        model.load_weights('model.h5')
        
        generator = self.get_generator(self.sampledf).generate_batch( batch_size=16)
        val_samples = self.y_sample.shape[0]
        y_pred = model.predict_generator(generator, val_samples)
        print("true labels {}".format(self.y_sample))
        print("predicted labels {}".format(y_pred.reshape(-1)))
        print("mean absolute error: {:.3f}".format(mean_absolute_error(self.y_sample, y_pred)))
        
        return
    def show_batched_data_distribution(self):
        generator = self.get_generator(self.df, select_bybin=False)
        generator_fun = generator.generate_batch( batch_size=16)
        num = 0
        for _, _ in generator_fun:
            num += 1
            if num > 100:
                break
        
        df = pd.DataFrame(generator.input_label_tracking, columns=['labels'])
        print(df.describe())
        df.hist()
        return
    def show_image(self):
        image_path = './data/simulator-linux/IMG/right_2016_12_07_08_03_54_837.jpg'
        
        before_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = 1
        before_title = str(label) + "  " + image_path
        

        before_img = before_img[...,::-1] #convert from opencv bgr to standard rgb
        ax = plt.gca()
        ax.imshow(before_img)
        ax.set_title(before_title,loc='left')
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return
    def show_side_images(self):
        center_img_path = './data/simulator-linux/IMG/center_2016_12_09_06_08_47_642.jpg'
        
#         center_img_path = '/home/levin/workspace/carnd/behavioural-cloning-p3/data/simulator-linux/IMG/center_2016_12_05_20_26_51_925.jpg'
      
        
        left_img_path = center_img_path.replace('center', 'left')
        
        right_img_path = center_img_path.replace('center', 'right')
        
        paths = [left_img_path, center_img_path, right_img_path]
        _, axies = plt.subplots(1, 3)
        for i in range(len(axies)):
            axis = axies[i]
            image_path = paths[i]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, None, fx=0.5, fy=0.5)
            img = img[...,::-1] #convert from opencv bgr to standard rgb
            axis.imshow(img)
            axis.set_title(os.path.basename(image_path)[:-4],loc='left')
        return
    
    def run(self):
#         self.show_side_images()
#         self.show_image()
#         self.show_imgs_labels()

#         self.show_angle()
        self.show_prediction()
#         self.test_sample()

#         self.show_batched_data_distribution()
        
        plt.show()
        return
    


if __name__ == "__main__":   
    obj= Visualzie()
    obj.run()