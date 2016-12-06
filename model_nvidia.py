import sys
import os
sys.path.insert(0, os.path.abspath('..'))
from keras.models import Sequential
from keras.layers import Dense, Activation,Convolution2D, Flatten,BatchNormalization
from myimagedatagenerator import MyImageDataGenerator
from keras.optimizers import Adam
import pandas as pd
import matplotlib.pyplot as plt



class NvidiaModel(object):
    def __init__(self):
       
        return
    def setup_model(self):
        model = Sequential()
        model.add(Convolution2D(24, 5, 5, border_mode='valid',  subsample=(2,2), input_shape=(160, 320,3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(36, 5, 5, border_mode='valid',  subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(48, 5, 5, border_mode='valid',  subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='valid',  subsample=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3, border_mode='valid',  subsample=(1,1)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(1164))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(50))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(10,))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(1))
        optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss='mse')
        self.model = model
        return
    
    def train_model(self):
        gen = MyImageDataGenerator()
        batch_size = 16
        train_gen = gen.generate_batch(gen.X_train, gen.y_train, batch_size=batch_size, data_augmentation=False)
        val_gen = gen.generate_batch(gen.X_val, gen.y_val, batch_size=batch_size)
        
        
        nb_epoch =10
        
        #train fully connected layer   
        self.model.fit_generator(train_gen, gen.y_train.shape[0], nb_epoch, verbose=2, callbacks=[], 
                            validation_data=val_gen, nb_val_samples=gen.y_val.shape[0])
        with open("model.json", "w") as text_file:
            text_file.write(self.model.to_json())
        self.model.save_weights('model.h5')
        
        #tracing input data label
        input_label = pd.DataFrame(gen.input_label_tracking, columns=['steering_angle'])
        print(input_label.describe())
        input_label.hist()
        plt.show()
        return
    
        
    def run(self):
        self.setup_model()
        self.train_model()
       
        return
    


if __name__ == "__main__":   
    obj= NvidiaModel()
    obj.run()