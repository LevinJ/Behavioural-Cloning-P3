import sys
import os
sys.path.insert(0, os.path.abspath('..'))


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np
import glob
import pandas as pd
from keras.layers import Input, AveragePooling2D,Flatten
from keras.layers import Dense, Activation,merge
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import callbacks
from keras.models import load_model
from sklearn.utils import shuffle
from myimagedatagenerator import MyImageDataGenerator


class BCModel(object):
    def __init__(self):
       
        return
    
    def fine_tune_model(self):
        # create the base pre-trained model
        input_tensor = Input(shape=(160,320,3))
        base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
        # add a global spatial average pooling layer
#         x = base_model.layers[13].output
        x = base_model.output
        x = Flatten()(x)
#         x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        x = BatchNormalization()(x)
        
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        x = Dense(1024)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # and the output layer,one for the speed, the other for throttle
        predictions = Dense(1)(x)
        
        # this is the model we will train
        model = Model(input=base_model.input, output=predictions)
        # first: train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional InceptionV3 layers
        for layer in base_model.layers:
            layer.trainable = False
        # compile the model (should be done *after* setting layers to non-trainable)
        optimizer = Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        tsb = callbacks.TensorBoard(histogram_freq=1)
        cbks = [tsb]
        batch_size = 16
        nb_epoch = 1
        model.compile(optimizer=optimizer, loss='mse')
        
        gen = MyImageDataGenerator()
        train_gen = gen.generate_batch(gen.X_train, gen.y_train, batch_size=32, data_augmentation=False)
        val_gen = gen.generate_batch(gen.X_val, gen.y_val, batch_size=32)
        
        
        #train fully connected layer   
        model.fit_generator(train_gen, gen.y_train.shape[0], nb_epoch, verbose=2, callbacks=[], 
                            validation_data=val_gen, nb_val_samples=gen.y_val.shape[0])
         
        
        
        #now let's do some fine tuning
        for layer in base_model.layers[15:]:
            layer.trainable = True
        optimizer = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        nb_epoch = 1
        model.compile(optimizer=optimizer, loss='mse')
        
        model.fit_generator(train_gen, gen.y_train.shape[0], nb_epoch, verbose=2, callbacks=[], 
                            validation_data=val_gen, nb_val_samples=gen.y_val.shape[0])

        
        with open("model.json", "w") as text_file:
            text_file.write(model.to_json())
        model.save_weights('model.h5')
    

        return
    def run(self):

        self.fine_tune_model()
       
        return
    


if __name__ == "__main__":   
    obj= BCModel()
    obj.run()