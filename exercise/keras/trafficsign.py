from utility.dumpload import DumpLoad
import numpy as np
from sklearn.preprocessing import scale
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.preprocessing import OneHotEncoder


class TrafficeSign(object):
    def __init__(self):
    
        return
    def __get_data(self, filepath):
        dump_load = DumpLoad(filepath)
        data = dump_load.load()
        features = data['features']
        labels = data['labels'][:, np.newaxis]
        return features, labels
    def load_data(self):
        self.X_train, self.y_train =self. __get_data('./train.p')
        self.X_test, self.y_test = self.__get_data('./test.p')
        assert(self.X_train.shape[0] == self.y_train.shape[0]), "The number of images is not equal to the number of labels."
        assert(self.X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."
        return
    def normalize_data(self):
        max = 0.5
        min = -0.5
        X_std = (self.X_train - self.X_train.min()) / (self.X_train.max() - self.X_train.min())
        X_scaled = X_std * (max - min) + min
        self.X_train = X_scaled
        
#         scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
#         self.X_train = scaler.fit_transform(self.X_train.ravel())
        assert(round(np.mean(self.X_train)) == 0), "The mean of the input data is: %f" % np.mean(self.X_train)
        assert(np.min(self.X_train) == -0.5 and np.max(self.X_train) == 0.5), "The range of the input data is: %.1f to %.1f" % (np.min(self.X_train), np.max(self.X_train))
        return
    def two_layer_net(self):
        model = Sequential()
        
        model.add(Dense(128, input_dim=32*32*3,  name="hidden1"))
        model.add(Activation("relu"))
        
        model.add(Dense(output_dim=43,  name="output"))
        model.add(Activation("softmax"))
        
        # STOP: Do not change the tests below. Your implementation should pass these tests.
        assert(model.get_layer(name="hidden1").input_shape == (None, 32*32*3)), "The input shape is: %s" % model.get_layer(name="hidden1").input_shape
        assert(model.get_layer(name="output").output_shape == (None, 43)), "The output shape is: %s" % model.get_layer(name="output").output_shape 
        
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.encoder = OneHotEncoder(sparse=False).fit(self.y_train)

        y_train_encoded  = self.encoder.transform(self.y_train)
        history  = model.fit(self.X_train.reshape(-1,32*32*3), y_train_encoded, nb_epoch=2, batch_size=32)
        
        # STOP: Do not change the tests below. Your implementation should pass these tests.
        assert(history.history['acc'][0] > 0.5), "The training accuracy was: %.3f" % history.history['acc']
        return
        
    def run(self):
        self.load_data()
        self.normalize_data()
        self.two_layer_net()
        return
    

    
if __name__ == "__main__":   
    obj= TrafficeSign()
    obj.run()