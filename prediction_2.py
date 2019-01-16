from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,Conv2DTranspose
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D,UpSampling2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from PIL import Image
from sklearn.metrics import classification_report,confusion_matrix
from numpy import *
from keras import optimizers
import h5py
import cv2
import tensorflow as tf
import csv
import numpy
import keras
import LoadBatches
from random import shuffle
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape,Permute
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils

from keras import backend as K
numpy.set_printoptions(threshold=np.nan)
img_rows, img_cols ,img_channels = 128,128,3
start_neurons = 16
path1 = 'test/images/'
listing = os.listdir(path1)
num_samples = size(listing)
#shuffle(listing)
print(num_samples)
inputs =  keras.layers.Input(shape=(img_cols, img_rows,img_channels))
conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(inputs)
conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
pool1 = Dropout(0.25)(pool1)

# 64 -> 32
conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
pool2 = Dropout(0.5)(pool2)

# 32 -> 16
conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)
pool3 = Dropout(0.5)(pool3)

# 16 -> 8
conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(0.5)(pool4)
#8 --> 4
conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
conv5 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(conv5)
pool4 = MaxPooling2D((2, 2))(conv5)
pool4 = Dropout(0.5)(pool4)


# Middle
convm = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(pool4)
convm = Conv2D(start_neurons * 32, (3, 3), activation="relu", padding="same")(convm)
convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)

#4-->8
deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
#deconv4 = UpSampling2D(size = (2,2))(convm)
print(deconv5.shape,conv5.shape)
uconv4 = keras.layers.concatenate([deconv5, conv5])
uconv4 = Dropout(0.5)(uconv4)
uconv4 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(uconv4)
# 8 -> 16
deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
#deconv4 = UpSampling2D(size = (2,2))(convm)
print(deconv4.shape,conv4.shape)
uconv4 = keras.layers.concatenate([deconv4, conv4])
uconv4 = Dropout(0.5)(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)

# 16 -> 32
deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = keras.layers.concatenate([deconv3, conv3])
uconv3 = Dropout(0.5)(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)

# 32 -> 64
deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = keras.layers.concatenate([deconv2, conv2])
uconv2 = Dropout(0.5)(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)

# 64 -> 128
deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = keras.layers.concatenate([deconv1, conv1])
uconv1 = Dropout(0.5)(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)

# uconv1 = Dropout(0.5)(uconv1)
output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

model = keras.Model(input=inputs, output=output_layer)

model.load_weights("Unet2.hdf5")
path2 = 'result_mask_2/'
n_classes = 2
colors = [  ( random.randint(0,255),random.randint(0,255),random.randint(0,255)   ) for _ in range(n_classes)  ]

import pickle

with open('colours', 'wb+') as fp:
    pickle.dump(colors, fp)

for image in listing:
    x = []
    X = LoadBatches.getImageArr(path1+image, img_rows, img_cols)
    #print(X.shape)
    X.reshape(img_rows,img_cols,img_channels)
    pr = model.predict( np.array([X]) )[0]
    pr = np.array(pr)
    pr = pr.reshape((img_rows, img_cols))#.argmax(axis=2)
    print(pr)
    #break
    seg_img = np.zeros((img_rows,img_cols, 1))
    for i in range(img_rows) :
        for j in range(img_cols ):
            if pr[i][j] < 0.5 :
                seg_img[i][j] = 0
                print('absent')
            else :
                seg_img[i][j] = 255
                print('present')
    seg_img = cv2.resize(seg_img, (101, 101))
    cv2.imwrite(path2 + image, seg_img)

print(colors[0][0],colors[0][1],colors[0][2])
print(colors[1][0],colors[1][1],colors[1][2])
