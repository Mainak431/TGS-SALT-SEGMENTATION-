from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten,Conv2DTranspose,concatenate
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
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout,Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D , ZeroPadding3D , UpSampling3D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
#from keras.layers.advanced_activations import LeakyRsselu
from keras.optimizers import Adam , SGD
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
import pickle
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects



path1 = 'train/images/'
path2 = 'train/masks/'
path3 = 'train/images/'
path4 = 'train_test/'
path5 = 'train_test_mask/'
path6 = '/media/titanx/ACER DATA/Mainak/TGS SALT/create_mask/'
img_rows, img_cols ,img_channels = 256,256,3

listing = os.listdir(path3)
num_samples = size(listing)
shuffle(listing)
print(num_samples)
dp = {}
reader = []
with open("depths.csv",'r') as f:
    reader = csv.reader(f,delimiter=',')
    reader = list(reader)[1:]
    depths = []
    for i in reader :
        dp[i[0]] = int(i[1])
        depths.append(int(i[1]))
    mean_d = np.mean(depths)
    mean_std = np.std(depths)
    dp['mean'] = mean_d
    dp['std'] = mean_std
    print(dp)
    print(mean_d, mean_std)
    #exit()
pickle_out = open("dp","wb")
pickle.dump(dp,pickle_out)
with open('dp','rb') as f:
    dp = pickle.load(f)
print(dp)

#exit()
'''
for i in range(len(listing)):
    im = Image.open(path1 + listing[i])
    img = im.resize((img_rows, img_cols))
    gray = img
    # need to do some more processing here
    if i <= len(listing) * 0.8 :
        gray.save(path3 +listing[i], "JPEG")
    else :
        gray.save(path4 + listing[i], "JPEG")
    im = Image.open(path2 + listing[i])
    img = im.resize((img_rows, img_cols))
    gray = img.convert('L')
    # need to do some more processing here
    if i <= len(listing) * 0.8:
        gray.save(path6 + listing[i], "JPEG")
    else:
        gray.save(path5 + listing[i], "JPEG")
'''
start_neurons = 16
inputs = keras.layers.Input(shape=(img_cols, img_rows, img_channels))
conv1 = Conv2D(start_neurons * 1, (3, 3), activation="selu", padding="same")(inputs)
conv1 = Conv2D(start_neurons * 1, (3, 3), activation="selu", padding="same")(conv1)
pool1 = MaxPooling2D((2, 2))(conv1)
pool1 = Dropout(0.5)(pool1)

# 64 -> 32
conv2 = Conv2D(start_neurons * 2, (3, 3), activation="selu", padding="same")(pool1)
conv2 = Conv2D(start_neurons * 2, (3, 3), activation="selu", padding="same")(conv2)
pool2 = MaxPooling2D((2, 2))(conv2)
pool2 = Dropout(0.5)(pool2)

# 32 -> 16
conv3 = Conv2D(start_neurons * 4, (3, 3), activation="selu", padding="same")(pool2)
conv3 = Conv2D(start_neurons * 4, (3, 3), activation="selu", padding="same")(conv3)
pool3 = MaxPooling2D((2, 2))(conv3)
pool3 = Dropout(0.5)(pool3)

# 16 -> 8
conv4 = Conv2D(start_neurons * 8, (3, 3), activation="selu", padding="same")(pool3)
conv4 = Conv2D(start_neurons * 8, (3, 3), activation="selu", padding="same")(conv4)
pool4 = MaxPooling2D((2, 2))(conv4)
pool4 = Dropout(0.5)(pool4)
# 8 --> 4
conv5 = Conv2D(start_neurons * 16, (3, 3), activation="selu", padding="same")(pool4)
conv5 = Conv2D(start_neurons * 16, (3, 3), activation="selu", padding="same")(conv5)
pool4 = MaxPooling2D((2, 2))(conv5)
pool4 = Dropout(0.5)(pool4)

# 4---> 2

conv6 = Conv2D(start_neurons * 32, (3, 3), activation="selu", padding="same")(pool4)
conv6 = Conv2D(start_neurons * 32, (3, 3), activation="selu", padding="same")(conv6)
pool4 = MaxPooling2D((2, 2))(conv6)
pool4 = Dropout(0.5)(pool4)
#2 ---> 1
conv7 = Conv2D(start_neurons * 64, (3, 3), activation="selu", padding="same")(pool4)
conv7 = Conv2D(start_neurons * 64, (3, 3), activation="selu", padding="same")(conv7)
pool4 = MaxPooling2D((2, 2))(conv7)
pool4 = Dropout(0.5)(pool4)
# Middle
convm = Conv2D(start_neurons * 128, (3, 3), activation="selu", padding="same")(pool4)
convm1 = convm
convm = Conv2D(start_neurons * 128, (3, 3), activation="selu", padding="same")(convm)
convm = Conv2D(start_neurons * 128, (3, 3), activation="selu", padding="same")(convm)
convm = keras.layers.concatenate([convm, convm1])
#1----> 2
deconv7 = Conv2DTranspose(start_neurons * 64, (3, 3), strides=(2, 2), padding="same")(convm)
# deconv4 = UpSampling2D(size = (2,2))(convm)
print(deconv7.shape, conv7.shape)
uconv4 = keras.layers.concatenate([deconv7, conv7])
uconv4 = Dropout(0.5)(uconv4)
print(uconv4.shape)
uconv4 = Conv2D(start_neurons * 64, (3, 3), activation="selu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 64, (3, 3), activation="selu", padding="same")(uconv4)
# 2--->4
deconv6 = Conv2DTranspose(start_neurons * 32, (3, 3), strides=(2, 2), padding="same")(uconv4)
# deconv4 = UpSampling2D(size = (2,2))(convm)
print(deconv6.shape, conv6.shape)
uconv4 = keras.layers.concatenate([deconv6, conv6])
uconv4 = Dropout(0.5)(uconv4)
print(uconv4.shape)
uconv4 = Conv2D(start_neurons * 32, (3, 3), activation="selu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 32, (3, 3), activation="selu", padding="same")(uconv4)
# 4-->8
deconv5 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(uconv4)
# deconv4 = UpSampling2D(size = (2,2))(convm)
print(deconv5.shape, conv5.shape)
uconv4 = keras.layers.concatenate([deconv5, conv5])
uconv4 = Dropout(0.5)(uconv4)
print(uconv4.shape)
uconv4 = Conv2D(start_neurons * 16, (3, 3), activation="selu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 16, (3, 3), activation="selu", padding="same")(uconv4)
# 8 -> 16
deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
# deconv4 = UpSampling2D(size = (2,2))(convm)
print(deconv4.shape, conv4.shape)
uconv4 = keras.layers.concatenate([deconv4, conv4])
uconv4 = Dropout(0.5)(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="selu", padding="same")(uconv4)
uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="selu", padding="same")(uconv4)

# 16 -> 32
deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
uconv3 = keras.layers.concatenate([deconv3, conv3])
uconv3 = Dropout(0.5)(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="selu", padding="same")(uconv3)
uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="selu", padding="same")(uconv3)

# 32 -> 64
deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
uconv2 = keras.layers.concatenate([deconv2, conv2])
uconv2 = Dropout(0.5)(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="selu", padding="same")(uconv2)
uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="selu", padding="same")(uconv2)

# 64 -> 128
deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
uconv1 = keras.layers.concatenate([deconv1, conv1])
uconv1 = Dropout(0.5)(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="selu", padding="same")(uconv1)
uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="selu", padding="same")(uconv1)

uconv1 = Dropout(0.5)(uconv1)
output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

# output_layer = Conv2D(1, (1, 1), padding="same", activation=tf.ceil)(output_layer)

model = keras.Model(input=inputs, output=output_layer)
# model = keras.Model(input=inputs, output=conv7)
model.compile(loss="binary_crossentropy", optimizer= 'adadelta' , metrics=['accuracy'] )

G  = LoadBatches.imageSegmentationGenerator( path3, path6 ,  20, 2 , img_cols , img_rows , img_cols , img_rows,dp  )
G2  = LoadBatches.imageSegmentationGenerator( path4, path5 ,  20, 2 , img_cols , img_rows , img_cols , img_rows ,dp)

#hist = model.fit_generator(G, 901,validation_data=G2 , validation_steps=20, epochs=40,verbose=1)
hist = model.fit_generator(G, 500, epochs=50,verbose=1)

fname = "Unet3selufull.hdf5"
model.save_weights(fname,overwrite=True)
fname = "Unet3selufull.hdf5"
model.load_weights(fname)

#visualizing loss and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(40)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
print (plt.style.available)# use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()
#score
