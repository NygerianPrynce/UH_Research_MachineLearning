import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import itertools
import os
import shutil
import random
import glob
import pickle
import matplotlib.pyplot as plt
import warnings
from PIL import Image

warnings.simplefilter(action='ignore', category=FutureWarning)

train_path1 = 'cifar-10-batches-py\data_batch_1'
train_path2 = 'cifar-10-batches-py\data_batch_2'
train_path3 = 'cifar-10-batches-py\data_batch_3'
train_path4 = 'cifar-10-batches-py\data_batch_4'
train_path5 = 'cifar-10-batches-py\data_batch_5'
valid_path = 'cifar-10-batches-py\data_bach_2'
test_path = 'cifar-10-batches-py\dest_batch'

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train_imgs = []
train_labels = []

test_imgs = []
test_labels = []

databatch1 = unpickle(train_path1)
databatch2 = unpickle(train_path2)
databatch3 = unpickle(train_path3)
databatch4 = unpickle(train_path4)
databatch5 = unpickle(train_path5)

testbatch = unpickle(test_path)

for key in databatch1[b'data']:
    key = key.reshape(3, 32, 32).transpose(1, 2, 0)
    train_imgs.append(key)
for key in databatch2[b'data']:
    key = key.reshape(3, 32, 32).transpose(1, 2, 0)
    train_imgs.append(key)
for key in databatch3[b'data']:
    key = key.reshape(3, 32, 32).transpose(1, 2, 0)
    train_imgs.append(key)
for key in databatch4[b'data']:
    key = key.reshape(3, 32, 32).transpose(1, 2, 0)
    train_imgs.append(key)
for key in databatch5[b'data']:
    key = key.reshape(3, 32, 32).transpose(1, 2, 0)
    train_imgs.append(key)

for key in databatch1[b'labels']:
    train_labels.append(key)
for key in databatch2[b'labels']:
    train_labels.append(key)
for key in databatch3[b'labels']:
    train_labels.append(key)
for key in databatch4[b'labels']:
    train_labels.append(key)
for key in databatch5[b'labels']:
    train_labels.append(key)
    







train_imgs = np.array(train_imgs)
train_labels = np.array(train_labels)

train_imgs, train_labels = shuffle(train_imgs, train_labels)

train_imgs = train_imgs/255

for key in testbatch[b'data']:
    key = key.reshape(3, 32, 32).transpose(1, 2, 0)
    test_imgs.append(key)

for key in testbatch[b'labels']:
    test_labels.append(key)

test_imgs = np.array(test_imgs)
test_labels = np.array(test_labels)

test_imgs, test_labels = shuffle(test_imgs, test_labels)

test_imgs = test_imgs/255


model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(filters=64, kernel_size=(4, 4), activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=34, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_imgs, train_labels, validation_split=0.1,  epochs=20, batch_size=200)

predictions = model.predict(test_imgs)
predictions = [np.argmax(arr) for arr in predictions]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predictions)

import seaborn as sns

plt.figure(figsize=(14, 7))
sns.heatmap(cm, annot=True)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion matrix for CIFAR-10')
plt.show()

