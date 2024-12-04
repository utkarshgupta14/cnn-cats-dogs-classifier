import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization, Dropout, SpatialDropout2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l1,l2
from sklearn.metrics import accuracy_score
import glob
import random
import shutil
import itertools

os.chdir('data/dogs-and-cats')
if os.path.isdir('Train/dogs') is False:
    os.makedirs('Train/dogs')
    os.makedirs('Train/cats')
    os.makedirs('Val/dogs')
    os.makedirs('Val/cats')
    os.makedirs('Test/dogs')
    os.makedirs('Test/cats')
    for f in random.sample(glob.glob('cat*'), 1000):
        shutil.move(f, 'Train/cats')
    for f in random.sample(glob.glob('dog*'), 1000):
        shutil.move(f, 'Train/dogs')
    for f in random.sample(glob.glob('cat*'), 200):
        shutil.move(f, 'Val/cats')
    for f in random.sample(glob.glob('dog*'), 200):
        shutil.move(f, 'Val/dogs')
    for f in random.sample(glob.glob('cat*'), 200):
        shutil.move(f, 'Test/cats')
    for f in random.sample(glob.glob('dog*'), 200):
        shutil.move(f, 'Test/dogs')
os.chdir('../../')



train_path = 'data/dogs-and-cats/Train'
val_path = 'data/dogs-and-cats/Val'
test_path = 'data/dogs-and-cats/Test'
train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=train_path, target_size=(224,224), batch_size = 16)
val_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=val_path, target_size=(224,224), batch_size = 16)
test_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input).flow_from_directory(directory=test_path, target_size=(224,224), batch_size = 16, shuffle = False)



model2 = Sequential([
    Conv2D(filters = 64, padding = 'same', kernel_size = (3, 3), activation = 'relu', input_shape = (224, 224, 3), kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 64, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 128, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 128, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 256, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 256, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 256, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 512, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 512, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 512, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Conv2D(filters = 512, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 512, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    Conv2D(filters = 512, padding = 'same', kernel_size = (3, 3), activation = 'relu', kernel_regularizer = l2(0.001)),
    BatchNormalization(),
    MaxPool2D(pool_size = (2, 2), strides = 2),
    Flatten(),
    Dense(units = 4096, activation = 'relu'),
    Dropout(0.25),
    Dense(units = 4096, activation = 'relu'),
    Dropout(0.25),
    Dense(units = 2, activation = 'softmax')
])




model2.summary()




model2.compile(optimizer = Adam(learning_rate = 0.0001), loss = categorical_crossentropy, metrics = ['accuracy'])


model2.fit(x = train_batches, validation_data = val_batches, epochs = 22, verbose = 2)

model2.evaluate(test_batches, verbose = 2)

model2.save('trained_cats_dogs_model.keras')