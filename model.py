import os
import csv
import cv2
import sklearn
import numpy as np
import random
from keras.layers import Dense, Dropout, Flatten, Conv2D, Cropping2D, Lambda
from keras.regularizers import l2
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.python.control_flow_ops = tf


### import data from csv logfile
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


### split data into train/validation sets w' 80/20 distribution
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples,
                                                    test_size=0.2,
                                                    random_state=36)


### define generator that imports and augments image and steering data
### each line from the logfile yields 6 image/steering measurement pairs:
### center/left/right + center/left/right mirrored
def generator(samples, batch_size=64):
    nb_samples = len(samples)
    while 1:
        random.shuffle(samples)
        for offset in range(0, nb_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                
                #set image path for center/left/right cameras
                center_path = './data/IMG/'+batch_sample[0].split('\\')[-1]
                left_path = './data/IMG/'+batch_sample[1].split('\\')[-1]
                right_path = './data/IMG/'+batch_sample[2].split('\\')[-1]
                
                #load images with cv2
                center_img = cv2.imread(center_path)
                left_img = cv2.imread(left_path)
                right_img = cv2.imread(right_path)
                
                #convert images from BGR to RGB to match drive.py color format
                center_img = cv2.cvtColor(center_img, cv2.COLOR_BGR2RGB)
                left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
                right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
                
                #set steering angles; apply +-0.2 to center steering angle for left/right cameras 
                steering_correction = 0.2
                steering_center = float(batch_sample[3])
                steering_left = steering_center + steering_correction
                steering_right = steering_center - steering_correction
                
                #flip images angles to augment training data
                center_flip = cv2.flip(center_img, 1)
                left_flip = cv2.flip(center_img, 1)
                right_flip = cv2.flip(center_img, 1)
                
                #flip steering angles to match images
                steering_center_flip = -steering_center
                steering_left_flip = -steering_left
                steering_right_flip = -steering_right
                
                #add data to image/steering lists
                images.extend([center_img, left_img, right_img,
                               center_flip, left_flip, right_flip])
                steering_angles.extend([steering_center, steering_left, steering_right,
                                        steering_center_flip, steering_left_flip, steering_right_flip])
                
            X_train = np.array(images)
            y_train = np.array(steering_angles)
            
            yield sklearn.utils.shuffle(X_train, y_train)
            
            
### create train/validation generators
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
x, y = next(train_generator)
            
### set hyperparameters & variables
batch_size = 64
epochs = 20            
input_shape = x.shape[1:]


### build model & compile
model = Sequential()

#crop top/bottom from image to focus on road segment of image
model.add(Cropping2D(cropping=((50, 20), (0,0)), 
                     input_shape=(input_shape)))

#normalize image array to zero mean
model.add(Lambda(lambda x: x/127.5 - 1.0))

#architecture based on NVIDIA 'End-to-End Learning for Self-Driving Cars' paper
#added to base model: l2 regularization on convolutional layers, dropout on dense layers
model.add(Conv2D(24,5,5,subsample=(2,2), W_regularizer='l2', activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2), W_regularizer='l2', activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2), W_regularizer='l2', activation='relu'))
model.add(Conv2D(64,3,3,subsample=(2,2), W_regularizer='l2', activation='relu'))
model.add(Conv2D(64,3,3,subsample=(2,2), W_regularizer='l2', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(1))

#compile model using adam optimizer to minimize mean squared error
model.compile(loss='mse', optimizer='adam')


### train model
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples * 6),
                    nb_epoch=epochs,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples * 6),
                    callbacks=[ModelCheckpoint('model.h5', save_best_only=True)])










            
