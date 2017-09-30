import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

lines = []
images = []
measurements = []

samples = []

# load driving data: camera images and steering angles

with open('my_data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

# offset factor for images from left and right camera
factor = 0.2
offset = factor*np.array([0.0, 1.0, -1.0])

# fill variables with image and steering data (normal driving set)
for line in lines[1:]:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'my_data/IMG/' + filename
		image = cv2.imread(current_path)
		if image is not None:
			image_flipped = np.fliplr(image)
			images.append(image_flipped)
			images.append(image)
			measurement = float(line[3])+offset[i]
			measurements.append(-measurement)
			measurements.append(measurement)

lines = []

# fill variables with image and steering data (recovery set)

with open('recovery_laps/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

factor = 0.2
offset = factor*np.array([0.0, 1.0, -1.0])
for line in lines[1:]:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1]
		current_path = 'recovery_laps/IMG/' + filename
		image = cv2.imread(current_path)
		if image is not None:
			image_flipped = np.fliplr(image)
			images.append(image_flipped)
			images.append(image)
			measurement = float(line[3])+offset[i]
			measurements.append(-measurement)
			measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Convolution2D, Lambda
from keras.layers import Cropping2D

# define neural network

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0, 0))))
model.add(Convolution2D(6, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(18, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(54, 5, 5, activation = "relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(108, activation = "softsign"))
model.add(Dense(54, activation = "softsign"))
model.add(Dense(1))

# compile and fit the model

model.compile(loss='mse', optimizer='adam')
model.fit (X_train, y_train, validation_split = 0.2, shuffle= True, nb_epoch = 5)

# save model

model.save('model.h5')
