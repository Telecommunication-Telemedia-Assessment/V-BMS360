import readFlo as fl
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


# read all the flow files, to generate the dataset
def load_data():

	SRC = [1, 10, 11, 12, 15, 16]		# List of SRC
	SRC_CLASS = [1, 2, 3, 3, 3, 2]	# Categories of motion => 1: floor moving, 2: camera moving, 3: static scene
	NB_IMAGES = 900
	FLOW_SIZE = [45, 80] # don't work with full resolution, that would require too much memory...
	ROOT_FLOW_FILES = "/Users/pierre/Downloads/test3/flow"


	# Pre-allocation of memory
	flowArray  = np.ndarray(shape=(len(SRC)*NB_IMAGES, FLOW_SIZE[0], FLOW_SIZE[1], 2), dtype=float, order='F')
	labelArray = np.ones((len(SRC)*NB_IMAGES), dtype=int)


	# Prepare the data
	for srcIdx in range(0,len(SRC)):

		# Set the class
		labelArray[(srcIdx*NB_IMAGES):((srcIdx+1)*NB_IMAGES)] = SRC_CLASS[srcIdx]


		# Load all the flow files
		rootPath = ROOT_FLOW_FILES + "/" + str(SRC[srcIdx]) + "/"

		for frameIdx in range(2,NB_IMAGES+2):
			path = rootPath + str(frameIdx) + "_" + str(frameIdx+1) + "/flow.flo"
			flow   = fl.readFlow(path)
			flowSc = cv.resize(flow, (FLOW_SIZE[1],FLOW_SIZE[0]))

			flowArray[srcIdx*NB_IMAGES+frameIdx-2,:,:,:] = flowSc


	# Partition the data into a training and verification set
	indices = np.random.permutation(flowArray.shape[0])
	training_idx, test_idx = indices[:int(flowArray.shape[0]*.8)], indices[int(flowArray.shape[0]*.8):]

	return (flowArray[training_idx, :, :, :], labelArray[training_idx]), (flowArray[test_idx, :, :, :], labelArray[test_idx])


# Convert class number to a vector. For example 4 => [0 0 0 1 0 0 ] ;  2 => [0 1 0 0 0 0 ] 
def to_categorical(labels_int):
	sz = len(np.unique(labels_int))
	cat = np.zeros((len(labels_int)*sz), dtype=int)
	idx = np.arange(labels_int.shape[0])*sz + (labels_int-1)
	cat[idx] = 1

	cat = np.reshape(cat, (len(labels_int),sz))

	return cat



def useDNN():

	# Load data
	(train_images, train_labels), (test_images, test_labels) = load_data()


	# Reshape the data to a single vector
	dimData = np.prod(train_images.shape[1:])
	train_data = train_images.reshape(train_images.shape[0], dimData)
	test_data  = test_images.reshape(test_images.shape[0], dimData)


	# Change the labels from integer to categorical data
	nClasses = len(np.unique(train_labels))
	train_labels_one_hot = to_categorical(train_labels)
	test_labels_one_hot = to_categorical(test_labels)
	

	# Build the model (simple fully connected neural network + dropout regularization)
	model = Sequential()
	model.add(Dense(32, activation='relu', input_shape=(dimData,)))
	model.add(Dropout(0.5))
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nClasses, activation='softmax'))

	model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


	# Train it
	history = model.fit(train_data, train_labels_one_hot, batch_size=256, epochs=30, verbose=1, validation_data=(test_data, test_labels_one_hot))


	# Evaluate it
	[test_loss, test_acc] = model.evaluate(test_data, test_labels_one_hot)
	print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


	# Save the model so it can be used later on
	model.save_weights('./model_weights.h5', overwrite=True)

	return history




def useCNN():

	# Load data
	(train_images, train_labels), (test_images, test_labels) = load_data()


	# Change the labels from integer to categorical data
	nClasses = len(np.unique(train_labels))
	train_labels_one_hot = to_categorical(train_labels)
	test_labels_one_hot = to_categorical(test_labels)


	img_rows, img_cols = train_images.shape[1], train_images.shape[2]
	input_shape = (img_rows, img_cols, 2)


	# Build the CNN neural network 
	model = Sequential()
	model.add(Conv2D(16, kernel_size=(3, 3),
						 padding='valid', 
						 activation='relu', 
						 input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(32, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nClasses, activation='softmax'))


	model.compile(optimizer=keras.optimizers.Adadelta(), loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])


	# Train it
	history = model.fit(train_images, train_labels_one_hot, batch_size=256, epochs=10, verbose=1, validation_data=(test_images, test_labels_one_hot))


	# Evaluate it
	[test_loss, test_acc] = model.evaluate(test_images, test_labels_one_hot)
	print("Evaluation result on Test Data : Loss = {}, accuracy = {}".format(test_loss, test_acc))


	# Save the model so it can be used later on
	model.save('./keras_model.h5', overwrite=True)


	return history




if __name__ == '__main__':


	# Train the model using a deep neural network
	history = useCNN()


	#Plot the Loss Curves
	plt.figure(figsize=[8,6])
	plt.plot(history.history['loss'],'r',linewidth=3.0)
	plt.plot(history.history['val_loss'],'b',linewidth=3.0)
	plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Loss',fontsize=16)
	plt.title('Loss Curves',fontsize=16)
	 
	#Plot the Accuracy Curves
	plt.figure(figsize=[8,6])
	plt.plot(history.history['acc'],'r',linewidth=3.0)
	plt.plot(history.history['val_acc'],'b',linewidth=3.0)
	plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Accuracy',fontsize=16)
	plt.title('Accuracy Curves',fontsize=16)

	plt.show()

	
