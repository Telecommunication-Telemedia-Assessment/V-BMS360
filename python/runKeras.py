import numpy as np
import sys
import cv2 as cv
import readFlo as fl

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D




def showFlo(flow):
	motion1 = np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:,:,1]))

	cv.imshow("motion1", motion1 / np.max(motion1[:]))
	cv.waitKey()



def predict(flow):
	FLOW_SIZE = [45, 80]
	nClasses = 3
	input_shape = (FLOW_SIZE[0], FLOW_SIZE[1], 2)

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
	model.load_weights('./keras_model.h5')




	
	flowArray  = np.ndarray(shape=(1, FLOW_SIZE[0], FLOW_SIZE[1], 2), dtype=float, order='F')
	flowArray[0,:,:,:] = flow

	print(model.predict(flow[None,:,:,:]))






data2D1 = fl.readFlow('motion.flo')

predict(data2D1)
