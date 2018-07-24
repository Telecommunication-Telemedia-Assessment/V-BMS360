#!/bin/python

import cv2 as cv
import sys

if len(sys.argv) != 4:
	sys.exit("Not enough arguments. Usage getFrame <file> <frameIdx> <out file>")

capture  = cv.VideoCapture(str(sys.argv[1]))
frameIdx = int(str(sys.argv[2])) 


for i in range(0, frameIdx):
	ret, frame = capture.read()

	
# cv.imshow('frame',frame)
# cv.waitKey()


cv.imwrite(str(sys.argv[3]), frame)