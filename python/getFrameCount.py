#!/usr/bin/env python3

import cv2
import sys
import os

if len(sys.argv) < 2:
	sys.exit("Not enough arguments. Usage getFrameCount <file> [<width=2048>] [<height=1024>]")

filename, file_extension = os.path.splitext(str(sys.argv[1]))


if file_extension != ".bin":

	capture  = cv2.VideoCapture(sys.argv[1])
	print(capture.get(cv2.CAP_PROP_FRAME_COUNT))

else:

	width = 2048
	height = 1024

	if(len(sys.argv) > 2):
		width = int(sys.argv[2])

	if(len(sys.argv) > 3):
		height = int(sys.argv[3])

	fsize = os.path.getsize(sys.argv[1])

	print(fsize/(width*height*4))

