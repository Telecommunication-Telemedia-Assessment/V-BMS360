import numpy as np
import sys
import cv2 as cv
import readFlo as fl


# Add two optical flow 
# If flow1 is the motion between A and B, and flow2 is the motion between B and C, warpFlow(flow1, flow2) return the motion between A and C
def warpFlow(flow1, flow2):
	yy = np.arange(flow1.shape[0])
	yy = np.transpose(np.kron(np.ones((flow1.shape[1],1)), yy))

	xx = np.arange(flow1.shape[1])
	xx = np.kron(np.ones((flow1.shape[0],1)), xx) 

	idxx = np.minimum(np.maximum(xx + flow1[:,:,0], 0), flow1.shape[1]-1)
	idxy = np.array(np.minimum(np.maximum(yy + flow1[:,:,1], 0), flow1.shape[0]-1), dtype=np.integer)

	idx_flat = np.array(np.reshape(idxx + idxy * flow1.shape[1], idxx.size) , dtype=np.integer)

	flowx = flow1[:,:,1] + np.reshape(np.reshape(flow2[:,:,1], idxx.size)[idx_flat], idxx.shape)
	flowy = flow1[:,:,0] + np.reshape(np.reshape(flow2[:,:,0], idxx.size)[idx_flat], idxx.shape)

	return np.dstack((flowx, flowy)) 


# Project an image using a dense optical flow.
# If flow is the motion between images A and B, warpImage(flow, A) return image B 
def warpImage(flow1, image):
	yy = np.arange(flow1.shape[0])
	yy = np.transpose(np.kron(np.ones((flow1.shape[1],1)), yy))

	xx = np.arange(flow1.shape[1])
	xx = np.kron(np.ones((flow1.shape[0],1)), xx) 

	idxx = np.minimum(np.maximum(xx + flow1[:,:,0], 0), flow1.shape[1]-1)
	idxy = np.array(np.minimum(np.maximum(yy + flow1[:,:,1], 0), flow1.shape[0]-1), dtype=np.integer)

	idx_flat = np.array(np.reshape(idxx + idxy * flow1.shape[1], idxx.size) , dtype=np.integer)

	image[:,:,0] = np.reshape(np.reshape(image[:,:,0], idxx.size)[idx_flat], idxx.shape)
	image[:,:,1] = np.reshape(np.reshape(image[:,:,1], idxx.size)[idx_flat], idxx.shape)
	image[:,:,2] = np.reshape(np.reshape(image[:,:,2], idxx.size)[idx_flat], idxx.shape)

	return image


# If flow is the motion between two images A and B, flowSphere return the position of source and destination of each point of the image in spherical coordinate
def flowSphere(flow1):
	# xyzSphere = np.zeros(flow.size, 6)

	yy = np.arange(flow1.shape[0])
	yy = np.transpose(np.kron(np.ones((flow1.shape[1],1)), yy))

	xx = np.arange(flow1.shape[1])
	xx = np.kron(np.ones((flow1.shape[0],1)), xx) 

	lng_src = 2*3.1415926535898 * (xx - xx.shape[1]/2) / xx.shape[1]
	lat_src =   3.1415926535898 * (yy - yy.shape[0]/2) / yy.shape[0]


	idxx = np.mod(xx + flow1[:,:,0], flow1.shape[1])
	idxy = np.mod(yy + flow1[:,:,1], flow1.shape[0])

	lng_dst = 2*3.1415926535898 * (idxx - xx.shape[1]/2) / xx.shape[1]
	lat_dst =   3.1415926535898 * (idxy - yy.shape[0]/2) / yy.shape[0]

	radius = 1
	x_src = radius * np.cos(lng_src) * np.sin(lat_src)
	y_src = radius * np.sin(lng_src) * np.sin(lat_src)
	z_src = radius * np.cos(lat_src)

	x_dst = radius * np.cos(lng_dst) * np.sin(lat_dst)
	y_dst = radius * np.sin(lng_dst) * np.sin(lat_dst)
	z_dst = radius * np.cos(lat_dst)

	return np.array([np.reshape(x_src, x_src.size), np.reshape(y_src, x_src.size), np.reshape(z_src, x_src.size), np.reshape(x_dst, x_src.size), np.reshape(y_dst, x_src.size), np.reshape(z_dst, x_src.size)])


if __name__ == '__main__':

	if len(sys.argv) > 2:
		

		data2D1 = fl.readFlow(sys.argv[1])
		data2D2 = fl.readFlow(sys.argv[2])

		# print np.min(data2D1[:,:,0]), np.max(data2D1[:,:,0])
		# print np.min(data2D1[:,:,1]), np.max(data2D1[:,:,1])

		# f0 = ((data2D1[:,:,0]-np.min(data2D1[:,:,0])) / (np.max(data2D1[:,:,0])-np.min(data2D1[:,:,0]))) 
		# f1 = ((data2D1[:,:,1]-np.min(data2D1[:,:,1])) / (np.max(data2D1[:,:,1])-np.min(data2D1[:,:,1])))

		# cv.imshow("0",f0)
		# cv.imshow("1", f1)
		# cv.waitKey()


		if len(sys.argv) > 4:
			image1 = cv.imread(sys.argv[3])
			image2 = cv.imread(sys.argv[4])

			image1 = cv.resize(image1, (data2D1.shape[1], data2D1.shape[0]))
			image2 = cv.resize(image2, (data2D1.shape[1], data2D1.shape[0]))


			cv.imshow("source", image1)
			cv.imshow("target", image2)

			print flowSphere(data2D1).shape
			
			imageW = warpImage(data2D1, image1)
			cv.imshow("image projected", imageW)
			cv.waitKey()



		else:

			data2D3 = warpFlow(data2D1, data2D2)

			csvData = np.transpose(flowSphere(data2D3))

			print csvData.shape

			np.savetxt("output.csv", csvData[1:121600,], delimiter=",")



			# # Apply a scaling on the amplitude of the motion to take into account the geometry of equirectangular images
			# scaling = np.cos( 3.1415926535898 * (np.arange(data2D1.shape[0])-data2D1.shape[0]/2) / data2D1.shape[0])
			# scaling = np.transpose(np.kron(np.ones((data2D1.shape[1],1)), scaling))

			# data2D1[:,:,0] = np.multiply(data2D1[:,:,0], scaling)
			# motion1 = np.sqrt(np.square(data2D1[:,:,0]) + np.square(data2D1[:,:,1]))


			# data2D2[:,:,0] = np.multiply(data2D2[:,:,0], scaling)
			# motion2 = np.sqrt(np.square(data2D2[:,:,0]) + np.square(data2D2[:,:,1]))

			# data2D3[:,:,0] = np.multiply(data2D3[:,:,0], scaling)
			# motion3 = np.sqrt(np.square(data2D3[:,:,0]) + np.square(data2D3[:,:,1]))


			# print np.min(motion1[:]), np.max(motion1[:])
			# print np.min(motion2[:]), np.max(motion2[:])
			# print np.min(motion3[:]), np.max(motion3[:])


			# cv.imshow("motion1", motion1 / np.max(motion1[:]))
			# cv.imshow("motion2", motion2 / np.max(motion2[:]))
			# cv.imshow("motion3", motion3 / np.max(motion3[:]))
			# cv.waitKey()

	else:

		print "You should provide the input files"


