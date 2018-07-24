import numpy as np
import os


# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def readFlow(fileName) :
    with open(fileName, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))

            # print 'Reading %d x %d flo file' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h, w, 2))

            return data2D
