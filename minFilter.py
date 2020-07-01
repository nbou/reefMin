# Script to estimate the reef structure underneath the corals. Used to help close the meshes of individual colonies
# extracted from a reef record (e.g. for the Palau data).

from osgeo import gdal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import scipy.ndimage
import time
from itertools import islice



def min2D(data):
    if data[0] == 10 or data[-2]==10:
        minz = 10
    else:
        minz = np.min(data)
    return minz


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# read the DEM into a numpy array
DEMpth = '/home/nader/scratch/palau/Palau_DEMs/022_pats_dropoff_circle_01/pats_dropoff_022_circle01_DEM.tif'
ds = gdal.Open(DEMpth)
RB = ds.GetRasterBand(1)
depth = np.array(RB.ReadAsArray())
# cv2.imshow("Depth", depth)
# plt.imshow(depth)

# make areas of no data 10m above the surface so they don't interfere with the minimum filter
depth_log = depth==depth[0][0]
depth_log = depth_log.astype(int)*(depth[0][0]-10)
depth = depth-depth_log
plt.imshow(depth)
plt.colorbar()
plt.show()

filtsize = 100
tic = time.time()
minz = scipy.ndimage.filters.generic_filter(depth, min2D, size=filtsize)
toc = time.time()
print("Calculated min filter in {0} seconds".format(toc-tic))
plt.imshow(minz)
plt.show()
np.save('./min_filter_{}'.format(filtsize),minz)

#
# for ro in range(500,5000,500):
#     dline = depth[:][ro]
#     # plt.plot(dline)
#
#     for d in range(10,500,50):
#         tst = []
#         wind = window(dline,d)
#
#         for w in wind:
#             if w[0]==10:
#                 tst.append(10)
#             else:
#                 tst.append(min(w))
#
#         plt.plot(dline)
#         plt.plot(tst)
#         plt.pause(0.2)
#         plt.cla()

# dline = depth[:][2000]
#
# #
#
# tst = []
# wind = window(dline,300)
# for w in wind:
#     if w[0]==10:
#         tst.append(10)
#     else:
#         tst.append(min(w))
# plt.plot(dline)
# plt.plot(tst)
# plt.show()
# print(len(tst),len(dline))

# window_size = 300
# min_img = np.zeros((1,np.shape(depth)[1]-window_size+1))
# for ro in range(0,np.shape(depth)[0]):
#     dline = depth[:][ro]
#     # plt.plot(dline)
#
#     tst = []
#     wind = window(dline,window_size)
#     for w in wind:
#       tst.append(min(w))
#     tst = np.reshape(np.array(tst),(1,np.shape(depth)[1]-window_size+1))
#     min_img = np.vstack((min_img,tst))
#         # plt.plot(dline)
#         # plt.plot(tst)
#         # plt.pause(0.2)
#         # plt.cla()
#
# plt.imshow(min_img)
# plt.show()