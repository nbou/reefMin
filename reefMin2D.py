#!/usr/bin/python
# Script to estimate the reef structure underneath the corals. Used to help close the meshes of individual colonies
# extracted from a reef record (e.g. for the Palau data).

from osgeo import gdal
import numpy as np
# import cv2
import matplotlib.pyplot as plt
# import os
# import scipy.ndimage
# import time
from itertools import islice,product
# from scipy.signal import find_peaks
#
from mpl_toolkits.mplot3d import Axes3D

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

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
MAKSpth = '/home/nader/scratch/palau/Pats_01_Colonies/segmented/coral_binary_mask.tif'
ds = gdal.Open(DEMpth)
RB = ds.GetRasterBand(1)
dmat = np.array(RB.ReadAsArray())

ds = gdal.Open(MAKSpth)
RB = ds.GetRasterBand(1)
mask = np.array(RB.ReadAsArray())
# cv2.imshow("Depth", depth)
# plt.imshow(depth)

# make areas of no data 10m above the surface so they don't interfere with the minimum filter
depth_log = dmat==dmat[0][0]
depth_log = depth_log.astype(int)*(dmat[0][0]-10)
depth = dmat-depth_log
# plt.imshow(depth)
# plt.colorbar()
# plt.show()

rowN = 2000
dline = depth[:][rowN]
Mline = mask[:][rowN]

minz_100 = np.load('min_filter_100.npy')
minz_200 = np.load('min_filter_200.npy')
mline100 = minz_100[:][rowN]
mline200 = minz_200[:][rowN]
#
# # peaks = find_peaks(-dline,distance=100)
#
# # tst = []
# # wind = window(dline,300)
# # for w in wind:
# #     if w[0]==10:
# #         tst.append(10)
# #     else:
# #         tst.append(min(w))
# plt.plot(dline)
# plt.plot(mline100)
# plt.plot(mline200)
# # plt.plot(tst)
# # plt.plot(peaks[0],dline[peaks[0]])
# plt.plot(Mline)
#
#
# plt.show()
#
# plt.imshow(((depth - minz_100)<0.01).astype(int))
# plt.colorbar()
# plt.show()

# generate training points for GP
depth_log = dmat==dmat[0][0]
depth_log = depth_log.astype(int)*(-10)
depth=depth-depth_log
# plt.imshow(depth)
# plt.show()
x0,x01 = np.where(depth==minz_100)
# plt.scatter(x0,x01)
# plt.show()

print(len(x0))
X = np.empty((len(x0),2), int)
y = np.zeros((len(x0)))
for i in range(len(x0)):
    X[i]= [int(x0[i]),int(x01[i])]
    y[i] = depth[X[i][0]][X[i][1]]


# Input space
rs = 100
x1 = np.linspace(X[:,0].min(), X[:,0].max(),num=rs) #p
x2 = np.linspace(X[:,1].min(), X[:,1].max(),num=rs) #q
x = (np.array([x1, x2])).T

kernel = C(1.0, (1e-3, 1e3)) * RBF([100,100], (1e-7, 1e7))
gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)

gp.fit(X, y)

x1x2 = np.array(list(product(x1, x2)))
y_pred, MSE = gp.predict(x1x2, return_std=True)

X0p, X1p = x1x2[:,0].reshape(rs,rs), x1x2[:,1].reshape(rs,rs)
Zp = np.reshape(y_pred,(rs,rs))

# alternative way to generate equivalent X0p, X1p, Zp
# X0p, X1p = np.meshgrid(x1, x2)
# Zp = [gp.predict([(X0p[i, j], X1p[i, j]) for i in range(X0p.shape[0])]) for j in range(X0p.shape[1])]
# Zp = np.array(Zp).T

fig = plt.figure(figsize=(10,8))
# ax = fig.add_subplot(111)
# pcm = ax.pcolormesh(X0p, X1p, Zp)
# ax.invert_yaxis()
# fig.colorbar(pcm, ax=ax)

# plt.scatter(x0,x1)
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X0p, X1p, Zp, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
# plt.scatter(x0,x01)
# plt.show()
plt.figure()
plt.imshow(depth)
plt.colorbar()
plt.scatter(x0,x01)
# plt.show()


predIm = np.zeros(np.shape(depth))
it = 0
for el in x1x2:
    x_c,y_c = np.round(el).astype(int)
    predIm[x_c][y_c] = y_pred[it]
    it+=1

plt.figure()
plt.imshow(predIm)
plt.show()

# plt.imshow(dmat)
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