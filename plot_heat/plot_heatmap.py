import csv
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from functools import reduce

# my_data = genfromtxt('my_file.csv', delimiter=',')
file = open("elastic_so8_600x_600y_600z_t600_v100.csv")
performance_map = np.loadtxt(file, delimiter=",")

data = np.delete(performance_map, (3, 4, 5, 6, 8), axis=1) # delete not needed columns

bx_data = np.unique(data[:, 0])
bx_data = bx_data.astype(int)
by_data = np.unique(data[:, 1])
by_data = by_data.astype(int)
bz_data = np.unique(data[:, 2])
bz_data = bz_data.astype(int)
xindexing = np.zeros((len(bx_data), 1), dtype=int)
yindexing = np.zeros((len(by_data), 1), dtype=int)
zindexing = np.zeros((len(bz_data), 1), dtype=int)

xindexing = bx_data
yindexing = by_data
zindexing = bz_data

data3 = np.zeros((len(bx_data), len(by_data), len(bz_data)))
eps = np.spacing(0.0)
vmin=eps
data3[:,:,:]= -1

cmap = plt.get_cmap('rainbow')
cmap.set_under('white')


for i in range(len(data)):
    ids = data[i,0:3].astype(int)
    value = data[i,3]
    map = [np.where(xindexing==ids[0]),np.where(yindexing==ids[1]),np.where(zindexing==ids[2])]
    data3[map[0], map[1], map[2]] = value


# data3 contains the data in 3d format

tids = np.unique(data[:, 0])


Tot = data3.shape[0]  # number_of_subplots
Cols = 2  # number_of_columns
Rows = Tot // Cols 
Rows += Tot % Cols
Position = range(1,Tot + 1)


fig = plt.figure(1, figsize=(6,8))
for slice in range(data3.shape[0]):

    # add every single subplot to the figure with a for loop

    ax = fig.add_subplot(Rows, Cols, Position[slice])
    import pdb;pdb.set_trace()

    data2d = data3[slice, :, :]
    im = ax.imshow(data2d, cmap=cmap, vmin=eps, vmax=np.amax(data3))  # Or whatever you want in the subplot

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(by_data)))
    ax.set_ylabel("y tile")
    ax.set_yticks(np.arange(len(bz_data)))
    ax.set_xlabel("z tile")

    # ... and label them with the respective list entries
    ax.set_xticklabels(by_data)
    ax.set_yticklabels(bz_data)
    ax.invert_yaxis()
    ax.set_title("Slice tile(%d,:,:)" % bx_data[slice])
    fig.tight_layout()

import pdb;pdb.set_trace()
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.suptitle("Gpts/s *ac_so24 section0<403,768,768,768>.", fontsize=14)


hc = fig.colorbar(im, cmap=cmap, cax=cbar_ax)
minl = 0
maxl = np.amax(data3)
hc.set_ticks([minl, maxl])
plt.show();plt.pause(3)
print(data3)
