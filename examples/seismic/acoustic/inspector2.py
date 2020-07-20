from examples.cfd import plot_field, init_hat
import numpy as np
from devito import Grid, TimeFunction, Function, Eq, solve
from sympy.abc import a
from devito import Operator, Constant, norm, ConditionalDimension
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model, plot_velocity
import matplotlib as mpl
import matplotlib.pyplot as plt
from examples.seismic import plot_image
from scipy.sparse import csr_matrix
import sys
from sympy import And


# Some variable declarations
nx = 20
ny = 20
nz = 20

# Define a physical size
shape = (nx, ny, nz)  # Number of grid point (nx, nz)
spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)  # What is the location of the top left corner. This is necessary to define
# the absolute location of the source and receivers

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points

# Define a velocity profile. The velocity is in km/s
v = np.empty(shape, dtype=np.float32)
v[:, :, :51] = 2
v[:, :, 51:] = 2

# With the velocity and model size defined, we can create the seismic model that
# encapsulates this properties. We also define the size of the absorbing layer as 10 grid points
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing,
              space_order=2, nbl=10)

t0 = 0  # Simulation starts a t=0
tn = 1000  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)
# print ("steps")

f0 = 0.010 # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=2, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .48
src.coordinates.data[0, -1] = 18.  # Depth is 20m
src.coordinates.data[1, :] = np.array(model.domain_size) * .48
src.coordinates.data[1, -1] = 28.  # Depth is 20m

print(src.coordinates.data)
# src.show()

u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)

# We can plot the time signature to see the wavelet

src_term = src.inject(field=u, expr=src)

op = Operator(src_term)
op(time=time_range.num-1)
#print ("After another", tn, "timesteps")
# plot_field(u.data[0][0], zmax=4.5)
# print(u.data[0])

print(norm(u))

nzinds = np.nonzero(u.data[0])
print(nzinds)
#import pdb; pdb.set_trace()

source_mask = Function(name='source_mask', grid=model.grid, dtype=np.int32)
source_mask = Function(name='source_mask', grid=model.grid, dtype=np.int32)
source_id = Function(name='source_id', grid=model.grid, dtype=np.int32)



# source_mask = np.zeros(u.data[0].shape, dtype=np.uint8)
#source_id = np.zeros(u.data[0].shape, dtype=np.uint8)

source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(len(nzinds[0])))
# source_id[nzinds[0][:], nzinds[1][:], nzinds[2][:]] = tuple(np.arange(len(nzinds[0])))
source_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1.

#for k in range(len(nzinds[0])):
    # print(k)
    # source_id[nzinds[0][k], nzinds[1][k], nzinds[2][k]] = k
    # print(source_id[nzinds[0][k], nzinds[1][k], nzinds[2][k]])
    # source_mask[nzinds[0][k], nzinds[1][k], nzinds[2][k]] = 1
    # print(source_mask[nzinds[0][k], nzinds[1][k], nzinds[2][k]])

print ("Number of unique affected points is:", len(nzinds[0]))
assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]]==0)
assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]]==len(nzinds[0])-1)
assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1], nzinds[2][len(nzinds[0])-1]]==len(nzinds[0])-1)

print("---Create functions-------")

spGrid = Grid(shape=(source_mask.shape[0],source_mask.shape[1],len(nzinds[0])))

# sparse_source_mask = Function(name='sparse_source_mask', grid=spGrid, dtype=np.int32)

sparse_source_id = Function(name='sparse_source_id', grid=spGrid, dtype=np.int32)

sparse_source_mask_NNZ = Function(name='sparse_source_mask_NNZ', grid=Grid(shape=(source_mask.shape[0],source_mask.shape[1])), dtype=np.int32)


import pdb; pdb.set_trace()

ci = ConditionalDimension(name='ci', parent=source_mask.dimensions[2], condition=Eq(source_mask[source_mask.dimensions[0],source_mask.dimensions[1],source_mask.dimensions[2]], 0))


sparse_source_mask = Function(name='sparse_source_mask', shape=(source_mask.shape[0],source_mask.shape[1],len(nzinds[0])) , dimensions=(source_mask.dimensions[0],source_mask.dimensions[1],ci), dtype=np.int32)

op = Operator([Eq(sparse_source_mask, 1)]).apply()
import pdb; pdb.set_trace()



assert(source_id.shape>sparse_source_id.shape)
assert(source_mask.shape>sparse_source_mask.shape)


# assert(np.count_nonzero(sparse_source_id - np.diag(np.diagonal(sparse_source_mask))))
assert(source_id.shape>sparse_source_id.shape)

import pdb; pdb.set_trace()


sparse_source_id[nzinds[0],nzinds[1],:]=source_id[nzinds[0], nzinds[1], nzinds[2]]


save_src = np.zeros((len(nzinds[0]),time_range.num-1), dtype=np.uint32)

# sparse_source_mask = sparse.COO(nzinds, mydata, shape=((44,) * 3))

#sparse_src_mask = Function(name="sid", grid=Grid(), time_order=2, space_order=2)
# smask = TimeFunction(name="smask", grid=model.grid, time_order=2, space_order=2)

np.set_printoptions(threshold=sys.maxsize)