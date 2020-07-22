from devito import *
from examples.seismic.source import WaveletSource, RickerSource, GaborSource, TimeAxis
from examples.seismic import plot_image
import numpy as np

import numpy as np

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt
import argparse

from devito.logger import info
from devito import TimeFunction, Function, Dimension, Eq, Inc, solve
from devito import Operator, norm
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

from devito.types.basic import Scalar, Symbol # noqa
from mpl_toolkits.mplot3d import Axes3D # noqa


def plot3d(data, model):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    z, x, y = data.nonzero()
    ax.scatter(x, y, z, zdir='y', c='red', s=20, marker='.')
    ax.set_xlim(model.spacing[0], data.shape[0]-model.spacing[0])
    ax.set_ylim(model.spacing[1], data.shape[1]-model.spacing[1])
    ax.set_zlim(model.spacing[2], data.shape[2]-model.spacing[2])
    plt.savefig("sources_demo.pdf")

parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(11, 11, 11), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-tn", "--tn", default=40,
                    type=float, help="Simulation time in millisecond")

args = parser.parse_args()

# --------------------------------------------------------------------------------------


nx, ny, nz = args.shape
# Define a physical size
shape = (nx, ny, nz)  # Number of grid point (nx, nz)
spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)
so = args.space_order

# Initial grid: 1km x 1km, with spacing 100m
extent = (1500., 1500., 1500)

x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=extent[1]/(shape[1]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[2]/(shape[2]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, y, z))


class DGaussSource(WaveletSource):

    def wavelet(self, f0, t):
        a = 0.004
        return -2.*a*(t - 1/f0) * np.exp(-a * (t - 1/f0)**2)


# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., 300.
dt = (10. / np.sqrt(2.)) / 6.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
src.coordinates.data[:] = [750., 750., 750]
# src.show()

# Now we create the velocity and pressure fields

v = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
tau = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)

# Now let's try and create the staggered updates
t = grid.stepping_dim
time = grid.time_dim

# We need some initial conditions
V_p = 2.0
V_s = 1.0
density = 1.8

# The source injection term
src_xx = src.inject(field=tau.forward[0, 0], expr=src)
src_yy = src.inject(field=tau.forward[1, 1], expr=src)
src_zz = src.inject(field=tau.forward[2, 2], expr=src)

# Thorbecke's parameter notation
cp2 = V_p*V_p
cs2 = V_s*V_s
ro = 1/density

mu = cs2*density
l = (cp2*density - 2*mu)

# fdelmodc reference implementation
u_v = Eq(v.forward, v + dt*ro*div(tau))
u_t = Eq(tau.forward, tau + dt * l * diag(div(v.forward)) + dt * mu * (grad(v.forward) + grad(v.forward).T))
op = Operator([u_v] + [u_t]  + src_xx + src_yy + src_zz)
# op = Operator(src_xx + src_yy + src_zz)
op()

# plot_image(v[0].data[0,int(nx/2),:,:], cmap="seismic"); pause(1)
# plot_image(v[1].data[0,int(nx/2),:,:], cmap="seismic"); pause(1)
# plot_image(v[2].data[0,int(nx/2),:,:]); pause(1)

# plot_image(tau[0].data[0,int(nx/2),:,:], cmap="seismic"); pause(1)
# plot_image(tau[0,1].data[0,int(nx/2),:,:], cmap="seismic"); pause(1)
# plot_image(tau[0,1].data[0,int(nx/2),:,:]); pause(1)
# plot_image(tau[0].data[0,int(nx/2),:,:]); pause(1)


norm_v0 = norm(v[0])
norm_v1 = norm(v[1])
norm_v2 = norm(v[2])

print(norm_v0)
print(norm_v1)
print(norm_v2)

norm_t0 = norm(tau[0])
norm_t1 = norm(tau[1])
norm_t2 = norm(tau[2])

print(norm_t0)
print(norm_t1)
print(norm_t2)


print("Let's inspect")
# import pdb; pdb.set_trace()

# f : perform source injection on an empty grid
# Inspection The source injection term
ftau = TensorTimeFunction(name='ftau', grid=grid, space_order=so, time_order=1)

src_fxx = src.inject(field=ftau.forward[0, 0], expr=src)
src_fyy = src.inject(field=ftau.forward[1, 1], expr=src)
src_fzz = src.inject(field=ftau.forward[2, 2], expr=src)

# op_f = Operator([src_f], opt=('advanced', {'openmp': True}))
op_f = Operator(src_fxx + src_fyy + src_fzz)
op_f()

normf = norm(ftau[0])
normf = norm(ftau[1])
normf = norm(ftau[2])

print("==========")
print(norm(ftau[0]))
print(norm(ftau[1]))
print(norm(ftau[2]))
print("===========")


import pdb; pdb.set_trace()

#Get the nonzero indices
nzinds = np.nonzero(f.data[0])  # nzinds is a tuple
assert len(nzinds) == len(shape)

x, y, z = grid.dimensions
time = grid.time_dim
source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.float32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y, z), space_order=0, dtype=np.int32)
info("source_id data indexes start from 1 not 0 !!!")

# source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(1, len(nzinds[0])+1))
source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(len(nzinds[0])))

source_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1
# plot3d(source_mask.data, model)

info("Number of unique affected points is: %d", len(nzinds[0])+1)

# Assert that first and last index are as expected
assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 0)
assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0])-1)
assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1],
       nzinds[2][len(nzinds[0])-1]] == len(nzinds[0])-1)

assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(source_mask.data)))
assert(np.all(np.nonzero(source_id.data)) == np.all(np.nonzero(f.data[0])))

info("-At this point source_mask and source_id have been popoulated correctly-")

nnz_shape = (model.grid.shape[0], model.grid.shape[1])  # Change only 3rd dim

nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=(list(nnz_shape)), dimensions=(x, y), space_order=0, dtype=np.int32)


nnz_sp_source_mask.data[:, :] = source_mask.data[:, :, :].sum(2)
inds = np.where(source_mask.data == 1.)

maxz = len(np.unique(inds[2]))
sparse_shape = (model.grid.shape[0], model.grid.shape[1], maxz)  # Change only 3rd dim

assert(len(nnz_sp_source_mask.dimensions) == 2)

# Note:sparse_source_id is not needed as long as sparse info is kept in mask
# sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

id_dim = Dimension(name='id_dim')
b_dim = Dimension(name='b_dim')

save_src = TimeFunction(name='save_src', shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

save_src_term = src.inject(field=save_src[src.dimensions[0], source_id], expr=src * dt**2 / model.m)

op1 = Operator([save_src_term])
op1.apply(time=time_range.num-1)


usol = TimeFunction(name="usol", grid=model.grid, space_order=so, time_order=2)

sp_zi = Dimension(name='sp_zi')

# import pdb; pdb.set_trace()

sp_source_mask = Function(name='sp_source_mask', shape=(list(sparse_shape)), dimensions=(x, y, sp_zi), space_order=0, dtype=np.int32)

# Now holds IDs
sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[2][:len(np.unique(inds[2]))])

assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
assert(len(sp_source_mask.dimensions) == 3)
