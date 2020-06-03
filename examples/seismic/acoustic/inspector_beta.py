import numpy as np
from devito.logger import warning
from devito import TimeFunction, Function, Dimension, Eq, Inc
from devito import Operator, norm
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size
from matplotlib.pyplot import pause
import matplotlib.pyplot as plt
from devito.types.basic import Scalar, Symbol

# Some variable declarations
nx = 20
ny = 20
nz = 20
# Define a physical size
shape = (nx, ny, nz)  # Number of grid point (nx, nz)
spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)

# Init v field
v = np.empty(shape, dtype=np.float32)
v[:, :, :51] = 2
v[:, :, 51:] = 2

# Construct model
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=1, nbl=10)

t0 = 0  # Simulation starts a t=0
tn = 1000  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=3, time_range=time_range)


# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .45
src.coordinates.data[0, -1] = 15.  # Depth is 20m
src.coordinates.data[1, :] = np.array(model.domain_size) * .45
src.coordinates.data[1, -1] = 125.  # Depth is 20m
src.coordinates.data[2, :] = np.array(model.domain_size) * .45
src.coordinates.data[2, -1] = 105.  # Depth is 20m

u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
src_term = src.inject(field=u, expr=src)
op = Operator(src_term)

op(time=time_range.num-1)

# Get the nonzero indices
nzinds = np.nonzero(u.data[0])

shape = model.grid.shape
x, y, z = model.grid.dimensions
time = model.grid.time_dim

source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y, z),
                       dtype=np.int32)
source_id = Function(name='source_id', grid=model.grid, dtype=np.int32,
                     space_order=1)

source_id.data[nzinds[0], nzinds[1], nzinds[2]] = tuple(np.arange(1, len(nzinds[0])+1))
source_mask.data[nzinds[0], nzinds[1], nzinds[2]] = 1

# import pdb; pdb.set_trace()

print("Number of unique affected points is:", len(nzinds[0]))
assert(source_id.data[nzinds[0][0], nzinds[1][0], nzinds[2][0]] == 1)
assert(source_id.data[nzinds[0][-1], nzinds[1][-1], nzinds[2][-1]] == len(nzinds[0]))
assert(source_id.data[nzinds[0][len(nzinds[0])-1], nzinds[1][len(nzinds[0])-1],
       nzinds[2][len(nzinds[0])-1]] == len(nzinds[0]))

warning("---Source_mask and source_id is built here-------")

nnz_shape = (model.grid.shape[0], model.grid.shape[1])  # Change only 3rd dim

nnz_sp_source_mask = TimeFunction(name='nnz_sp_source_mask', shape=([1] + list(shape[:2])), dimensions=(time, x, y), time_order=0, dtype=np.int32)
nnz_sp_source_mask.data[0, :, :] = source_mask.data.sum(2)
inds = np.where(source_mask.data == 1)

#  = nnz_sp_source_mask.data[:,:].max()
maxz = len(np.unique(inds[2]))
sparse_shape = (model.grid.shape[0], model.grid.shape[1], maxz)  # Change only 3rd dim

assert(len(nnz_sp_source_mask.dimensions) == 3)

sp_source_mask = TimeFunction(name='sp_source_mask', shape=([1] + list(sparse_shape)),
                          dimensions=(time, x, y, z), time_order=0, dtype=np.int32)

# Now holds IDs
sp_source_mask.data[0, inds[0], inds[1], :] = tuple(inds[2][:len(np.unique(inds[2]))])

assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
assert(len(sp_source_mask.dimensions) == 4)

# Note:sparse_source_id is not needed as long as sparse info is kept in mask
# sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

id_dim = Dimension(name='id_dim')

save_src = TimeFunction(name='save_src', grid=model.grid, shape=(src.shape[0],
                        nzinds[1].shape[0]), dimensions=(src.dimensions[0], id_dim))

src_term = src.inject(field=save_src[src.dimensions[0], source_id], expr=src)

op1 = Operator([src_term])
op1.apply()


u2 = TimeFunction(name="u2", grid=model.grid, time_order=2)
sp_zdim = Dimension(name='sp_zdim')

# zind = TimeFunction(name='zind', shape=(2, 1, 1, 1), dimensions=u.dimensions, dtype=np.int32)
zind = TimeFunction(name="zind", shape=(time_range.num, 40), dimensions=(time, z), time_order=0, dtype=np.int32)


source_mask_f = TimeFunction(name='source_mask_f', grid=model.grid, time_order=0, dtype=np.int32)

source_mask_f.data[0, :, :, :] = source_mask.data[:, :, :] 



eq0 = Eq(u2.forward, u2)
# eq1 = Eq(zind[0, 0], nnz_sp_source_mask[0, x, y], implicit_dims=(time, x, y, z))
eq1 = Eq(zind, source_id, implicit_dims=(time, x, y, z))
eq2 = Inc(u2.forward, source_mask * save_src[time, zind] )
#eq0 = Eq(zind, nnz_sp_source_mask, implicit_dims=sp_zdim)


op2 = Operator([eq1, eq2])
op2.apply()
print(op2.ccode)
assert( norm(u) == norm(u2))

print(norm(u))
print(norm(u2))
assert(norm(u)==norm(u2))
# Unique z positions or unique x,y pairs?

# c- land should have

# (Pdb) source_mask.data[19,19,zind] * save_src.data[source_id.data[19, 19,zind], 3]
