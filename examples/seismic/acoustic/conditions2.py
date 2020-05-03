import numpy as np
from devito.logger import warning
from devito import TimeFunction, Function, Dimension
from devito import Operator
from examples.seismic import RickerSource, TimeAxis
from examples.seismic import Model
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

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
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=2, nbl=10)

t0 = 0  # Simulation starts a t=0
tn = 1000  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing

time_range = TimeAxis(start=t0, stop=tn, step=dt)

f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=2, time_range=time_range)

# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .48
src.coordinates.data[0, -1] = 18.  # Depth is 20m
src.coordinates.data[1, :] = np.array(model.domain_size) * .48
src.coordinates.data[1, -1] = 128.  # Depth is 20m

u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=2)
src_term = src.inject(field=u, expr=src)
op = Operator(src_term)
op(time=time_range.num-1)

nzinds = np.nonzero(u.data[0])
print(nzinds)

shape = model.grid.shape
x = Dimension(name='x')
y = Dimension(name='y')
z = Dimension(name='z')

source_mask = Function(name='source_mask', shape=shape, dimensions=(x, y, z),
                       dtype=np.int32)
source_id = Function(name='source_id', shape=shape, dimensions=(x, y, z), dtype=np.int32)

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
x = Dimension(name='x')
y = Dimension(name='y')
z = Dimension(name='z')

nnz_sp_source_mask = Function(name='nnz_sp_source_mask', shape=shape[:2],
                              dimensions=(x, y), dtype=np.int32)
nnz_sp_source_mask.data[:, :] = source_mask.data.sum(2)
inds = np.where(source_mask.data == 1)

#  = nnz_sp_source_mask.data[:,:].max()
maxz = len(np.unique(inds[2]))
sparse_shape = (model.grid.shape[0], model.grid.shape[1], maxz)  # Change only 3rd dim

assert(len(nnz_sp_source_mask.dimensions) == 2)

sp_source_mask = Function(name='sp_source_mask', shape=sparse_shape,
                          dimensions=(x, y, z), dtype=np.int32)

# Now holds IDs
sp_source_mask.data[inds[0], inds[1], :] = tuple(inds[2][:len(np.unique(inds[2]))])


assert(np.count_nonzero(sp_source_mask.data) == len(nzinds[0]))
assert(len(sp_source_mask.dimensions) == 3)

# Note:sparse_source_id is not needed as long as sparse info is kept in mask
# sp_source_id.data[inds[0],inds[1],:] = inds[2][:maxz]

# import pdb; pdb.set_trace()

id_dim = Dimension(name='id_dim')
save_src = Function(name='save_src', shape=(len(nzinds[0]), src.shape[0]),
                    dimensions=(id_dim, src.dimensions[0]), dtype=np.int32)


# excond=CondEq(source_mask[x,y,z],1)
# ci = ConditionalDimension(name='ci', parent=z, condition=excond)
# spzi = Function(name = 'spzi', shape=1, dimensions=z, dtype=np.int32)
# spzi = Constant(name = 'spzi', dtype=np.int32)
# source_mask.data
# Eq1 = Eq(sp_source_mask[x,y,spzi], z, implicit_dims=ci)
# Eq3 = Eq(nnz_sp_source_mask[x,y], nnz_sp_source_mask[x,y]+1, implicit_dims=ci)
# Eq4 = Eq(spzi , 0, implicit_dims=ci)
# Eq5 = Eq(spzi[z] , spzi[z]+1, implicit_dims=ci)

# op = Operator([Eq1, Eq3, Eq5])
# print(op.ccode)
# import pdb; pdb.set_trace()
