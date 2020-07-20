import numpy as np

from matplotlib.pyplot import pause # noqa
import matplotlib.pyplot as plt

from devito.logger import info
from devito import TimeFunction, Function, Dimension, Eq, Inc, configuration, solve
from devito import Operator, norm
from examples.seismic import RickerSource, TimeAxis, plot_velocity
from examples.seismic import Model
import sys
np.set_printoptions(threshold=sys.maxsize)  # pdb print full size

from devito.types.basic import Scalar, Symbol # noqa
from mpl_toolkits.mplot3d import Axes3D # noqa


# Some variable declarations

import pdb; pdb.set_trace()

nx, ny, nz = 10, 10, 10
# Define a physical size
shape = (nx, ny, nz)  # Number of grid point (nx, nz)
spacing = (10., 10., 10)  # Grid spacing in m. The domain size is now 1km by 1km
origin = (0., 0., 0.)
so = 8
# Initialize v field
v = np.empty(shape, dtype=np.float32)
v[:, :, :int(nz/2)] = 2
v[:, :, int(nz/2):] = 1

# Construct model
model = Model(vp=v, origin=origin, shape=shape, spacing=spacing, space_order=so,
              nbl=10, bcs="damp")

# plt.imshow(model.vp.data[10, :, :]) ; pause(1)

t0 = 0  # Simulation starts a t=0
tn = 20  # Simulation last 1 second (1000 ms)
dt = model.critical_dt  # Time step from model grid spacing
time_range = TimeAxis(start=t0, stop=tn, step=dt)
f0 = 0.010  # Source peak frequency is 10Hz (0.010 kHz)
src = RickerSource(name='src', grid=model.grid, f0=f0,
                   npoint=3, time_range=time_range)


# First, position source centrally in all dimensions, then set depth
src.coordinates.data[0, :] = np.array(model.domain_size) * .2
src.coordinates.data[0, -1] = 20  # Depth is 20m
src.coordinates.data[1, :] = np.array(model.domain_size) * .2
src.coordinates.data[1, -1] = 20  # Depth is 20m
src.coordinates.data[2, :] = np.array(model.domain_size) * .2
src.coordinates.data[2, -1] = 20  # Depth is 20m

# src.show()
# pause(1)

# f : perform source injection on an ampty grid
f = TimeFunction(name="f", grid=model.grid, space_order=so, time_order=2)
src_f = src.inject(field=f.forward, expr=src * dt**2 / model.m)
# op_f = Operator([src_f], opt=('advanced', {'openmp': True}))
op_f = Operator([src_f], opt=('advanced', {'openmp': True}))
op_f.apply(time=time_range.num-1)


# uref : reference solution
uref = TimeFunction(name="uref", grid=model.grid, space_order=so, time_order=2)
src_term_ref = src.inject(field=uref.forward, expr=src * dt**2 / model.m)
opref = Operator([src_term_ref])
opref.apply(time=time_range.num-1)


print("==========")
print(norm(f))
print("===========")

print("==========")
print(norm(uref))
print("===========")