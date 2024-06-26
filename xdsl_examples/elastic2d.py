#!/usr/bin/env python
# coding: utf-8

# # Elastic wave equation implementation on a staggered grid
# This is a first attempt at implemenenting the elastic wave equation as described in:
#
# [1] Jean Virieux (1986). ”P-SV wave propagation in heterogeneous media:
# Velocity‐stress finite‐difference method.” GEOPHYSICS, 51(4), 889-901.
# https://doi.org/10.1190/1.1442147
#
# The current version actually attempts to mirror the FDELMODC implementation
# by Jan Thorbecke:
# [2] https://janth.home.xs4all.nl/Software/fdelmodcManual.pdf
#
# ## Explosive source
#
# We will first attempt to replicate the explosive source test case described in [1],
# Figure 4. We start by defining the source signature $g(t)$, the derivative of a
# Gaussian pulse, given by Eq 4:
#
# $$g(t) = -2 \alpha(t - t_0)e^{-\alpha(t-t_0)^2}$$

import argparse
import numpy as np
import matplotlib.pyplot as plt
from devito import (Grid, TensorTimeFunction, VectorTimeFunction, div, grad, diag, solve,
                    Operator, Eq, Constant, norm, SpaceDimension)
from devito.tools import as_tuple

from examples.seismic.source import WaveletSource, RickerSource, TimeAxis
from examples.seismic import plot_image

# flake8: noqa

from sympy import init_printing
init_printing(use_latex='mathjax')


parser = argparse.ArgumentParser(description='Process arguments.')

parser.add_argument("-d", "--shape", default=(110, 110), type=int, nargs="+",
                    help="Number of grid points along each axis")
parser.add_argument("-so", "--space_order", default=2,
                    type=int, help="Space order of the simulation")
parser.add_argument("-nt", "--nt", default=40,
                    type=int, help="Simulation time in millisecond")
parser.add_argument("-devito", "--devito", default=False, type=bool, help="Devito run")
parser.add_argument("-xdsl", "--xdsl", default=False, type=bool, help="xDSL run")
parser.add_argument("-plot", "--plot", default=False, type=bool, help="Plot2D")

args = parser.parse_args()

# Some variable declarations
nt = args.nt
so = args.space_order
to = 1


# Initial grid: km x km, with spacing
shape = args.shape  # Number of grid point (nx, nz)
spacing = as_tuple(10.0 for _ in range(len(shape)))  # Grid spacing in m. The domain size is now 1km by 1km

extent = tuple([s*(n-1) for s, n in zip(spacing, shape)])

x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))
grid = Grid(extent=extent, shape=shape, dimensions=(x, z))

# Timestep size from Eq. 7 with V_p=6000. and dx=100
t0, tn = 0., nt
dt = (10. / np.sqrt(2.)) / 6.
time_range = TimeAxis(start=t0, stop=tn, step=dt)

src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
src.coordinates.data[:] = [int(extent[0]/2), int(extent[1]/2)]

# Plot source
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
src_zz = src.inject(field=tau.forward[1, 1], expr=src)

# Thorbecke's parameter notation
cp2 = V_p*V_p
cs2 = V_s*V_s
ro = 1/density

mu = cs2*density
l = (cp2*density - 2*mu)

# First order elastic wave equation
pde_v = v.dt - ro * div(tau)
pde_tau = (tau.dt - l * diag(div(v.forward)) - mu * (grad(v.forward) +
           grad(v.forward).transpose(inner=False)))

# Time update
u_v = Eq(v.forward, solve(pde_v, v.forward))
u_t = Eq(tau.forward, solve(pde_tau, tau.forward))

# This contains if conditions!!!
# We use it to preinject data
op = Operator([u_v] + [u_t] + src_xx + src_zz)
op(dt=dt)


v_init = v[0].data[:, :], v[1].data[:, :]
tau_init = tau[0].data[:, :], tau[1].data[:, :], tau[2].data[:, :]

# Up to here, we have the same code as in the Devito example
# assert np.isclose(norm(v[0]), 0.6285093, atol=1e-4, rtol=0)

# This should NOT have conditions, we should use XDSL!

if args.xdsl:
    op = Operator([u_v] + [u_t], opt='xdsl')
    op(dt=dt, time_M=nt)
    print("norm v0:", norm(v[0]))
    print("norm tau0:", norm(tau[0]))

if args.devito:
    op = Operator([u_v] + [u_t], opt='advanced')
    op(dt=dt, time_M=nt)
    print("norm v0:", norm(v[0]))
    print("norm tau0:", norm(tau[0]))

if args.plot:
    # Save the plotted images locally
    plt.imsave('v0.pdf', v[0].data_with_halo[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
    plt.imsave('v1.pdf', v[1].data_with_halo[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
    plt.imsave('tau00.pdf', tau[0, 0].data_with_halo[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
    plt.imsave('tau11.pdf', tau[1, 1].data_with_halo[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")
    plt.imsave('tau01.pdf', tau[0, 1].data_with_halo[0], vmin=-.5*1e-2, vmax=.5*1e-2, cmap="seismic")


# Store the results
v_xdsl = v[0].data[:, :], v[1].data[:, :]
tau_xdsl = tau[0].data[:, :], tau[1].data[:, :], tau[2].data[:, :]

# Reset the initial conditions
v[0].data[:, :], v[1].data[:, :] = v_init[0][:, :], v_init[1][:, :]
tau[0].data[:, :], tau[1].data[:, :], tau[2].data[:, :] = (
    tau_init[0][:, :],
    tau_init[1][:, :],
    tau_init[2][:, :],
)

# Now we run the same code with the Devito operator
# We should get the same results
op = Operator([u_v] + [u_t], opt='advanced')
op(dt=dt, time_M=nt)

print(norm(v[0]))
print(norm(tau[0]))

# Assert that the norms are the same
assert np.isclose(norm(v[0]), np.linalg.norm(v_xdsl[0]), atol=1e-6, rtol=0)
assert np.isclose(norm(v[1]), np.linalg.norm(v_xdsl[1]), atol=1e-6, rtol=0)
assert np.isclose(norm(tau[0]), np.linalg.norm(tau_xdsl[0]), atol=1e-6, rtol=0)
assert np.isclose(norm(tau[1]), np.linalg.norm(tau_xdsl[1]), atol=1e-6, rtol=0)
assert np.isclose(norm(tau[2]), np.linalg.norm(tau_xdsl[2]), atol=1e-6, rtol=0)

# Assert that the results are the same
assert np.allclose(v_xdsl[0].data, v[0].data, rtol=1e-8)
assert np.allclose(v_xdsl[1].data, v[1].data, rtol=1e-8)
assert np.allclose(tau[0].data, tau_xdsl[0].data, rtol=1e-8)
assert np.allclose(tau[1].data, tau_xdsl[1].data, rtol=1e-8)
assert np.allclose(tau[2].data, tau_xdsl[2].data, rtol=1e-8)


