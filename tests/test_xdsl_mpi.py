"""
DEVITO_MPI=0 DEVITO_LOGGING=DEBUG pytest -m "not parallel" tests/test_xdsl_*
DEVITO_MPI=1 DEVITO_LOGGING=DEBUG pytest -m parallel tests/test_xdsl_mpi.py
"""

from conftest import skipif
import pytest
import numpy as np

from devito import (Grid, TimeFunction, Eq, norm, solve, Operator,
                    TensorTimeFunction, VectorTimeFunction, div, grad, diag)
from devito.types import SpaceDimension, Constant
from devito.tools import as_tuple
from examples.seismic.source import RickerSource, TimeAxis
from xdsl.dialects.stencil import StoreOp


pytestmark = skipif(['nompi'], whole_module=True)


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32, 32))
        x, y = grid.dimensions
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x-1, y] + f[t, x+1, y] + 1), opt='xdsl')
        op.apply(time=2)

        assert np.isclose(norm(f), 515.9845, rtol=1e-4)

    @pytest.mark.parallel(mode=[2])
    @pytest.mark.parametrize('shape', [(101, 101, 101), (202, 10, 45)])
    @pytest.mark.parametrize('so', [2, 4, 8])
    @pytest.mark.parametrize('to', [2])
    @pytest.mark.parametrize('nt', [10, 20, 100])
    def test_acoustic_3D(self, shape, so, to, nt):

        grid = Grid(shape=shape)
        dt = 0.0001

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=grid, time_order=to, space_order=so)

        pde = u.dt2 - u.laplace
        eq0 = solve(pde, u.forward)

        stencil = Eq(u.forward, eq0)
        u.data[:, :, :] = 0
        u.data[:, 40:50, 40:50] = 1

        # Devito Operator
        op = Operator([stencil])
        op.apply(time=nt, dt=dt)
        devito_norm = norm(u)

        u.data[:, :, :] = 0
        u.data[:, 40:50, 40:50] = 1

        # XDSL Operator
        xdslop = Operator([stencil], opt='xdsl')
        xdslop.apply(time=nt, dt=dt)
        xdsl_norm = norm(u)

        assert np.isclose(devito_norm, xdsl_norm, rtol=1e-04).all()


class TestTopology(object):

    @pytest.mark.parallel(mode=[2])
    @pytest.mark.parametrize('shape', [(10, 10, 10)])
    @pytest.mark.parametrize('to', [2])
    @pytest.mark.parametrize('nt', [10])
    def test_topology(self, shape, to, nt):

        grid = Grid(shape=shape)
        dt = 0.0001

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=grid, time_order=to)

        stencil = Eq(u.forward, u + 1)
        u.data[:, :, :] = 0

        # XDSL Operator
        xdslop = Operator([stencil], opt='xdsl')
        topology, _ = xdslop.mpi_shape

        assert topology == (2, 1, 1)

        xdslop.apply(time=nt, dt=dt)

    @pytest.mark.parallel(mode=[4])
    @pytest.mark.parametrize('shape', [(10, 10, 10)])
    @pytest.mark.parametrize('to', [2])
    @pytest.mark.parametrize('nt', [10])
    def test_topology_4(self, shape, to, nt):

        grid = Grid(shape=shape)
        dt = 0.0001

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=grid, time_order=to)

        stencil = Eq(u.forward, u + 1)
        u.data[:, :, :] = 0

        # XDSL Operator
        xdslop = Operator([stencil], opt='xdsl')
        topology, _ = xdslop.mpi_shape

        assert topology == (2, 2, 1)

        xdslop.apply(time=nt, dt=dt)

    @pytest.mark.parallel(mode=[8])
    @pytest.mark.parametrize('shape', [(10, 10, 10)])
    @pytest.mark.parametrize('to', [2])
    @pytest.mark.parametrize('nt', [10])
    def test_topology_8(self, shape, to, nt):

        grid = Grid(shape=shape)
        dt = 0.0001

        # Define the wavefield with the size of the model and the time dimension
        u = TimeFunction(name="u", grid=grid, time_order=to)

        stencil = Eq(u.forward, u + 1)
        u.data[:, :, :] = 0

        # XDSL Operator
        xdslop = Operator([stencil], opt='xdsl')
        topology, _ = xdslop.mpi_shape

        assert topology == (2, 2, 2)

        xdslop.apply(time=nt, dt=dt)


class TestElastic():

    @pytest.mark.xfail(reason=" With MPI, data can only be set via scalars, numpy arrays or other data")  # noqa
    @pytest.mark.parallel(mode=[1])
    @pytest.mark.parametrize('shape', [(101, 101), (201, 201), (301, 301)])
    @pytest.mark.parametrize('so', [2, 4, 8])
    @pytest.mark.parametrize('nt', [10, 20, 50, 100])
    def test_elastic_2D(self, shape, so, nt):

        # Initial grid: km x km, with spacing
        shape = shape  # Number of grid point (nx, nz)
        spacing = as_tuple(10.0 for _ in range(len(shape)))
        extent = tuple([s*(n-1) for s, n in zip(spacing, shape)])

        x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=extent[0]/(shape[0]-1)))  # noqa
        z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=extent[1]/(shape[1]-1)))  # noqa
        grid = Grid(extent=extent, shape=shape, dimensions=(x, z))

        # To be checked again in the future
        # class DGaussSource(WaveletSource):

        #    def wavelet(self, f0, t):
        #        a = 0.004
        #        return -2.*a*(t - 1/f0) * np.exp(-a * (t - 1/f0)**2)

        # Timestep size from Eq. 7 with V_p=6000. and dx=100
        t0, tn = 0., nt
        dt = (10. / np.sqrt(2.)) / 6.
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        src = RickerSource(name='src', grid=grid, f0=0.01, time_range=time_range)
        src.coordinates.data[:] = [250., 250.]

        # Now we create the velocity and pressure fields
        v = VectorTimeFunction(name='v', grid=grid, space_order=so, time_order=1)
        tau = TensorTimeFunction(name='t', grid=grid, space_order=so, time_order=1)

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

        # Inject sources. We use it to preinject data
        # Up to here, let's only use Devito
        op = Operator([u_v] + [u_t] + src_xx + src_zz)
        op(dt=dt)

        opx = Operator([u_v] + [u_t], opt='xdsl')
        opx(dt=dt, time_M=nt)

        store_ops = [op for op in opx._module.walk() if isinstance(op, StoreOp)]
        assert len(store_ops) == 5

        xdsl_norm_v0 = norm(v[0])
        xdsl_norm_v1 = norm(v[1])
        xdsl_norm_tau0 = norm(tau[0])
        xdsl_norm_tau1 = norm(tau[1])
        xdsl_norm_tau2 = norm(tau[2])
        xdsl_norm_tau3 = norm(tau[3])

        # Reinitialize the fields to zero

        v[0].data[:] = 0
        v[1].data[:] = 0

        tau[0].data[:] = 0
        tau[1].data[:] = 0
        tau[2].data[:] = 0
        tau[3].data[:] = 0

        # Inject sources. We use it to preinject data
        op = Operator([u_v] + [u_t] + src_xx + src_zz)
        op(dt=dt)

        op = Operator([u_v] + [u_t], opt='advanced')
        op(dt=dt, time_M=nt)

        dv_norm_v0 = norm(v[0])
        dv_norm_v1 = norm(v[1])
        dv_norm_tau0 = norm(tau[0])
        dv_norm_tau1 = norm(tau[1])
        dv_norm_tau2 = norm(tau[2])
        dv_norm_tau3 = norm(tau[3])

        assert np.isclose(xdsl_norm_v0, dv_norm_v0, rtol=1e-04)
        assert np.isclose(xdsl_norm_v1, dv_norm_v1, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau0, dv_norm_tau0, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau1, dv_norm_tau1, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau2, dv_norm_tau2, rtol=1e-04)
        assert np.isclose(xdsl_norm_tau3, dv_norm_tau3, rtol=1e-04)

        assert not np.isclose(xdsl_norm_v0, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_v1, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau0, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau1, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau2, 0.0, rtol=1e-04)
        assert not np.isclose(xdsl_norm_tau3, 0.0, rtol=1e-04)
