"""
DEVITO_MPI=0 DEVITO_LOGGING=DEBUG pytest -m "not parallel" tests/test_xdsl_*
DEVITO_MPI=1 DEVITO_LOGGING=DEBUG pytest -m parallel tests/test_xdsl_mpi.py
"""

import numpy as np
import pytest

from conftest import skipif
from devito import Grid, TimeFunction, Eq, Operator, XDSLOperator
from devito.data import LEFT

pytestmark = skipif(['nompi'], whole_module=True)


class TestOperatorSimple(object):

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        op = Operator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
        op.apply(time=1)

        f.data_with_halo[:] = 1.

        xdsl_op = XDSLOperator(Eq(f.forward, f[t, x-1] + f[t, x+1] + 1))
        xdsl_op.__class__ = XDSLOperator

        xdsl_op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 3.)
        if f.grid.distributor.myrank == 0:
            assert f.data_ro_domain[0, 0] == 5.
            assert np.all(f.data_ro_domain[0, 1:] == 7.)
        elif f.grid.distributor.myrank == f.grid.distributor.nprocs - 1:
            assert f.data_ro_domain[0, -1] == 5.
            assert np.all(f.data_ro_domain[0, :-1] == 7.)
        else:
            assert np.all(f.data_ro_domain[0] == 7.)

    @pytest.mark.parallel(mode=[1])
    def test_trivial_eq_1d_asymmetric(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        t = grid.stepping_dim

        f = TimeFunction(name='f', grid=grid)
        f.data_with_halo[:] = 1.

        xdsl_op = XDSLOperator(Eq(f.forward, f[t, x+1] + 1))
        xdsl_op.__class__ = XDSLOperator
        xdsl_op.apply(time=1)

        assert np.all(f.data_ro_domain[1] == 2.)
        if f.grid.distributor.myrank == 0:
            assert np.all(f.data_ro_domain[0] == 3.)
        else:
            assert np.all(f.data_ro_domain[0, :-1] == 3.)
            assert f.data_ro_domain[0, -1] == 2.

    @pytest.mark.parallel(mode=1)
    def test_trivial_eq_1d_save(self):
        grid = Grid(shape=(32,))
        x = grid.dimensions[0]
        time = grid.time_dim

        f = TimeFunction(name='f', grid=grid, save=5)
        f.data_with_halo[:] = 1.

        xdsl_op = XDSLOperator(Eq(f.forward, f[time, x-1] + f[time, x+1] + 1))
        xdsl_op.__class__ = XDSLOperator
        xdsl_op.apply()

        time_M = xdsl_op._prepare_arguments()['time_M']

        assert np.all(f.data_ro_domain[1] == 3.)
        glb_pos_map = f.grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(f.data_ro_domain[-1, time_M:] == 31.)
        else:
            assert np.all(f.data_ro_domain[-1, :-time_M] == 31.)
