import numpy as np
import pytest

from devito import (Grid, TensorTimeFunction, VectorTimeFunction, div, grad,
                    diag, solve, Operator, Eq, Constant, norm, SpaceDimension,
                    switchconfig, sin, cos, tan)
from devito.types import Array, Function, TimeFunction
from devito.tools import as_tuple

from examples.seismic.source import RickerSource, TimeAxis
from xdsl.dialects.scf import For, Yield
from xdsl.dialects.arith import Addi, Addf, Mulf
from xdsl.dialects.arith import Constant as xdslconstant
from xdsl.dialects.func import Call, Return
from xdsl.dialects.stencil import FieldType, ApplyOp, LoadOp, StoreOp
from xdsl.dialects.llvm import LLVMPointerType
from xdsl.dialects.memref import Load
from xdsl.dialects.experimental import math


@pytest.mark.parametrize('shape, rtol', [(5, 1e-7), (100, 1e-6)])
def test_precision_I(shape, rtol):
    # Define an upper bound of precision

    nt = 10
    grid = Grid(shape=(shape,))
    u = TimeFunction(name='u', grid=grid, space_order=2)

    init = 1.
    u.data[:, :] = init

    eq = Eq(u.forward, u.dx*0.1)
    opd = Operator([eq])
    opd.apply(time_M=nt)
    dnorm = norm(u)

    u.data[:, :] = init
    opx = Operator([eq], opt='xdsl')
    opx.apply(time_M=nt)
    xnorm = norm(u)

    assert np.isclose(dnorm, xnorm, atol=0, rtol=rtol)


@pytest.mark.parametrize('shape, rtol', [(5, 1e-5), (100, 1e-6)])
def test_precision_II(shape, rtol):
    # Define an upper bound of precision

    nt = 1
    grid = Grid(shape=(shape,))
    u = TimeFunction(name='u', grid=grid, time_order=2, space_order=2)

    init0 = 1.
    init1 = 2.
    init2 = 3.

    dt = 0.1
    u.data[0, :] = init0
    u.data[1, :] = init1
    u.data[2, :] = init2

    import pdb; pdb.set_trace()

    pde = u.dt2 - u.laplace
    stencil = Eq(u.forward, solve(pde, u.forward))

    opd = Operator([stencil])
    opd.apply(time_M=nt, dt=dt)
    dnorm = norm(u)
    dnorm0 = np.linalg.norm(u.data[0])

    import pdb; pdb.set_trace()

    u.data[0, :] = init0
    u.data[1, :] = init1
    u.data[2, :] = init2

    opx = Operator([stencil], opt='xdsl')
    opx.apply(time_M=nt, dt=dt)
    xnorm = norm(u)
    xnorm0 = np.linalg.norm(u.data[0])

    import pdb; pdb.set_trace()

    assert np.isclose(dnorm0, xnorm0, atol=0, rtol=rtol)
    assert np.isclose(dnorm, xnorm, atol=0, rtol=rtol)
