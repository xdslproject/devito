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


def test_xdsl_I():
    # Define a simple Devito Operator

    nt = 10
    grid = Grid(shape=(5,))
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

    assert np.isclose(dnorm, xnorm, atol=0, rtol=1.e-7)
