import numpy as np
from devito import Grid, TimeFunction, Eq, Operator, norm
import pytest
# flake8: noqa

def test_udx():

    # Define a simple Devito Operator
    grid = Grid(shape=(5, 5))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0.1
    eq = Eq(u.forward, u.dx)
    op = Operator([eq])
    op.apply(time_M=5)
    norm1 = norm(u)

    u.data[:] = 0.1

    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)
    assert len(xdsl_op._module.ops) == 3

    norm2 = norm(u)
    
    assert np.isclose(norm1, norm2,   atol=1e-5, rtol=0)
    assert np.isclose(norm1, 14636.3955, atol=1e-5, rtol=0)

def test_u_plus1_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0
    eq = Eq(u.forward, u + 1)
    op = Operator([eq])
    op.apply(time_M=5)
    norm1 = norm(u)
    
    u.data[:] = 0
    xdsl_op = Operator([eq], opt='xdsl')
    xdsl_op.apply(time_M=5)
    assert len(xdsl_op._module.ops) == 3
    norm2 = norm(u)

    assert np.isclose(norm1, norm2, atol=1e-5, rtol=0)
    assert np.isclose(norm1, 23.43075, atol=1e-5, rtol=0)


@pytest.mark.xfail(reason="Cannot load and store the same field")
def test_u_and_v_conversion():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    v = TimeFunction(name='v', grid=grid)
    u.data[:] = 0.0001
    v.data[:] = 0.0001
    eq0 = Eq(u.forward, u.dt)
    # eq1 = Eq(v.forward, u.dt)
    op = Operator([eq0])
    op.apply(time_M=5, dt=0.1)
    norm_u = norm(u)
    norm_v = norm(v)

    u.data[:] = 0.0001
    v.data[:] = 0.0001
    xdsl_op = Operator([eq0], opt='xdsl')
    
    xdsl_op.apply(time_M=5, dt=0.1)
    norm_u2 = norm(u)
    norm_v2 = norm(v)

    assert np.isclose(norm_u, norm_u2, atol=1e-5, rtol=0)
    assert np.isclose(norm_u, 26.565891, atol=1e-5, rtol=0)
    assert np.isclose(norm_v, norm_v2, atol=1e-5, rtol=0)
    assert np.isclose(norm_v, 292.49646, atol=1e-5, rtol=0)


@pytest.mark.xfail(reason="Cannot load and store the same field")
def test_u_simple():
    # Define a simple Devito Operator
    grid = Grid(shape=(3, 3))
    u = TimeFunction(name='u', grid=grid)
    u.data[:] = 0.0001
    eq0 = Eq(u, u + 1)
    op = Operator([eq0])
    op.apply(time_M=5, dt=0.1)
    import pdb;pdb.set_trace()
    norm_u = norm(u)

    u.data[:] = 0.0001
    xdsl_op = Operator([eq0], opt='xdsl')
    xdsl_op.apply(time_M=5, dt=0.1)
    norm_u2 = norm(u)

    assert np.isclose(norm_u, norm_u2, atol=1e-5, rtol=0)
    assert np.isclose(norm_u, 26.565891, atol=1e-5, rtol=0)


# @pytest.mark.xfail(reason="Cannot load and store the same field")
def test_v_to_u():
    # Define a simple Devito Operator
    grid = Grid(shape=(30, 30))
    u = TimeFunction(name='u', grid=grid)
    v = TimeFunction(name='v', grid=grid)
    u.data[:] = 0.0001
    v.data[:] = 0.0001
    eq0 = Eq(u.forward, v + 1)
    op = Operator([eq0])
  
    u.data[:] = 0.0001
    v.data[:] = 0.0001
    op = Operator([eq0], opt='xdsl')
    op.apply(time_M=5, dt=0.1)

    import pdb;pdb.set_trace()
    print(norm(u))
    print(norm(v))
