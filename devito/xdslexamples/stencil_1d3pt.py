from devito.ir import ietxdsl
from devito import (Grid, TimeFunction, Eq, Operator,
                    XDSLOperator, norm, configuration,
                    Constant)
from devito import solve

import numpy as np
from xdsl.printer import Printer

configuration['log-level'] = 'DEBUG'

if __name__ == '__main__':

    nx = 300
    grid = Grid(shape=(nx, ))
    steps = 5
    nu = .5
    init_val = 10
    dx = 1. / (nx - 1)
    sigma = 0.1
    dt = sigma * dx / nu

    u = TimeFunction(name='u', grid=grid, space_order=32)

    u.data[:, :] = 1
    u.data[:, int(nx/2)] = init_val
    print(f"Original norm is {norm(u)}")

    # Create an operator with second-order derivatives
    eq0 = Eq(u.dt, 0.5*u.dx2)
    stencil = solve(eq0, u.forward)
    eq = Eq(u.forward, stencil)
    xop = XDSLOperator([eq])
    xop.apply(time_M=steps, dt=dt)


    xdsl_data: np.array = u.data.copy()
    xdsl_norm = norm(u)

    u.data[:, :] = 1
    u.data[:, int(nx/2)] = init_val

    print(f"Original norm is {norm(u)}")

    op = Operator([eq])
    op.apply(time_M=steps, dt=dt)
    orig_norm = norm(u)
    print("orig={}, xdsl={}".format(xdsl_norm, orig_norm))

    orig_data: np.array = u.data.copy()
    

    
    # assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()

    import pdb;pdb.set_trace()
    # module = ietxdsl.transform_devito_to_iet_ssa(op)
    # p = Printer(target=Printer.Target.MLIR)
    # p.print(module)

    # print("\n\nAFTER REWRITE:\n")

    # ietxdsl.iet_to_standard_mlir(module)

    # p = Printer(target=Printer.Target.MLIR)
    # p.print(module)


    import matplotlib.pyplot as plt
    plt.plot(xdsl_data[1]); plt.pause(1)