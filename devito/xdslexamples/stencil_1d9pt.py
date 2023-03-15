from devito.ir import ietxdsl
from devito import (Grid, TimeFunction, Eq, Operator,
                    XDSLOperator, norm, configuration)
import numpy as np
from xdsl.printer import Printer

configuration['log-level'] = 'DEBUG'

if __name__ == '__main__':

    grid = Grid(shape=(300, ))
    steps = 5
    u = TimeFunction(name='u', grid=grid, space_order=8)

    u.data[:, :] = 10
    print(f"Original norm is {norm(u)}")

    eq = Eq(u.forward, u.dx)

    xop = XDSLOperator([eq])
    xop.apply(time_M=steps)
    xdsl_data: np.array = u.data_with_halo.copy()
    xdsl_norm = norm(u)

    u.data[:, :] = 10
    op = Operator([eq])
    op.apply(time_M=steps)
    orig_data: np.array = u.data_with_halo.copy()
    orig_norm = norm(u)

    print("orig={}, xdsl={}".format(xdsl_norm, orig_norm))
    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()

    import pdb;pdb.set_trace()
    # module = ietxdsl.transform_devito_to_iet_ssa(op)
    # p = Printer(target=Printer.Target.MLIR)
    # p.print(module)

    # print("\n\nAFTER REWRITE:\n")

    # ietxdsl.iet_to_standard_mlir(module)

    # p = Printer(target=Printer.Target.MLIR)
    # p.print(module)


    import matplotlib.pyplot as plt
    plt.plot(xdsl_data[0]); plt.pause(1)