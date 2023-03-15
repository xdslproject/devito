from devito.ir import ietxdsl
from devito import (Grid, TimeFunction, Eq, Operator,
                    XDSLOperator, norm, configuration)
import numpy as np
from xdsl.printer import Printer

configuration['log-level'] = 'DEBUG'

if __name__ == '__main__':

    grid = Grid(shape=(3000, 3000))
    steps = 5_000

    u = TimeFunction(name='u', grid=grid)

    
    u.data[:, :] = 4
    eq = Eq(u.forward, u + 0.01)
    xop = XDSLOperator([eq])
    xop.apply(time_M=steps)
    xdsl_data: np.array = u.data_with_halo.copy()
    xdsl_norm = norm(u)

    u.data[:, :] = 4
    op = Operator([eq])
    op.apply(time_M=steps)
    orig_data: np.array = u.data_with_halo.copy()
    orig_norm = norm(u)

    print("orig={}, xdsl={}".format(xdsl_norm, orig_norm))
    assert np.isclose(xdsl_data, orig_data, rtol=1e-06).all()


    # module = ietxdsl.transform_devito_to_iet_ssa(op)
    # p = Printer(target=Printer.Target.MLIR)
    # p.print(module)

    # print("\n\nAFTER REWRITE:\n")

    # ietxdsl.iet_to_standard_mlir(module)

    # p = Printer(target=Printer.Target.MLIR)
    # p.print(module)
