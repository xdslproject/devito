import numpy as np
from argparse import ArgumentParser

from devito.logger import info
from devito import Constant, Function, smooth, configuration, norm
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic import demo_model, setup_geometry


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-d", "--shape", default=(51, 51, 51), type=int, nargs="+",
                        help="Determine the grid size")
    parser.add_argument('-f', '--full', default=False, action='store_true',
                        help="Execute all operators and store forward wavefield")
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbl", default=40,
                        type=int, help="Number of boundary layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("-opt", default="advanced",
                        choices=configuration._accepted['opt'],
                        help="Performance optimization level")
    parser.add_argument('-a', '--autotune', default='off',
                        choices=(configuration._accepted['autotuning']),
                        help="Operator auto-tuning mode")
    args = parser.parse_args()

    # 3D preset parameters
    shape = args.shape
    ndim = len(shape)
    spacing = tuple(ndim * [15.0])
    tn = 750. if ndim < 3 else 250.

    spacing=(20.0, 20.0, 20.0)
    tn=100.0
    space_order=4
    full_run=False
    autotune=False
    preset='layers-isotropic'
    checkpointing=False

    model = demo_model(preset, space_order=space_order, shape=shape, nbl=args.nbl,
                       dtype=np.float32, spacing=spacing)

    # Source and receiver geometries
    geometry = setup_geometry(model, tn)

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, geometry, kernel=args.kernel,
                                space_order=space_order)

    info("Applying Forward")
    # Define receiver geometry (spread across x, just below surface)
    u, summary = solver.forward(save=full_run, autotune=autotune)

    print("------------------------")
    print("norm(u)=", norm(u))
    print("-------------------------------")
