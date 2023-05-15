import argparse
from functools import reduce
from math import prod
import pathlib
from subprocess import PIPE, Popen
from typing import Any

import numpy as np

from devito import Constant, Eq, Grid, Operator, TimeFunction, XDSLOperator, solve
from devito.ir.ietxdsl.cluster_to_ssa import generate_launcher_base
from devito.logger import info
from devito.operator.profiling import PerfEntry, PerfKey, PerformanceSummary
import sys


CFLAGS = '-O3 -march=native -mtune=native' 

CPU_PIPELINE = "builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"
OPENMP_PIPELINE = "builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"
GPU_PIPELINE = "builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=1024,1,1}, canonicalize, func.func(gpu-map-parallel-loops), convert-parallel-loops-to-gpu, lower-affine, gpu-kernel-outlining,func.func(gpu-async-region),canonicalize,convert-arith-to-llvm{index-bitwidth=64},${MEMREF_TO_LLVM_PASS}{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize)"

XDSL_CPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir"
XDSL_GPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-gpu"

MAIN_MLIR_FILE_PIPELINE = '"builtin.module(canonicalize, convert-scf-to-cf, convert-cf-to-llvm{index-bitwidth=64}, convert-math-to-llvm, convert-arith-to-llvm{index-bitwidth=64},finalize-memref-to-llvm{index-bitwidth=64}, convert-func-to-llvm, reconcile-unrealized-casts, canonicalize)"'

def get_equation(name:str, shape:tuple[int, ...], so: int, to: int, init_value: int):
    match name:
        case '2d5pt':
            nt = args.nt
            nx, ny = shape
            nu = 0.5
            dx = 2.0 / (nx - 1)
            dy = 2.0 / (ny - 1)
            sigma = 0.25
            dt = sigma * dx * dy / nu
            # Field initialization
            grid = Grid(shape=(nx, ny))
            u = TimeFunction(name="u", grid=grid, space_order=so, time_order=to)
            u.data[:, :, :] = 0
            u.data[:, int(nx / 2), int(nx / 2)] = init_value
            u.data[:, int(nx / 2), -int(nx / 2)] = -init_value

            # Create an equation with second-order derivatives
            a = Constant(name="a")
            eq = Eq(u.dt, a * u.laplace + 0.01)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case '3d_diff':
            nx, ny, nz = shape
            nu = .5
            dx = 2. / (nx - 1)
            dy = 2. / (ny - 1)
            sigma = .25
            grid = Grid(shape=(nx, ny, nz), extent=(2., 2., 2.))
            u = TimeFunction(name='u', grid=grid, space_order=so)
            # init_hat(field=u.data[0], dx=dx, dy=dy, value=2.)
            u.data[:, :, :, :] = 0
            u.data[:, int(nx/2), :, :] = 1

            a = Constant(name='a')
            # Create an equation with second-order derivatives
            eq = Eq(u.dt, a * u.laplace, subdomain=grid.interior)
            stencil = solve(eq, u.forward)
            eq0 = Eq(u.forward, stencil)
        case _:
            raise Exception("Unknown benchamark!")

    return (grid, u, eq0)


def dump_input(input: TimeFunction, filename: str):
    input.data_with_halo[0,:,:].tofile(filename)


def dump_main(bench_name: str, grid: Grid, u: TimeFunction, xop: XDSLOperator, dt: float, nt: int):
    info("Operator in " + bench_name + ".main.mlir")
    with open(bench_name + ".main.mlir", "w") as f:
        f.write(
            generate_launcher_base(
                xop._module,
                {
                    "time_m": 0,
                    "time_M": nt,
                    **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
                    "a": 0.1,
                    "dt": dt,
                },
                u.shape_allocated[1:],
            )
        )

def compile_main(bench_name: str, grid: Grid, u: TimeFunction, xop: XDSLOperator, dt: float, nt: int):
    main = generate_launcher_base(
        xop._module,
        {
            "time_m": 0,
            "time_M": nt,
            **{str(k): float(v) for k, v in dict(grid.spacing_map).items()},
            "a": 0.1,
            "dt": dt,
        },
        u.shape_allocated[1:],
    )
    cmd = f'mlir-opt --pass-pipeline={MAIN_MLIR_FILE_PIPELINE} | mlir-translate --mlir-to-llvmir | clang -x ir -c -o {bench_name}.main.o - {CFLAGS} 2>&1'
    out:str
    try:
        print(f"Trying to compile {bench_name}.main.o with:")
        print(cmd)
        mlir_opt = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
        t = mlir_opt.communicate(main.encode())
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e

def compile_interop(bench_name: str):
    cmd = f'clang -O3 -c interop.c -o {bench_name}.interop.o {CFLAGS} -DOUTFILE_NAME="\\"{pathlib.Path(__file__).parent.resolve()}/{bench_name}.stencil.data\\"" -DINFILE_NAME="\\"{pathlib.Path(__file__).parent.resolve()}/{bench_name}.input.data\\""'
    out:str
    try:
        print(f"Trying to compile {bench_name}.main.o with:")
        print(cmd)
        mlir_opt = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)
        t = mlir_opt.communicate()
        mlir_opt.wait()
        out = t[0].decode() + "\n" + t[1].decode()
        print(out)
    except Exception as e:
        print("something went wrong... Used command:")
        print(cmd)
        print("Output:")
        print(out)
        raise e


def run_operator(op: Operator, nt: int, dt: float):
    res = op.apply(time_M=nt, a=0.1, dt=dt)
    assert isinstance(res, PerformanceSummary)
    o = res[PerfKey("section0", None)]
    assert isinstance(o, PerfEntry)
    return o.time


def main(bench_name: str, nt:int, dt:Any):

    grid, u, eq0 = get_equation(bench_name, args.shape, so, to, init_value)

    if args.xdsl:
        dump_input(u, bench_name + ".input.data")
        xop = XDSLOperator([eq0])
        if args.dump_main:
            dump_main(bench_name, grid, u, xop, dt, nt)
        compile_main(bench_name, grid, u, xop, dt, nt)
        compile_interop(bench_name)
        if args.dump_mlir:
            info("Dump mlir code in  in " + bench_name + ".mlir")
            with open(bench_name + ".mlir", "w") as f:
                f.write(xop.mlircode)
        
    else:
        op = Operator([eq0])
        rt = run_operator(op, nt, dt)
        print(f"Devito finer runtime: {rt} s")
        if args.no_output_dump:
            info("Skipping result data saving.")
        else:
            # get final data step
            # this is cursed math, but we assume that:
            #  1. Every kernel always writes to t1
            #  2. The formula for calculating t1 = (time + n - 1) % n, where n is the number of time steps we have
            #  3. the loop goes for (...; time <= time_M; ...), which means that the last value of time is time_M
            #  4. time_M is always nt in this example
            t1 = (nt + u._time_size - 1) % (2)

            res_data: np.array = u.data[t1,:,:]
            info("Save result data to " + bench_name + ".devito.data")
            res_data.tofile(bench_name + ".devito.data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process arguments.")

    parser.add_argument('benchmark_name', choices=["2d5pt", "3d_diff"])
    parser.add_argument(
        "-d",
        "--shape",
        default=(11, 11),
        type=int,
        nargs="+",
        help="Number of grid points along each axis",
    )
    parser.add_argument(
        "-so",
        "--space_order",
        default=2,
        type=int,
        help="Space order of the simulation",
    )
    parser.add_argument(
        "-to", "--time_order", default=1, type=int, help="Time order of the simulation"
    )
    parser.add_argument(
        "-nt", "--nt", default=10, type=int, help="Simulation time in millisecond"
    )
    parser.add_argument(
        "-bls", "--blevels", default=2, type=int, nargs="+", help="Block levels"
    )
    parser.add_argument("-xdsl", "--xdsl", default=False, action="store_true")
    parser.add_argument("-nod", "--no_output_dump", default=False, action="store_true")
    parser.add_argument('--dump_mlir', default=False, action='store_true')
    parser.add_argument('--dump_main', default=False, action='store_true')
    
    args = parser.parse_args()

    benchmark_dim: int
    match args.benchmark_name:
        case '2d5pt':
            benchmark_dim = 2
        case '3d_diff':
            benchmark_dim = 3
    
    if len(args.shape) != benchmark_dim:
        print(f"Expected {benchmark_dim}d shape for this benchmark, got {args.shape}")
        sys.exit(1)

    d = (2.0 / (n - 1) for n in args.shape)
    nu = .5
    sigma = .25
    dt = sigma * reduce(lambda a,b: a*b, d) / nu
    

    so = args.space_order
    to = args.time_order

    init_value = 10
    main(args.benchmark_name,
         args.nt,
         dt)
