dims = {"2d5pt": 2, "3d_diff": 3}

import argparse
from math import prod
from devito.operator.operator import Operator
from devito.operator.xdsl_operator import XDSLOperator
import fast_benchmarks


parser = argparse.ArgumentParser(description="Process arguments.")

parser.add_argument('benchmark_name', choices=["2d5pt", "3d_diff"])
parser.add_argument('-i', '--init_size', type=int, default=128)
parser.add_argument('-m', '--max_total_size', type=int, default=2048**3)

args = parser.parse_args()

bench_name = args.benchmark_name
init_size = args.init_size
max_size = args.max_total_size
size = tuple([init_size] * dims[bench_name])
csv_name = f"{bench_name}_grid_runtimes.csv"


def get_runtimes_for_size(
    size: tuple[int, ...]
) -> tuple[tuple[int, ...], list[float], list[float]]:
    grid, u, eq0, dt = fast_benchmarks.get_equation(bench_name, size, 2, 1, 10)
    xop = XDSLOperator([eq0])
    nt = 100
    fast_benchmarks.compile_main(bench_name, grid, u, xop, dt, nt)
    fast_benchmarks.compile_kernel(bench_name, xop.mlircode, fast_benchmarks.XDSL_CPU_PIPELINE, fast_benchmarks.CPU_PIPELINE)
    fast_benchmarks.link_kernel(bench_name)
    xdsl_runs = [fast_benchmarks.run_kernel(bench_name) for _ in range(10)]

    op = Operator([eq0])
    devito_runs = [fast_benchmarks.run_operator(op, nt, dt) for _ in range(10)]

    return (size, xdsl_runs, devito_runs)

next_mul = len(size) - 1

fast_benchmarks.compile_interop(bench_name, True)

with open(csv_name, "w") as f:
    
    f.write("Grid Size,Devito/xDSL,Devito\n")
    f.flush()

    while prod(size) <= max_size:
        runtime = get_runtimes_for_size(size)
        f.write(f"{runtime}\n")
        f.flush()
        size = list(size)
        size[next_mul] *= 2
        size = tuple(size)
        next_mul = (next_mul - 1) % len(size)
