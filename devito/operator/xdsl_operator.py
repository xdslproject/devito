import os
import subprocess
import ctypes
import tempfile

from math import ceil
from collections import OrderedDict
from io import StringIO
from operator import attrgetter

from cached_property import cached_property

from devito import Operator
from devito.arch import compiler_registry, platform_registry
from devito.data import default_allocator
from devito.exceptions import InvalidOperator
from devito.ir.clusters import ClusterGroup, clusterize
from devito.ir.equations import LoweredEq, lower_exprs
from devito.ir.iet import (Callable, CInterface, EntryFunction, FindSymbols, MetaCall,
                           derive_parameters, iet_build)
from devito.ir.ietxdsl import (finalize_module_with_globals)
from devito.ir.stree import stree_build
from devito.ir.support import AccessMode, SymbolRegistry
from devito.ir.ietxdsl.cluster_to_ssa import (ExtractDevitoStencilConversion,
                                              convert_devito_stencil_to_xdsl_stencil)
from devito.logger import debug, info, perf, warning, is_log_enabled_for
from devito.operator.operator import IRs
from devito.operator.profiling import AdvancedProfilerVerbose, create_profile
from devito.operator.registry import operator_selector
from devito.parameters import configuration
from devito.passes import (Graph, lower_index_derivatives, generate_implicit,
                           generate_macros, minimize_symbols, unevaluate)
from devito.passes.iet import CTarget
from devito.symbolics import estimate_cost
from devito.tools import (DAG, ReducerMap, as_tuple, flatten,
                          filter_sorted, frozendict, is_integer, split, timed_pass,
                          contains_val)
from devito.types import Evaluable, TimeFunction, Grid
from devito.types.mlir_types import ptr_of, f32
from devito.mpi import MPI

from xdsl.printer import Printer


from devito.core.cpu import (MLIR_CPU_PIPELINE, XDSL_CPU_PIPELINE, XDSL_MPI_PIPELINE,
                             MLIR_OPENMP_PIPELINE, XDSL_GPU_PIPELINE, MLIR_GPU_PIPELINE)

__all__ = ['XDSLOperator']


# small interop shim script for stuff that we don't want to implement in mlir-ir
_INTEROP_C = """
#include <time.h>

double timer_start() {
  // return a number representing the current point in time
  // it might be offset by a fixed ammount
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec) + (t.tv_nsec * 1e-9);
}

double timer_end(double start) {
  // return time elaspes since start in seconds
  return (timer_start() - start);
}
"""


class XDSLOperator(Operator):

    _Target = CTarget

    def __new__(cls, expressions, **kwargs):
        self = super(XDSLOperator, cls).__new__(cls, expressions, **kwargs)
        delete = not os.getenv("XDSL_SKIP_CLEAN", False)
        self._tf = tempfile.NamedTemporaryFile(prefix="devito-jit-", suffix='.so',
                                               delete=delete)
        self._interop_tf = tempfile.NamedTemporaryFile(prefix="devito-jit-interop-",
                                                       suffix=".o", delete=delete)
        self._make_interop_o()
        self.__class__ = cls
        return self

    def _make_interop_o(self):
        """
        compile the interop.o file
        """
        res = subprocess.run(
            f'clang -x c - -c -o {self._interop_tf.name}',
            shell=True,
            input=_INTEROP_C,
            text=True,
            stderr=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
        )
        assert res.returncode == 0

    @property
    def mpi_shape(self) -> tuple:
        # TODO: move it elsewhere
        dist = self.functions[0].grid.distributor

        # reverse topology for row->column major
        return dist.topology, dist.myrank

    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.
        It is ensured that JIT compilation will only be performed
        once per Operator, reagardless of how many times this method
        is invoked.
        """

        with self._profiler.timer_on('jit-compile'):
            is_mpi = MPI.Is_initialized()
            is_gpu = os.environ.get("DEVITO_PLATFORM", None) == 'nvidiaX'
            is_omp = os.environ.get("DEVITO_LANGUAGE", None) == 'openmp'

            if is_mpi and is_gpu:
                raise RuntimeError("Cannot run MPI+GPU for now!")

            if is_omp and is_gpu:
                raise RuntimeError("Cannot run OMP+GPU!")

            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants,
                                         gpu_boilerplate=is_gpu)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            to_tile = len(list(filter(lambda d: d.is_Space, self.dimensions)))-1

            xdsl_pipeline = XDSL_CPU_PIPELINE(to_tile)
            mlir_pipeline = MLIR_CPU_PIPELINE

            if is_omp:
                mlir_pipeline = MLIR_OPENMP_PIPELINE

            if is_mpi:
                shape, mpi_rank = self.mpi_shape
                # Run with restrict domain=false so we only introduce the swaps but don't
                # reduce the domain of the computation
                # (as devito has already done that for us)
                slices = ','.join(str(x) for x in shape)

                decomp = "2d-grid" if len(shape) == 2 else "3d-grid"

                decomp = f"{{strategy={decomp} slices={slices} restrict_domain=false}}"
                xdsl_pipeline = XDSL_MPI_PIPELINE(decomp, to_tile)
            elif is_gpu:
                xdsl_pipeline = XDSL_GPU_PIPELINE
                # Get GPU blocking shapes
                block_sizes: list[int] = [min(target, self._jit_kernel_constants.get(f"{dim}_size", 1)) for target, dim in zip([32, 4, 8], ["x", "y", "z"])]  # noqa
                block_sizes = ','.join(str(bs) for bs in block_sizes)
                mlir_pipeline = MLIR_GPU_PIPELINE(block_sizes)

            # allow jit backdooring to provide your own xdsl code
            backdoor = os.getenv('XDSL_JIT_BACKDOOR')
            if backdoor is not None:
                if os.path.splitext(backdoor)[1] == ".so":
                    info(f"JIT Backdoor: skipping compilation and using {backdoor}")
                    self._tf.name = backdoor
                    return
                print("JIT Backdoor: loading xdsl file from: " + backdoor)
                with open(backdoor, 'r') as f:
                    module_str = f.read()
            source_name = os.path.splitext(self._tf.name)[0] + ".mlir"
            source_file = open(source_name, "w")
            source_file.write(module_str)
            source_file.close()

            # Compile IR using xdsl-opt | mlir-opt | mlir-translate | clang
            cflags = "-O3 -march=native -mtune=native -lmlir_c_runner_utils"

            try:
                cc = "clang"

                if is_mpi:
                    cflags += ' -lmpi '
                    cc = "mpicc -cc=clang"
                if is_omp:
                    cflags += " -fopenmp "
                if is_gpu:
                    cflags += " -lmlir_cuda_runtime "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                xdsl_cmd = f'xdsl-opt {source_name} -p {xdsl_pipeline}'
                mlir_cmd = f'mlir-opt -p {mlir_pipeline}'
                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                clang_cmd = f'{cc} {cflags} -shared -o {self._tf.name} {self._interop_tf.name} -xir -'  # noqa

                comp_steps = [xdsl_cmd,
                              mlir_cmd,
                              mlir_translate_cmd,
                              clang_cmd]

                # Execute each command and store the outputs
                outputs = []
                stdout = None
                for cmd in comp_steps:
                    return_code, stdout, stderr = self._cmd_compile(cmd, stdout)
                    # Use DEVITO_LOGGING=DEBUG to print
                    debug(cmd)
                    outputs.append({
                        'command': cmd,
                        'return_code': return_code,
                        'stdout': stdout,
                        'stderr': stderr
                    })

            except Exception as ex:
                print("error")
                raise ex

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))

    def _cmd_compile(self, cmd, input=None):

        # Could be dropped unless PIPE is never empty in the future
        stdin = subprocess.PIPE if input is not None else None  # noqa

        res = subprocess.run(
            cmd,
            input=input,
            shell=True,
            text=True,
            capture_output=True,
            executable="/bin/bash"
        )

        if res.returncode != 0:
            print("compilation failed with output:")
            print(res.stderr)

        assert res.returncode == 0
        return res.returncode, res.stdout, res.stderr

    @property
    def _soname(self):
        return self._tf.name

    def setup_memref_args(self):
        """
        Add memrefs to args dictionary so they can be passed to the cfunction
        """
        args = dict()
        for arg in self.functions:
            # For every TimeFunction add memref
            if isinstance(arg, TimeFunction):
                data = arg._data
                for t in range(data.shape[0]):
                    args[f'{arg._C_name}_{t}'] = data[t, ...].ctypes.data_as(ptr_of(f32))

        self._jit_kernel_constants.update(args)

    @classmethod
    def _build(cls, expressions, **kwargs) -> Callable:
        debug("-Building operator")
        # Python- (i.e., compile-) and C-level (i.e., run-time) performance
        profiler = create_profile('timers')

        # Lower the input expressions into an IET
        debug("-Lower expressions")
        irs, _, module = cls._lower(expressions, profiler=profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **irs.iet.args)
        Callable.__init__(op, **op.args)

        # Header files, etc.
        # op._headers = OrderedSet(*cls._default_headers)
        # op._headers.update(byproduct.headers)
        # op._globals = OrderedSet(*cls._default_globals)
        # op._includes = OrderedSet(*cls._default_includes)
        # op._includes.update(profiler._default_includes)
        # op._includes.update(byproduct.includes)
        op._module = module

        # Required for the jit-compilation
        op._compiler = kwargs['compiler']
        op._language = kwargs['language']
        op._lib = None
        op._cfunction = None

        # Potentially required for lazily allocated Functions
        op._mode = kwargs['mode']
        op._options = kwargs['options']
        op._allocator = kwargs['allocator']
        op._platform = kwargs['platform']

        # References to local or external routines
        op._func_table = OrderedDict()
        op._func_table.update(OrderedDict([(i, MetaCall(None, False))
                                           for i in profiler._ext_calls]))
        # op._func_table.update(OrderedDict([(i.root.name, i) for i in byproduct.funcs]))

        # Internal mutable state to store information about previous runs, autotuning
        # reports, etc
        op._state = cls._initialize_state(**kwargs)

        # Produced by the various compilation passes
        op._reads = filter_sorted(flatten(e.reads for e in irs.expressions))
        op._writes = filter_sorted(flatten(e.writes for e in irs.expressions))
        op._dimensions = set().union(*[e.dimensions for e in irs.expressions])
        op._dtype, op._dspace = irs.clusters.meta
        op._profiler = profiler

        return op

    # Compilation -- Expression level

    @classmethod
    def _lower(cls, expressions, **kwargs):
        """
        Perform the lowering Expressions -> Clusters -> ScheduleTree -> IET.
        """
        # Create a symbol registry
        kwargs['sregistry'] = SymbolRegistry()
        expressions = as_tuple(expressions)

        # Input check
        if any(not isinstance(i, Evaluable) for i in expressions):
            raise InvalidOperator("Only `devito.Evaluable` are allowed.")

        # Enable recursive lowering
        # This may be used by a compilation pass that constructs a new
        # expression for which a partial or complete lowering is desired
        kwargs['lower'] = cls._lower

        # [Eq] -> [LoweredEq]
        expressions = cls._lower_exprs(expressions, **kwargs)

        conv = ExtractDevitoStencilConversion(expressions)
        module = conv.convert()
        convert_devito_stencil_to_xdsl_stencil(module, timed=True)

        # [LoweredEq] -> [Clusters]
        clusters = cls._lower_clusters(expressions, **kwargs)

        # [Clusters] -> ScheduleTree
        stree = cls._lower_stree(clusters, **kwargs)

        # ScheduleTree -> unbounded IET
        uiet = cls._lower_uiet(stree, **kwargs)

        # unbounded IET -> IET
        iet, byproduct = cls._lower_iet(uiet, **kwargs)

        return IRs(expressions, clusters, stree, uiet, iet), byproduct, module


    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._jit_compile()
            self.setup_memref_args()
            self._lib = self._compiler.load(self._tf.name)
            self._lib.name = self._tf.name

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, "apply_kernel")
            # Associate a C type to each argument for runtime type check
            argtypes = self._construct_cfunction_args(self._jit_kernel_constants,
                                                      get_types=True)
            self._cfunction.argtypes = argtypes

        return self._cfunction

    def _construct_cfunction_args(self, args, get_types=False):
        """
        Either construct the args for the cfunction, or construct the
        arg types for it.
        """
        ps = {
            p._C_name: p._C_ctype for p in self.parameters
        }

        things = []
        things_types = []

        for name in get_arg_names_from_module(self._module):
            thing = args[name]
            things.append(thing)
            if name in ps:
                things_types.append(ps[name])
            else:
                things_types.append(type(thing))

        if get_types:
            return things_types
        else:
            return things


def get_arg_names_from_module(op):
    return [
        str_attr.data for str_attr in op.body.block.ops.first.attributes['param_names'].data  # noqa
    ]
