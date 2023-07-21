from devito import Operator
from devito.ir.ietxdsl import transform_devito_to_iet_ssa, iet_to_standard_mlir, finalize_module_with_globals
from devito.logger import perf

import os
import tempfile
import subprocess
import ctypes
import numpy as np

from collections import OrderedDict
from io import StringIO

from devito.exceptions import InvalidOperator
from devito.logger import perf
from devito.ir.iet import Callable, MetaCall
from devito.ir.support import SymbolRegistry
from devito.operator.operator import IRs
from devito.operator.profiling import create_profile
from devito.tools import OrderedSet, as_tuple, flatten, filter_sorted
from devito.types import Evaluable, TimeFunction
from devito.types.mlir_types import ptr_of, f32

from mpi4py import MPI

from xdsl.printer import Printer

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


CFLAGS = "-O3 -march=native -mtune=native -lmlir_c_runner_utils"

MLIR_CPU_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,expand-strided-metadata, loop-invariant-code-motion,lower-affine,convert-scf-to-cf,convert-math-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv},finalize-memref-to-llvm,canonicalize,cse)"'
MLIR_OPENMP_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,expand-strided-metadata, loop-invariant-code-motion,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm,convert-scf-to-cf,convert-func-to-llvm{use-bare-ptr-memref-call-conv},convert-openmp-to-llvm,convert-math-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"'
# gpu-launch-sink-index-computations seemed to have no impact
MLIR_GPU_PIPELINE = '"builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=128,1,1},func.func(gpu-map-parallel-loops),convert-parallel-loops-to-gpu,fold-memref-alias-ops,expand-strided-metadata,lower-affine,gpu-kernel-outlining,canonicalize,cse,convert-arith-to-llvm{index-bitwidth=64},finalize-memref-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},canonicalize,cse,gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize,cse)"'

XDSL_CPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir,printf-to-llvm"
XDSL_GPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir{target=gpu},printf-to-llvm"
XDSL_MPI_PIPELINE = lambda decomp: f'"dmp-decompose-2d{decomp},convert-stencil-to-ll-mlir,dmp-to-mpi{{mpi_init=false}},lower-mpi,printf-to-llvm"'


class XDSLOperator(Operator):

    def __new__(cls, expressions, **kwargs):
        self = super(XDSLOperator, cls).__new__(cls, expressions, **kwargs)
        self._tf = tempfile.NamedTemporaryFile(prefix="devito-jit-", suffix='.so')
        self._interop_tf = tempfile.NamedTemporaryFile(prefix="devito-jit-interop-", suffix=".o")
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
        dist = self.functions[0].grid.distributor
        return dist.topology, dist.myrank

    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.
        It is ensured that JIT compilation will only be performed once per
        Operator, reagardless of how many times this method is invoked.
        """
        #ccode = transform_devito_xdsl_string(self)
        #self.ccode = ccode
        with self._profiler.timer_on('jit-compile'):
            is_mpi = MPI.Is_initialized()
            is_gpu = os.environ.get("DEVITO_PLATFORM", None) == 'nvidiaX'
            is_omp = os.environ.get("DEVITO_LANGUAGE", None) == 'openmp'

            if is_mpi and is_gpu:
                raise RuntimeError("Cannot run MPI+GPU for now!")

            if is_omp and is_gpu:
                raise RuntimeError("Cannot run OMP+GPU!")

            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            xdsl_pipeline = XDSL_CPU_PIPELINE
            mlir_pipeline = MLIR_CPU_PIPELINE

            if is_omp:
                mlir_pipeline = MLIR_OPENMP_PIPELINE

            if is_mpi:
                shape, mpi_rank = self.mpi_shape
                # run with restrict domain=false so we only introduce the swaps but don't
                # reduce the domain of the computation (as devito has already done that for us)
                slices = ','.join(str(x) for x in shape)
                decomp = f"{{strategy=2d-grid slices={slices} restrict_domain=false}}"
                xdsl_pipeline = XDSL_MPI_PIPELINE(decomp)
            elif is_gpu:
                xdsl_pipeline = XDSL_GPU_PIPELINE
                mlir_pipeline = MLIR_GPU_PIPELINE

            # allow jit backdooring to provide your own xdsl code
            backdoor = os.getenv('XDSL_JIT_BACKDOOR')
            if backdoor is not None:
                print("JIT Backdoor: loading xdsl file from: " + backdoor)
                with open(backdoor, 'r') as f:
                    module_str = f.read()
            source_name = os.path.splitext(self._tf.name)[0] + ".mlir"
            source_file = open(source_name, "w")
            source_file.write(module_str)
            source_file.close()
            # compile IR using xdsl-opt | mlir-opt | mlir-translate | clang
            try:
                cflags = CFLAGS
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
                cmd = 'set -eo pipefail; '\
                    f'xdsl-opt {source_name} -p {xdsl_pipeline} |' \
                    f'mlir-opt -p {mlir_pipeline} | ' \
                    f'mlir-translate --mlir-to-llvmir | ' \
                    f'{cc} {cflags} -shared -o {self._tf.name} {self._interop_tf.name} -xir -'
                print(cmd)
                res = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    capture_output=True,
                    executable="/bin/bash"
                )

                if res.returncode != 0:
                    print("compilation failed with output:")
                    print(res.stderr)

                assert res.returncode == 0
            except Exception as ex:
                print("error")
                raise ex
            # print(res.stderr)

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))

    @property
    def _soname(self):
        return self._tf.name

    def setup_memref_args(self):
        """
        Add memrefs to args dictionary so they can be passed to the cfunction
        """
        args = dict()
        for arg in self.functions:
            if isinstance(arg, TimeFunction):
                data = arg._data_allocated
                # iterate over the first dimension (time)
                for t in range(data.shape[0]):
                    args[f'{arg._C_name}_{t}'] = data[t, ...].ctypes.data_as(ptr_of(f32))
        self._jit_kernel_constants.update(args)

    def _construct_cfunction_args(self, args, get_types = False):
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
            self._cfunction.argtypes = self._construct_cfunction_args(self._jit_kernel_constants, get_types=True)

        return self._cfunction

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

        from devito.ir.ietxdsl.cluster_to_ssa import ExtractDevitoStencilConversion, convert_devito_stencil_to_xdsl_stencil
        conv = ExtractDevitoStencilConversion(expressions)
        module = conv.convert()
        convert_devito_stencil_to_xdsl_stencil(module)

        # [LoweredEq] -> [Clusters]
        clusters = cls._lower_clusters(expressions, **kwargs)

        # [Clusters] -> ScheduleTree
        stree = cls._lower_stree(clusters, **kwargs)

        # ScheduleTree -> unbounded IET
        uiet = cls._lower_uiet(stree, **kwargs)

        # unbounded IET -> IET
        iet, byproduct = cls._lower_iet(uiet, **kwargs)

        return IRs(expressions, clusters, stree, uiet, iet), byproduct, module


    @classmethod
    def _build(cls, expressions, **kwargs) -> Callable:
        # Python- (i.e., compile-) and C-level (i.e., run-time) performance
        profiler = create_profile('timers')

        # Lower the input expressions into an IET
        irs, byproduct, module = cls._lower(expressions, profiler=profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **irs.iet.args)
        Callable.__init__(op, **op.args)

        # Header files, etc.
        op._headers = OrderedSet(*cls._default_headers)
        op._headers.update(byproduct.headers)
        op._globals = OrderedSet(*cls._default_globals)
        op._includes = OrderedSet(*cls._default_includes)
        op._includes.update(profiler._default_includes)
        op._includes.update(byproduct.includes)
        op._module = module

        # Required for the jit-compilation
        op._compiler = kwargs['compiler']
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
        op._func_table.update(OrderedDict([(i.root.name, i) for i in byproduct.funcs]))

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

    @property
    def mlircode(self):
        from xdsl.printer import Printer
        from io import StringIO
        file = StringIO()
        Printer(file).print(self._module)
        return file.getvalue()

def get_arg_names_from_module(op):
    return [
        str_attr.data 
        for str_attr in op.body.block.ops.first.attributes['param_names'].data
    ]
