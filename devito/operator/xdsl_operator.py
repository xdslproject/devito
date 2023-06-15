from devito import Operator
from devito.ir.ietxdsl import transform_devito_to_iet_ssa, iet_to_standard_mlir, finalize_module_with_globals
from devito.logger import perf

import tempfile
import subprocess
import ctypes

from collections import OrderedDict
from io import StringIO

from devito.exceptions import InvalidOperator
from devito.logger import perf
from devito.ir.iet import Callable, MetaCall
from devito.ir.support import SymbolRegistry
from devito.operator.operator import IRs
from devito.operator.profiling import create_profile
from devito.tools import OrderedSet, as_tuple, flatten, filter_sorted
from devito.types import Evaluable
from devito.types.mlir_types import make_memref_f32_struct_from_np

from xdsl.printer import Printer

__all__ = ['XDSLOperator']

DO_MPI = False

# TODO: get this from mpi4py! or devito or something!
mpi_grid = (2,2)
# run with restrict domain=false so we only introduce the swaps but don't
# reduce the domain of the computation (as devito has already done that for us)
decomp = f"{{strategy=2d-grid slices={','.join(str(x) for x in mpi_grid)} restrict_domain=false}}"

CFLAGS = "-O3 -march=native -mtune=native"

MLIR_CPU_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,finalize-memref-to-llvm,convert-scf-to-cf,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"'
MLIR_OPENMP_PIPELINE = '"builtin.module(canonicalize, cse, loop-invariant-code-motion, canonicalize, cse, loop-invariant-code-motion,cse,canonicalize,fold-memref-alias-ops,lower-affine,finalize-memref-to-llvm,loop-invariant-code-motion,canonicalize,cse,convert-scf-to-openmp,finalize-memref-to-llvm,convert-scf-to-cf,convert-openmp-to-llvm,convert-math-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts,canonicalize,cse)"'
# gpu-launch-sink-index-computations seemed to have no impact
MLIR_GPU_PIPELINE = '"builtin.module(test-math-algebraic-simplification,scf-parallel-loop-tiling{parallel-loop-tile-sizes=128,1,1},func.func(gpu-map-parallel-loops),convert-parallel-loops-to-gpu,fold-memref-alias-ops,lower-affine,gpu-kernel-outlining,canonicalize,cse,convert-arith-to-llvm{index-bitwidth=64},finalize-memref-to-llvm{index-bitwidth=64},convert-scf-to-cf,convert-cf-to-llvm{index-bitwidth=64},canonicalize,cse,gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,canonicalize,gpu-to-cubin),gpu-to-llvm,canonicalize,cse)"'

XDSL_CPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir"
XDSL_GPU_PIPELINE = "stencil-shape-inference,convert-stencil-to-ll-mlir{target=gpu}"
XDSL_MPI_PIPELINE = f'"dmp-decompose-2d{decomp},convert-stencil-to-ll-mlir,dmp-to-mpi{{mpi_init=false}},lower-mpi"'


class XDSLOperator(Operator):

    def __new__(cls, expressions, **kwargs):
        self = super(XDSLOperator, cls).__new__(cls, expressions, **kwargs)
        self._tf = tempfile.NamedTemporaryFile(suffix='.so')
        self.__class__ = cls
        return self

    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.
        It is ensured that JIT compilation will only be performed once per
        Operator, reagardless of how many times this method is invoked.
        """
        #ccode = transform_devito_xdsl_string(self)
        #self.ccode = ccode
        with self._profiler.timer_on('jit-compile'):
            
            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            xdsl_pipeline = XDSL_MPI_PIPELINE if DO_MPI else XDSL_CPU_PIPELINE

            # compile IR using xdsl-opt | mlir-opt | mlir-translate | clang
            try:
                cmd = f'xdsl-opt -p {xdsl_pipeline} |' \
                    f'mlir-opt -p {MLIR_CPU_PIPELINE} | ' \
                    f'mlir-translate --mlir-to-llvmir | ' \
                    f'clang -O3 -shared -xir - -o {self._tf.name}'
                print(f"compiling kernel using {cmd}")
                res = subprocess.run(
                    cmd,
                    shell=True,
                    input=module_str,
                    text=True
                )
                assert res.returncode == 0
            except Exception as ex:
                print("error")
                raise ex
            #print(res.stderr)

        elapsed = self._profiler.py_timers['jit-compile']
        
        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
                    (self.name, self._tf.name, elapsed))
        
    @property
    def _soname(self):
        return self._tf.name

    def setup_memref_args(self):
        """
        Add memrefs to args dictionary so they can be passed to the cfunction
        """
        self._memref_cache = dict()
        for arg in self.parameters:
            if hasattr(arg, 'data_with_halo') and isinstance(arg.data_with_halo, np.ndarray):
                # TODO: is this even correct lol?
                data = arg.data_with_halo
                time_slices = data.shape[0]
                for t in range(time_slices):
                    self._memref_cache[f'{arg._C_name}_{t}'] = make_memref_f32_struct_from_np(data[t, ...])
        print(f"constructed memrefs {list(self._memref_cache.keys())}")
        self._jit_kernel_constants.update(self._memref_cache)

    def _construct_cfunction_args(self, args):
        return [args[name] for name in get_arg_names_from_module(self._module)]

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
            self._cfunction.argtypes = [i._C_ctype for i in self._construct_cfunction_args(self._jit_kernel_constants)]

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

        #from xdsl.printer import Printer
        #p = Printer(target=Printer.Target.MLIR)
        #p.print(module)
        #import sys

        #cls._stencil_module = module
        #sys.exit(0)

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
