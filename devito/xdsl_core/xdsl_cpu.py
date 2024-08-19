from collections import OrderedDict
from contextlib import redirect_stdout
import ctypes
import io
import os
import subprocess
import sys
import tempfile

from io import StringIO

from devito.core.operator import CoreOperator
from devito.ir.iet import Callable, MetaCall
from devito.ir.iet.nodes import Section
from devito.ir.iet.visitors import FindNodes
from devito.logger import info, perf
from devito.mpi import MPI
from devito.operator.profiling import create_profile
from devito.tools import filter_sorted, flatten, as_tuple

from xdsl.printer import Printer
from xdsl.xdsl_opt_main import xDSLOptMain

from devito.ir.xdsl_iet.cluster_to_ssa import (ExtractDevitoStencilConversion,
                                               finalize_module_with_globals,
                                               setup_memref_args)  # noqa

from devito.ir.xdsl_iet.profiling import apply_timers
from devito.passes.iet import CTarget, OmpTarget
from devito.core.cpu import Cpu64OperatorMixin
from devito.xdsl_core.utils import generate_pipeline, generate_mlir_pipeline


__all__ = ['XdslnoopOperator', 'XdslAdvOperator']


class XdslnoopOperator(Cpu64OperatorMixin, CoreOperator):

    # This operator needs more testing as we currently compare the starting
    # initial generated code against the advanced one
    _Target = CTarget

    @classmethod
    def _build(cls, expressions, **kwargs):

        # Lots of duplicate code, to drop
        perf("Building an xDSL operator")
        # Python- (i.e., compile-) and C-level (i.e., run-time) performance
        profiler = create_profile('timers')

        # Lower the input expressions into an IET and a module. iet is not used
        perf("Lower expressions to a module")
        irs, byproduct = cls._lower(expressions, profiler=profiler, **kwargs)

        # Make it an actual Operator
        op = Callable.__new__(cls, **irs.iet.args)
        Callable.__init__(op, **op.args)

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
        op._func_table.update(OrderedDict([(i.root.name, i) for i in byproduct.funcs]))

        # Produced by the various compilation passes

        op._reads = filter_sorted(flatten(e.reads for e in irs.expressions))
        op._writes = filter_sorted(flatten(e.writes for e in irs.expressions))
        op._dimensions = set().union(*[e.dimensions for e in irs.expressions])
        op._dtype, op._dspace = irs.clusters.meta
        op._profiler = profiler

        # This has to be moved outside and drop this _build from here

        module = cls._lower_stencil(expressions, **kwargs)

        num_sections = len(FindNodes(Section).visit(irs.iet))
        if num_sections:
            apply_timers(module, **kwargs)

        op._module = module

        return op

    @classmethod
    def _lower_stencil(cls, expressions, **kwargs):
        """
        Lower the input expressions into an xDSL builtin.ModuleOp
        [Eq] -> [xdsl]
        Apply timers to the module
        """
        conv = ExtractDevitoStencilConversion(cls)
        module = conv.convert(as_tuple(expressions), **kwargs)

        return module

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
            finalize_module_with_globals(self._module, self._jit_kernel_constants)

            # Print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            xdsl_pipeline = generate_XDSL_CPU_noop_PIPELINE()
            mlir_pipeline = generate_MLIR_CPU_noop_PIPELINE()

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

            # Uncomment to print the module_str
            # Printer().print(module_str)
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

                cflags += " -shared "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                # Run the first pipeline, mostly xDSL-centric
                xdsl_args = [source_name,
                             "--allow-unregistered-dialect",
                             "-p",
                             xdsl_pipeline[1:-1],]
                # We use the Python API to run xDSL rather than a subprocess
                # This avoids reimport overhead
                xdsl = xDSLOptMain(args=xdsl_args)
                out = io.StringIO()
                perf("-----------------")
                perf(f"xdsl-opt {' '.join(xdsl_args)}")
                xdsl = xDSLOptMain(args=xdsl_args)
                out = io.StringIO()
                with redirect_stdout(out):
                    xdsl.run()

                # To use as input in the next stage
                out.seek(0)
                # Run the second pipeline, mostly MLIR-centric
                xdsl_mlir_args = ["--allow-unregistered-dialect",
                                  "-p",
                                  mlir_pipeline]
                # We drive it though xDSL rather than a mlir-opt call for:
                # - ability to use xDSL replacement passes in the middle
                # - Avoiding complex process cmanagement code here: xDSL provides
                xdsl = xDSLOptMain(args=xdsl_mlir_args)
                out2 = io.StringIO()
                perf("-----------------")
                perf(f"xdsl-opt {' '.join(xdsl_mlir_args)}")
                with redirect_stdout(out2):
                    old_stdin = sys.stdin
                    sys.stdin = out
                    xdsl.run()
                    sys.stdin = old_stdin

                # mlir-translate to translate to LLVM-IR
                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                out = self.compile(mlir_translate_cmd, out2.getvalue())

                # Compile with clang and get LLVM-IR
                clang_cmd = f'{cc} {cflags} -o {self._tf.name} {self._interop_tf.name} -xir -'  # noqa
                out = self.compile(clang_cmd, out)

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

    def apply(self, **kwargs):
        # Build the arguments list to invoke the kernel function
        with self._profiler.timer_on('arguments'):
            args = self.arguments(**kwargs)
            self._jit_kernel_constants = args

        cfunction = self.cfunction
        try:
            # Invoke kernel function with args
            arg_values = self._construct_cfunction_args(args)
            with self._profiler.timer_on('apply', comm=args.comm):
                cfunction(*arg_values)
        except ctypes.ArgumentError as e:
            if e.args[0].startswith("argument "):
                argnum = int(e.args[0][9:].split(':')[0]) - 1
                newmsg = "error in argument '%s' with value '%s': %s" % (
                    self.parameters[argnum].name,
                    arg_values[argnum],
                    e.args[0])
                raise ctypes.ArgumentError(newmsg) from e
            else:
                raise

        # Post-process runtime arguments
        self._postprocess_arguments(args, **kwargs)

        # Output summary of performance achieved
        return self._emit_apply_profiling(args)

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:

            delete = not os.getenv("XDSL_SKIP_CLEAN", False)
            self._tf = tempfile.NamedTemporaryFile(prefix="devito-jit-", suffix='.so',
                                                   delete=delete)
            self._interop_tf = tempfile.NamedTemporaryFile(prefix="devito-jit-interop-",
                                                           suffix=".o", delete=delete)
            self._make_interop_o()
            self._jit_compile()
            self._jit_kernel_constants.update(setup_memref_args(self.functions))
            self._lib = self._compiler.load(self._tf.name)
            self._lib.name = self._tf.name

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            # argtypes = self._construct_cfunction_args(self._jit_kernel_constants,
            # get_types=True)
            # self._cfunction.argtypes = argtypes

        return self._cfunction

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

    def compile(self, cmd, stdout=None):
        # Execute each command and store the outputs
        outputs = []
        return_code, stdout, stderr = self._cmd_compile(cmd, stdout)
        # Use DEVITO_LOGGING=DEBUG to print
        perf("-----------------")
        perf(cmd)
        outputs.append({
            'command': cmd,
            'return_code': return_code,
            'stdout': stdout,
            'stderr': stderr
        })

        return stdout

    def _construct_cfunction_types(self, args):
        # Unused, maybe drop
        ps = {p._C_name: p._C_ctype for p in self.parameters}

        objects_types = []
        for name in get_arg_names_from_module(self._module):
            if name in ps:
                object_type = ps[name]
                if object_type == DiscreteFunction._C_ctype:  # noqa
                    object_type = dict(object_type._type_._fields_)['data']
                objects_types.append(object_type)
            else:
                objects_types.append(type(object))
        return objects_types

    def _construct_cfunction_args(self, args):
        """
        Either construct the args for the cfunction, or construct the
        arg types for it.
        """

        objects = []
        for name in get_arg_names_from_module(self._module):
            object = args[name]
            objects.append(object)

        return objects


class XdslAdvOperator(XdslnoopOperator):

    _Target = OmpTarget

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
            finalize_module_with_globals(self._module, self._jit_kernel_constants)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            to_tile = len(list(filter(lambda d: d.is_Space, self.dimensions)))-1

            xdsl_pipeline = generate_XDSL_CPU_PIPELINE(to_tile)

            mlir_pipeline = generate_MLIR_CPU_PIPELINE()

            if is_omp:
                # We collapse as many loops as we tile
                kwargs = {'num_loops': to_tile}
                mlir_pipeline = generate_MLIR_OPENMP_PIPELINE(kwargs)

            if is_mpi:
                shape, _ = self.mpi_shape
                # Run with restrict domain=false so we only introduce the swaps but don't
                # reduce the domain of the computation
                # (as devito has already done that for us)
                slices = ','.join(str(x) for x in shape)

                decomp = "2d-grid" if len(shape) == 2 else "3d-grid"

                decomp = f"{{strategy={decomp} slices={slices} restrict_domain=false}}"
                xdsl_pipeline = generate_XDSL_MPI_PIPELINE(decomp, to_tile)

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

            # Uncomment to print the module_str
            # Printer().print(module_str)
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
                    cflags += " -fopenmp"
                if is_gpu:
                    cflags += " -lmlir_cuda_runtime"

                cflags += " -shared "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                # xdsl-opt, get xDSL IR
                # TODO: Remove quotes in pipeline; currently workaround with [1:-1]
                # Run the first pipeline, mostly xDSL-centric
                xdsl_args = [source_name,
                             "--allow-unregistered-dialect",
                             "--disable-verify",
                             "-p",
                             xdsl_pipeline[1:-1],]
                # We use the Python API to run xDSL rather than a subprocess
                # This avoids reimport overhead
                xdsl = xDSLOptMain(args=xdsl_args)
                out = io.StringIO()
                perf("-----------------")
                perf(f"xdsl-opt {' '.join(xdsl_args)}")
                xdsl = xDSLOptMain(args=xdsl_args)
                out = io.StringIO()
                with redirect_stdout(out):
                    xdsl.run()

                # To use as input in the next stage
                out.seek(0)
                # Run the second pipeline, mostly MLIR-centric
                xdsl_mlir_args = ["--allow-unregistered-dialect",
                                  "-p",
                                  mlir_pipeline]
                # We drive it though xDSL rather than a mlir-opt call for:
                # - ability to use xDSL replacement passes in the middle
                # - Avoiding complex process cmanagement code here: xDSL provides
                xdsl = xDSLOptMain(args=xdsl_mlir_args)
                out2 = io.StringIO()
                perf("-----------------")
                perf(f"xdsl-opt {' '.join(xdsl_mlir_args)}")
                with redirect_stdout(out2):
                    old_stdin = sys.stdin
                    sys.stdin = out
                    xdsl.run()
                    sys.stdin = old_stdin

                # mlir-translate to translate to LLVM-IR
                mlir_translate_cmd = 'mlir-translate --mlir-to-llvmir'
                out = self.compile(mlir_translate_cmd, out2.getvalue())
                # Compile with clang and get LLVM-IR
                clang_cmd = f'{cc} {cflags} -o {self._tf.name} {self._interop_tf.name} -xir -'  # noqa
                out = self.compile(clang_cmd, out)

            except Exception as ex:
                print("error")
                raise ex

        elapsed = self._profiler.py_timers['jit-compile']

        perf("XDSLOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))


# -----------XDSL
# This is a collection of xDSL optimization pipelines
# Ideally they should follow the same type of subclassing as the rest of
# the Devito Operatos

def generate_MLIR_CPU_PIPELINE():
    passes = [
        "canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "canonicalize",
        "cse",
        "loop-invariant-code-motion",
        "cse",
        "canonicalize",
        "fold-memref-alias-ops",
        "expand-strided-metadata",
        "loop-invariant-code-motion",
        "lower-affine",
        "convert-scf-to-cf",
        "convert-math-to-llvm",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
        "finalize-memref-to-llvm",
        "canonicalize",
        "cse"
    ]

    return generate_mlir_pipeline(passes)


def generate_MLIR_CPU_noop_PIPELINE():
    passes = [
        "canonicalize",
        "cse",
        # "remove-dead-values",
        "canonicalize",
        "expand-strided-metadata",
        "convert-scf-to-cf",
        "convert-math-to-llvm",
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
        "finalize-memref-to-llvm",
        "canonicalize",
    ]

    return generate_mlir_pipeline(passes)


def generate_MLIR_OPENMP_PIPELINE(kwargs):
    num_loops = kwargs.get('num_loops')

    return generate_pipeline([
        generate_mlir_pipeline([
            "canonicalize",
            "cse",
            "loop-invariant-code-motion",
            "canonicalize",
            "cse",
            "loop-invariant-code-motion",
            "cse",
            "canonicalize",
            "fold-memref-alias-ops",
            "expand-strided-metadata",
            "loop-invariant-code-motion",
            "lower-affine",
            # "finalize-memref-to-llvm",
            # "loop-invariant-code-motion",
            # "canonicalize",
            # "cse",
        ]),
        f"convert-scf-to-openmp{{{generate_collapse_arg(num_loops)}}}",
        generate_mlir_pipeline([
            "finalize-memref-to-llvm",
            "convert-scf-to-cf"
        ]),
        generate_mlir_pipeline([
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
            "convert-openmp-to-llvm",
            "convert-math-to-llvm",
            # "reconcile-unrealized-casts",
            "canonicalize",
            # "print-ir",
            "cse"
        ])
    ])[1:-1]


def generate_XDSL_CPU_PIPELINE(nb_tiled_dims):
    passes = [
        "canonicalize",
        "cse",
        "shape-inference",
        "stencil-bufferize",
        "convert-stencil-to-ll-mlir",
        f"scf-parallel-loop-tiling{{{generate_tiling_arg(nb_tiled_dims)}}}",
        "printf-to-llvm",
        "canonicalize"
    ]

    return generate_pipeline(passes)


def generate_XDSL_CPU_noop_PIPELINE():
    passes = [
        "canonicalize",
        "cse",
        "shape-inference",
        "stencil-bufferize",
        "convert-stencil-to-ll-mlir",
        "printf-to-llvm"
    ]

    return generate_pipeline(passes)


def generate_XDSL_MPI_PIPELINE(decomp, nb_tiled_dims):
    passes = [
        "canonicalize",
        "cse",
        f"distribute-stencil{decomp}",
        "shape-inference",
        "canonicalize-dmp",
        "stencil-bufferize",
        "dmp-to-mpi{mpi_init=false}",
        "convert-stencil-to-ll-mlir",
        f"scf-parallel-loop-tiling{{{generate_tiling_arg(nb_tiled_dims)}}}",
        "lower-mpi",
        "printf-to-llvm",
        "canonicalize"
    ]

    return generate_pipeline(passes)


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


def generate_tiling_arg(nb_tiled_dims: int):
    """
    Generate the tile-sizes arg for the convert-stencil-to-ll-mlir pass.
    Generating no argument if the diled_dims arg is 0
    """
    if nb_tiled_dims < 1:
        return 'parallel-loop-tile-sizes=0'

    # TOFIX: 64 is hardcoded, should be a parameter
    # TOFIX: Zero is also hardcoded, should be a parameter
    return "parallel-loop-tile-sizes=" + ",".join(["64"]*nb_tiled_dims) + ",0"


def generate_collapse_arg(num_loops: int):
    """
    Generate the number of loops that will be collapsed
    Resort to 1 if no number of loops is provided
    """

    if num_loops < 1:
        num_loops = 1

    ret_arg = "collapse=" + "".join(str(num_loops))  # noqa

    return ret_arg


def get_arg_names_from_module(op):
    return [
        str_attr.name_hint for str_attr in op.body.block.ops.first.body.block.args  # noqa
    ]
