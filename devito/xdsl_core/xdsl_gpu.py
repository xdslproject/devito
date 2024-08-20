from contextlib import redirect_stdout
import io
import os
import sys
from io import StringIO

from devito.arch.archinfo import get_nvidia_cc

from devito.xdsl_core.xdsl_cpu import XdslAdvOperator

from devito.ir.xdsl_iet.cluster_to_ssa import finalize_module_with_globals
from devito.mpi import MPI

from devito.logger import info, perf

from xdsl.printer import Printer
from xdsl.xdsl_opt_main import xDSLOptMain
from devito.passes.iet import DeviceOmpTarget
from devito.xdsl_core.utils import generate_pipeline, generate_mlir_pipeline


__all__ = ['XdslAdvDeviceOperator']


class XdslAdvDeviceOperator(XdslAdvOperator):

    _Target = DeviceOmpTarget

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

            if is_mpi and is_gpu:
                raise RuntimeError("Cannot run MPI+GPU for now!")

            # specialize the code for the specific apply parameters
            finalize_module_with_globals(self._module, self._jit_kernel_constants,
                                         self.name)

            # print module as IR
            module_str = StringIO()
            Printer(stream=module_str).print(self._module)
            module_str = module_str.getvalue()

            xdsl_pipeline = generate_XDSL_GPU_PIPELINE()
            # Get GPU blocking shapes
            block_sizes: list[int] = [min(target, self._jit_kernel_constants.get(f"{dim}_size", 1)) for target, dim in zip([32, 4, 8], ["x", "y", "z"])]  # noqa
            block_sizes = ','.join(str(bs) for bs in block_sizes)
            mlir_pipeline = generate_MLIR_GPU_PIPELINE(block_sizes)

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

                cflags += " -lmlir_cuda_runtime "
                cflags += " -shared "

                # TODO More detailed error handling manually,
                # instead of relying on a bash-only feature.

                # xdsl-opt, get xDSL IR
                # TODO: Remove quotes in pipeline; currently workaround with [1:-1]
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

        perf("XDSLAdvDeviceOperator `%s` jit-compiled `%s` in %.2f s with `mlir-opt`" %
             (self.name, source_name, elapsed))


def generate_XDSL_GPU_PIPELINE():
    passes = [
        "shape-inference",
        "convert-stencil-to-ll-mlir",
        "reconcile-unrealized-casts",
        "printf-to-llvm",
        "canonicalize"
    ]

    return generate_pipeline(passes)


# gpu-launch-sink-index-computations seemed to have no impact
def generate_MLIR_GPU_PIPELINE(block_sizes):
    return generate_pipeline([
        generate_mlir_pipeline([
            "test-math-algebraic-simplification",
            f"scf-parallel-loop-tiling{{parallel-loop-tile-sizes={block_sizes}}}",
        ]),
        "gpu-map-parallel-loops",
        generate_mlir_pipeline([
            "convert-parallel-loops-to-gpu",
            "lower-affine",
            "canonicalize",
            "cse",
            "fold-memref-alias-ops",
            "gpu-launch-sink-index-computations",
            "gpu-kernel-outlining",
            "canonicalize{region-simplify}",
            "cse",
            "fold-memref-alias-ops",
            "expand-strided-metadata",
            "lower-affine",
            "canonicalize",
            "cse",
            "func.func(gpu-async-region)",
            "canonicalize",
            "cse",
            "convert-arith-to-llvm{index-bitwidth=64}",
            "convert-scf-to-cf",
            "convert-cf-to-llvm{index-bitwidth=64}",
            "canonicalize",
            "cse",
            "convert-func-to-llvm{use-bare-ptr-memref-call-conv}",
            f"nvvm-attach-target{{O=3 ftz fast chip=sm_{get_nvidia_cc()}}}",
            "gpu.module(convert-gpu-to-nvvm,canonicalize,cse)",
            "gpu-to-llvm",
            "gpu-module-to-binary",
            "canonicalize",
            "cse"
        ]),
    ])[1:-1]
