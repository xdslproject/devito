from devito import Operator
from devito.ir.ietxdsl import transform_devito_to_iet_ssa, iet_to_standard_mlir
from devito.logger import perf

import tempfile

import subprocess

from io import StringIO


from xdsl.printer import Printer

__all__ = ['XDSLOperator']


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
            
            module_obj = transform_devito_to_iet_ssa(self)

            iet_to_standard_mlir(module_obj)

            module_str = StringIO()
            Printer(target=Printer.Target.MLIR, stream=module_str).print(module_obj)
            module_str = module_str.getvalue()

            f = self._tf.name

            try:
                res = subprocess.run(
                    #f'tee {f}.iet.mlir |'
                    #f'cat /run/user/1000/tmp7lgpw9x1.so.iet.mlir | '
                    f'mlir-opt -cse -loop-invariant-code-motion | '
                    f'mlir-opt -convert-scf-to-cf -convert-cf-to-llvm -convert-arith-to-llvm -convert-math-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | tee {f}.llv.mlir |  '
                    f'mlir-translate --mlir-to-llvmir | '
                    f'steam-run clang -O3 -shared -xir - -o {self._tf.name}',
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

    @property
    def cfunction(self):
        """The JIT-compiled C function as a ctypes.FuncPtr object."""
        if self._lib is None:
            self._jit_compile()
            self._lib = self._compiler.load(self._tf.name)
            self._lib.name = self._tf.name

        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            # Associate a C type to each argument for runtime type check
            self._cfunction.argtypes = [i._C_ctype for i in self.parameters]

        return self._cfunction