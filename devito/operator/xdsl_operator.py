from devito import Operator
from devito.ir.ietxdsl.xdsl_passes import transform_devito_xdsl_string
from devito.logger import debug, perf
from devito.ir.iet import CInterface

__all__ = ['XDSLOperator']


class XDSLOperator(Operator):

    def __new__(cls, expressions, **kwargs):
        return super(XDSLOperator, cls).__new__(cls, expressions, **kwargs)


    def _jit_compile(self):
        """
        JIT-compile the C code generated by the Operator.

        It is ensured that JIT compilation will only be performed once per
        Operator, reagardless of how many times this method is invoked.
        """

        ccode = transform_devito_xdsl_string(self)
        self.ccode = ccode
        if self._lib is None:
            with self._profiler.timer_on('jit-compile'):
                recompiled, src_file = self._compiler.jit_compile(self._soname,
                                                                    ccode)

            elapsed = self._profiler.py_timers['jit-compile']
            if recompiled:
                perf("Operator `%s` jit-compiled `%s` in %.2f s with `%s`" %
                     (self.name, src_file, elapsed, self._compiler))
            else:
                perf("Operator `%s` fetched `%s` in %.2f s from jit-cache" %
                     (self.name, src_file, elapsed))

    def cinterface(self, force=False):
        """
        Generate two files under the prescribed temporary directory:

            * `X.c` (or `X.cpp`): the code generated for this Operator;
            * `X.h`: an header file representing the interface of `X.c`.

        Where `X=self.name`.

        Parameters
        ----------
        force : bool, optional
            Overwrite any existing files. Defaults to False.
        """
        dest = self._compiler.get_jit_dir()
        name = dest.joinpath(self.name)

        cfile = name.with_suffix(".%s" % self._compiler.src_ext)
        hfile = name.with_suffix('.h')

        # Generate the .c and .h code

        ccode = transform_devito_xdsl_string(self)
        ccode_discard, hcode = CInterface().visit(self)

        for f, code in [(cfile, ccode), (hfile, hcode)]:
            if not force and f.is_file():
                debug("`%s` was not saved in `%s` as it already exists" % (f.name, dest))
            else:
                with open(str(f), 'w') as ff:
                    ff.write(str(code))
                debug("`%s` successfully saved in `%s`" % (f.name, dest))

        return ccode, hcode
