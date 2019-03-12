"""A pure-Python Python bytecode interpreter."""
# Based on:
# pyvm2 by Paul Swartz (z3p), from http://www.twistedmatrix.com/users/z3p/

import dis
import linecache
import logging
import sys

import six
from six.moves import reprlib

from byterun.instruction import VirtualMachine_instruction, VirtualMachineError
from byterun.pyobj import Frame

log = logging.getLogger(__name__)


# Create a repr that won't overflow.
repr_obj = reprlib.Repr()
repr_obj.maxother = 120
repper = repr_obj.repr


class VirtualMachine(VirtualMachine_instruction):
    def __init__(self):
        super().__init__()

    def make_frame(self, code, callargs=None, f_globals=None, f_locals=None, f_closure=None):
        callargs = callargs if callargs else {}
        log.info(f"make_frame: code={code}, callargs={callargs}")
        # 全局变量非空，局部变量为空时
        if f_globals is not None and f_locals is None:
            f_locals = f_globals
        # 调用栈非空时
        elif self.frames:
            f_globals = self.frame.f_globals
            f_locals = {}
        else:
            f_globals = f_locals = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__doc__': None,
                '__package__': None,
            }
        f_locals.update(callargs)
        frame = Frame(code, f_globals, f_locals, f_closure, self.frame)
        return frame

    def push_frame(self, frame):
        self.frames.append(frame)
        self.frame = frame

    def pop_frame(self):
        self.frames.pop()
        if self.frames:
            self.frame = self.frames[-1]
        else:
            self.frame = None

    def print_frames(self):
        """Print the call stack, for debugging."""
        for f in self.frames:
            filename = f.f_code.co_filename
            lineno = f.line_number()
            print(f'  File "{filename}", line {lineno}, in {f.f_code.co_name}')
            linecache.checkcache(filename)
            line = linecache.getline(filename, lineno, f.f_globals)
            if line:
                print('    ' + line.strip())

    def resume_frame(self, frame):
        frame.f_back = self.frame
        val = self.run_frame(frame)
        frame.f_back = None
        return val

    def run_frame(self, frame):
        """Run a frame until it returns (somehow).
        Exceptions are raised, the return value is returned.
        """
        self.push_frame(frame)
        while True:
            byteName, arguments, opoffset = self.parse_byte_and_args()
            print(byteName, arguments, opoffset)
            if log.isEnabledFor(logging.INFO):
                self.log(byteName, arguments, opoffset)

            # When unwinding the block stack, we need to keep track of why we
            # are doing it.
            why = self.dispatch(byteName, arguments)
            if why == 'exception':
                # TODO: ceval calls PyTraceBack_Here, not sure what that does.
                pass

            if why == 'reraise':
                why = 'exception'

            if why != 'yield':
                while why and frame.block_stack:
                    # Deal with any block management we need to do.
                    why = self.manage_block_stack(why)

            if why:
                break

        # TODO: handle generator exception state

        self.pop_frame()

        if why == 'exception':
            six.reraise(*self.last_exception)

        return self.return_value

    # 入口点
    def run_code(self, code, f_globals=None, f_locals=None):
        frame = self.make_frame(code, f_globals=f_globals, f_locals=f_locals)
        val = self.run_frame(frame)
        # Check some invariants
        if self.frames:                                 # pragma: no cover
            raise VirtualMachineError("Frames left over!")
        if self.frame and self.frame.stack:             # pragma: no cover
            raise VirtualMachineError(f"Data left on stack! {self.frame.stack}")

        return val

    def parse_byte_and_args(self):
        """ Parse bytecode into an instruction and optionally arguments
        3.6 uses two bytes for every instruction
        instead of a mix of one and three byte instructions."""
        f = self.frame
        opoffset = f.f_lasti
        byteCode = f.f_code.co_code[opoffset]
        f.f_lasti += 2
        byteName = dis.opname[byteCode]
        arg = None
        arguments = []
        if byteCode >= dis.HAVE_ARGUMENT:
            arg = f.f_code.co_code[opoffset + 1]
            if byteCode in dis.hasconst:
                arg = f.f_code.co_consts[arg]
            elif byteCode in dis.hasfree:
                if arg < len(f.f_code.co_cellvars):
                    arg = f.f_code.co_cellvars[arg]
                else:
                    var_idx = arg - len(f.f_code.co_cellvars)
                    arg = f.f_code.co_freevars[var_idx]
            elif byteCode in dis.hasname:
                arg = f.f_code.co_names[arg]
            elif byteCode in dis.hasjrel:
                arg = f.f_lasti + arg
            elif byteCode in dis.hasjabs:
                arg = arg
            elif byteCode in dis.haslocal:
                arg = f.f_code.co_varnames[arg]
            else:
                arg = arg
            if self.EXTENDED_ARG_ext:
                arg += self.EXTENDED_ARG_ext * 256
                self.EXTENDED_ARG_ext = 0
            arguments = [arg]

        return byteName, arguments, opoffset

    def dispatch(self, byteName, arguments):
        """ Dispatch by bytename to the corresponding methods.
        Exceptions are caught and set on the virtual machine."""
        why = None
        try:
            if byteName.startswith('UNARY_'):
                self.unaryOperator(byteName[6:])
            elif byteName.startswith('BINARY_'):
                self.binaryOperator(byteName[7:])
            elif byteName.startswith('INPLACE_'):
                self.inplaceOperator(byteName[8:])
            elif 'SLICE+' in byteName:
                self.sliceOperator(byteName)
            else:
                # dispatch
                bytecode_fn = getattr(self, f'byte_{byteName}', None)
                if not bytecode_fn:            # pragma: no cover
                    raise VirtualMachineError(
                        f"unknown bytecode type: {byteName}"
                    )
                why = bytecode_fn(*arguments)

        except:
            # deal with exceptions encountered while executing the op.
            self.last_exception = sys.exc_info()[:2] + (None,)
            log.exception("Caught exception during execution")
            why = 'exception'

        return why

    def log(self, byteName, arguments, opoffset):
        """ Log arguments, block stack, and data stack for each opcode."""
        op = f"{opoffset}: {byteName}"
        if arguments:
            op += f" {arguments[0]}"
        indent = "    "*(len(self.frames)-1)
        stack_rep = repper(self.frame.stack)
        block_stack_rep = repper(self.frame.block_stack)

        log.info(f"  {indent}data: {stack_rep}")
        log.info(f"  {indent}blks: {block_stack_rep}")
        log.info(f"{indent}{op}")
