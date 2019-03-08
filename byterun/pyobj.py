"""Implementations of Python fundamental objects for Byterun."""

import collections
import inspect
import types


def make_cell(value):
    # Construct an actual cell object by creating a closure right here,
    # and grabbing the cell object out of the function we create.
    fn = (lambda x: lambda: x)(value)
    return fn.__closure__[0]


class Function:
    __slots__ = [
        'func_code', 'func_name', 'func_defaults', 'func_globals',
        'func_locals', 'func_dict', 'func_closure',
        '__name__', '__dict__', '__doc__',
        '_vm', '_func',
    ]

    def __init__(self, name, code, globs, defaults, closure, vm):
        self._vm = vm
        self.func_code = code
        self.func_name = self.__name__ = name or code.co_name
        self.func_defaults = tuple(defaults)
        self.func_globals = globs
        self.func_locals = self._vm.frame.f_locals
        self.__dict__ = {}
        self.func_closure = closure
        self.__doc__ = code.co_consts[0] if code.co_consts else None

        # Sometimes, we need a real Python function.  This is for that.
        kw = {
            'argdefs': self.func_defaults,
        }
        if closure:
            kw['closure'] = tuple(make_cell(0) for _ in closure)
        self._func = types.FunctionType(code, globs, **kw)

    def __repr__(self):         # pragma: no cover
        return f'<Function {self.func_name} at 0x{id(self):<8}>'

    def __get__(self, instance, owner):
        if instance is not None:
            return Method(instance, owner, self)
        else:
            return self

    def __call__(self, *args, **kwargs):
        callargs = inspect.getcallargs(self._func, *args, **kwargs)

        # class inspect.Parameter
        # CPython generates implicit parameter names of the form .0
        # on the code objects used to implement comprehensions
        # and generator expressions.
        # Changed in version 3.6: These parameter names are
        # exposed by this module as names like implicit0.
        if 'implicit0' in callargs:
            callargs['.0'] = callargs['implicit0']

        frame = self._vm.make_frame(
            self.func_code, callargs, self.func_globals, {}
        )
        CO_GENERATOR = 32           # flag for "this code uses yield"
        if self.func_code.co_flags & CO_GENERATOR:
            gen = Generator(frame, self._vm)
            frame.generator = gen
            retval = gen
        else:
            retval = self._vm.run_frame(frame)
        return retval


class Method:
    def __init__(self, obj, _class, func):
        self.im_self = obj
        self.im_class = _class
        self.im_func = func

    def __repr__(self):         # pragma: no cover
        name = f"{self.im_class.__name__}.{self.im_func.func_name}"
        if self.im_self is not None:
            return f'<Bound Method {name} of {self.im_self}>'
        else:
            return f'<Unbound Method {name}s>'

    def __call__(self, *args, **kwargs):
        if self.im_self is not None:
            return self.im_func(self.im_self, *args, **kwargs)
        else:
            return self.im_func(*args, **kwargs)


class Cell:
    """A fake cell for closures.

    Closures keep names in scope by storing them not in a frame, but in a
    separate object called a cell.  Frames share references to cells, and
    the LOAD_DEREF and STORE_DEREF opcodes get and set the value from cells.

    This class acts as a cell, though it has to jump through two hoops to make
    the simulation complete:

        1. In order to create actual FunctionType functions, we have to have
           actual cell objects, which are difficult to make. See the twisty
           double-lambda in __init__.

        2. Actual cell objects can't be modified, so to implement STORE_DEREF,
           we store a one-element list in our cell, and then use [0] as the
           actual value.

    """
    def __init__(self, value):
        self.contents = value

    def get(self):
        return self.contents

    def set(self, value):
        self.contents = value


Block = collections.namedtuple("Block", "type, handler, level")


class Frame:
    def __init__(self, f_code, f_globals, f_locals, f_back):
        self.f_code = f_code                        # 当前frame被执行的code object
        self.f_globals = f_globals                  # 当前frame的全局变量
        self.f_locals = f_locals                    # 当前frame的局部变量
        self.f_back = f_back                        # 当前frame之前的frame，如果当前frame位于栈底，则为空
        self.stack = []                             # 当前frame的数据栈
        if f_back:
            self.f_builtins = f_back.f_builtins     # 当前frame的内置变量
        else:
            self.f_builtins = f_locals['__builtins__']
            if hasattr(self.f_builtins, '__dict__'):
                self.f_builtins = self.f_builtins.__dict__

        self.f_lineno = f_code.co_firstlineno       # 当前frame执行的具体行数（只针对栈底的frame）
        self.f_lasti = 0                            # 当前frame中code object字符串的当前下标

        if f_code.co_cellvars:
            self.cells = {}
            if not f_back.cells:
                f_back.cells = {}
            for var in f_code.co_cellvars:
                # Make a cell for the variable in our locals, or None.
                cell = Cell(self.f_locals.get(var))
                f_back.cells[var] = self.cells[var] = cell
        else:
            self.cells = None

        if f_code.co_freevars:
            if not self.cells:
                self.cells = {}
            for var in f_code.co_freevars:
                assert self.cells is not None
                assert f_back.cells, f"f_back.cells: {f_back.cells}"
                self.cells[var] = f_back.cells[var]

        self.block_stack = []                       # 当前frame的块栈
        self.generator = None

    def __repr__(self):         # pragma: no cover
        return f'<Frame at {id(self):<8}: ' \
               f'{self.f_code.co_filename} @ {self.f_lineno}>'

    def line_number(self):
        """Get the current line number the frame is executing."""
        # We don't keep f_lineno up to date, so calculate it based on the
        # instruction address and the line number table.
        lnotab = self.f_code.co_lnotab
        byte_increments = lnotab[0::2]
        line_increments = lnotab[1::2]

        byte_num = 0
        line_num = self.f_code.co_firstlineno

        for byte_incr, line_incr in zip(byte_increments, line_increments):
            byte_num += byte_incr
            if byte_num > self.f_lasti:
                break
            line_num += line_incr

        return line_num


class Generator:
    def __init__(self, g_frame, vm):
        self.gi_frame = g_frame
        self.vm = vm
        self.started = False
        self.finished = False

    def __iter__(self):
        return self

    def next(self):
        return self.send(None)

    def send(self, value=None):
        if not self.started and value is not None:
            raise TypeError("Can't send non-None value to a just-started generator")
        self.gi_frame.stack.append(value)
        self.started = True
        val = self.vm.resume_frame(self.gi_frame)
        if self.finished:
            raise StopIteration(val)
        return val

    __next__ = next
