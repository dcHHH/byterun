import inspect
import operator
import sys

import six

from byterun.pyobj import Block, Method, Function, Generator, Cell


class VirtualMachineError(Exception):
    """For raising errors in the operation of the VM."""
    pass


class VirtualMachine_instruction:
    def __init__(self):
        # The call stack of frames.
        self.frames = []
        # The current frame.
        self.frame = None
        self.return_value = None
        self.last_exception = None

        self.EXTENDED_ARG_ext = 0

    def top(self):
        """Return the value at the top of the stack, with no changes."""
        return self.frame.stack[-1]

    def pop(self, i=0):
        """Pop a value from the stack.
        Default to the top of the stack, but `i` can be a count from the top
        instead.
        """
        return self.frame.stack.pop(-1-i)

    def push(self, *vals):
        """Push values onto the value stack."""
        self.frame.stack.extend(vals)

    def popn(self, n):
        """Pop a number of values from the value stack.
        A list of `n` values is returned, the deepest value first.
        """
        if n:
            ret = self.frame.stack[-n:]
            self.frame.stack[-n:] = []
            return ret
        else:
            return []

    def peek(self, n):
        """Get a value `n` entries down in the stack, without changing the stack."""
        return self.frame.stack[-n]

    def jump(self, jump):
        """Move the bytecode pointer to `jump`, so it will execute next."""
        self.frame.f_lasti = jump

    # Block stack manipulation
    def push_block(self, type, handler=None, level=None):
        if level is None:
            level = len(self.frame.stack)
        self.frame.block_stack.append(Block(type, handler, level))

    def pop_block(self):
        return self.frame.block_stack.pop()

    def unwind_block(self, block):
        if block.type == 'except-handler':
            offset = 3
        else:
            offset = 0

        while len(self.frame.stack) > block.level + offset:
            self.pop()

        if block.type == 'except-handler':
            tb, value, exctype = self.popn(3)
            self.last_exception = exctype, value, tb

    def manage_block_stack(self, why):
        """ Manage a frame's block stack.
        Manipulate the block stack and data stack for looping,
        exception handling, or returning."""
        assert why != 'yield'

        block = self.frame.block_stack[-1]
        if block.type == 'loop' and why == 'continue':
            self.jump(self.return_value)
            why = None
            return why

        self.pop_block()
        self.unwind_block(block)

        if block.type == 'loop' and why == 'break':
            why = None
            self.jump(block.handler)
            return why

        if (
            why == 'exception' and
            block.type in ['setup-except', 'finally']
        ):
            self.push_block('except-handler')
            exctype, value, tb = self.last_exception
            self.push(tb, value, exctype)
            # PyErr_Normalize_Exception goes here
            self.push(tb, value, exctype)
            why = None
            self.jump(block.handler)
            return why

        elif block.type == 'finally':
            if why in ('return', 'continue'):
                self.push(self.return_value)
            self.push(why)

            why = None
            self.jump(block.handler)
            return why

        return why

    ### bytecode instructions

    ## Stack manipulation

    def byte_LOAD_CONST(self, const):
        self.push(const)

    def byte_POP_TOP(self):
        self.pop()

    def byte_DUP_TOP(self):
        self.push(self.top())

    def byte_DUP_TOPX(self, count):
        items = self.popn(count)
        for i in [1, 2]:
            self.push(*items)

    def byte_DUP_TOP_TWO(self):
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def byte_ROT_TWO(self):
        a, b = self.popn(2)
        self.push(b, a)

    def byte_ROT_THREE(self):
        a, b, c = self.popn(3)
        self.push(c, a, b)

    def byte_ROT_FOUR(self):
        a, b, c, d = self.popn(4)
        self.push(d, a, b, c)

    ## Names

    def byte_LOAD_NAME(self, name):
        frame = self.frame
        if name in frame.f_locals:
            val = frame.f_locals[name]
        elif name in frame.f_globals:
            val = frame.f_globals[name]
        elif name in frame.f_builtins:
            val = frame.f_builtins[name]
        else:
            raise NameError(f"name '{name}' is not defined")
        self.push(val)

    def byte_STORE_NAME(self, name):
        self.frame.f_locals[name] = self.pop()

    def byte_DELETE_NAME(self, name):
        del self.frame.f_locals[name]

    def byte_LOAD_FAST(self, name):
        if name in self.frame.f_locals:
            val = self.frame.f_locals[name]
        else:
            raise UnboundLocalError(
                f"local variable '{name}' referenced before assignment"
            )
        self.push(val)

    def byte_STORE_FAST(self, name):
        self.frame.f_locals[name] = self.pop()

    def byte_DELETE_FAST(self, name):
        del self.frame.f_locals[name]

    def byte_LOAD_GLOBAL(self, name):
        f = self.frame
        if name in f.f_globals:
            val = f.f_globals[name]
        elif name in f.f_builtins:
            val = f.f_builtins[name]
        else:
            raise NameError(f"global name '{name}' is not defined")
        self.push(val)

    def byte_STORE_GLOBAL(self, name):
        f = self.frame
        f.f_globals[name] = self.pop()

    def byte_LOAD_DEREF(self, name):
        self.push(self.frame.cells[name].get())

    def byte_STORE_DEREF(self, name):
        self.frame.cells[name].set(self.pop())

    def byte_LOAD_LOCALS(self):
        self.push(self.frame.f_locals)

    ## Operators

    # Unary operations take the top of the stack,
    # apply the operation, and push the result back on the stack.
    UNARY_OPERATORS = {
        'POSITIVE': operator.pos,
        'NEGATIVE': operator.neg,
        'NOT':      operator.not_,
        'INVERT':   operator.invert,
    }

    def unaryOperator(self, op):
        x = self.pop()
        self.push(self.UNARY_OPERATORS[op](x))

    def byte_GET_ITER(self):
        self.push(iter(self.pop()))

    def byte_GET_YIELD_FROM_ITER(self):
        if not (inspect.isgenerator(self.top()) or
                inspect.iscoroutine(self.top())):
            self.push(iter(self.pop()))

    # Binary operations remove
    # the top of the stack (TOS) and
    # the second top-most stack item (TOS1) from the stack.
    # They perform the operation, and put the result back on the stack.
    BINARY_OPERATORS = {
        'POWER':            pow,
        'MULTIPLY':         operator.mul,
        'MATRIX_MULTIPLY':  operator.matmul,
        'FLOOR_DIVIDE':     operator.floordiv,
        'TRUE_DIVIDE':      operator.truediv,
        'MODULO':           operator.mod,
        'ADD':              operator.add,
        'SUBTRACT':         operator.sub,
        'SUBSCR':           operator.getitem,
        'LSHIFT':           operator.lshift,
        'RSHIFT':           operator.rshift,
        'AND':              operator.and_,
        'XOR':              operator.xor,
        'OR':               operator.or_,
    }

    def binaryOperator(self, op):
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[op](x, y))

    # In-place operations are like binary operations,
    # in that they remove TOS and TOS1,
    # and push the result back on the stack,
    # but the operation is done in-place when TOS1 supports it,
    # and the resulting TOS may be (but does not have to be) the original TOS1.
    def inplaceOperator(self, op):
        x, y = self.popn(2)
        if op == 'POWER':
            x **= y
        elif op == 'MULTIPLY':
            x *= y
        elif op == 'MATRIX_MULTIPLY':
            x @= y
        elif op in 'FLOOR_DIVIDE':
            x //= y
        elif op == 'TRUE_DIVIDE':
            x /= y
        elif op == 'MODULO':
            x %= y
        elif op == 'ADD':
            x += y
        elif op == 'SUBTRACT':
            x -= y
        elif op == 'LSHIFT':
            x <<= y
        elif op == 'RSHIFT':
            x >>= y
        elif op == 'AND':
            x &= y
        elif op == 'XOR':
            x ^= y
        elif op == 'OR':
            x |= y
        else:           # pragma: no cover
            raise VirtualMachineError(f"Unknown in-place operator: {op}")
        self.push(x)

    def byte_STORE_SUBSCR(self):
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def byte_DELETE_SUBSCR(self):
        obj, subscr = self.popn(2)
        del obj[subscr]

    def sliceOperator(self, op):
        start = 0
        end = None          # we will take this to mean end
        op, count = op[:-2], int(op[-1])
        if count == 1:
            start = self.pop()
        elif count == 2:
            end = self.pop()
        elif count == 3:
            end = self.pop()
            start = self.pop()
        l = self.pop()
        if end is None:
            end = len(l)
        if op.startswith('STORE_'):
            l[start:end] = self.pop()
        elif op.startswith('DELETE_'):
            del l[start:end]
        else:
            self.push(l[start:end])

    # Performs a Boolean operation.
    # The operation name can be found in cmp_op[opname]
    def byte_COMPARE_OP(self, opnum):
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[opnum](x, y))

    # cmp_op = ('<', '<=', '==', '!=', '>', '>=', 'in', 'not in', 'is',
    #           'is not', 'exception match', 'BAD')
    COMPARE_OPERATORS = [
        operator.lt,
        operator.le,
        operator.eq,
        operator.ne,
        operator.gt,
        operator.ge,
        lambda x, y: x in y,
        lambda x, y: x not in y,
        lambda x, y: x is y,
        lambda x, y: x is not y,
        lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    ]

    ## Attributes and indexing

    def byte_LOAD_ATTR(self, attr):
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def byte_STORE_ATTR(self, name):
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def byte_DELETE_ATTR(self, name):
        obj = self.pop()
        delattr(obj, name)

    ## Building

    def byte_BUILD_TUPLE(self, count):
        elts = self.popn(count)
        self.push(tuple(elts))

    def byte_BUILD_LIST(self, count):
        elts = self.popn(count)
        self.push(elts)

    def byte_BUILD_SET(self, count):
        elts = self.popn(count)
        self.push(set(elts))

    # Pushes a new dictionary object onto the stack.
    # Pops 2 * count items
    # so that the dictionary holds count entries:
    # {..., TOS3: TOS2, TOS1: TOS}.
    def byte_BUILD_MAP(self, count):
        map_args = self.popn(2 * count)
        the_map = {map_args[i]: map_args[i + 1]
                   for i in range(0, len(map_args), 2)}
        self.push(the_map)

    def byte_STORE_MAP(self):
        the_map, val, key = self.popn(3)
        the_map[key] = val
        self.push(the_map)

    # Unpacks TOS into count individual values,
    # which are put onto the stack right-to-left.
    def byte_UNPACK_SEQUENCE(self, count):
        seq = self.pop()
        if len(seq) == count:
            seq_2_push = seq
        else:
            seq_2_push = [tuple(seq[:-(count - 1)])] + \
                         seq[-(count - 1):]
        for x in reversed(seq_2_push):
            self.push(x)

    def byte_BUILD_SLICE(self, count):
        if count == 2:
            x, y = self.popn(2)
            self.push(slice(x, y))
        elif count == 3:
            x, y, z = self.popn(3)
            self.push(slice(x, y, z))
        else:           # pragma: no cover
            raise VirtualMachineError(f"Strange BUILD_SLICE count: {count}")

    def byte_LIST_APPEND(self, count):
        val = self.pop()
        the_list = self.peek(count)
        the_list.append(val)

    def byte_SET_ADD(self, count):
        val = self.pop()
        the_set = self.peek(count)
        the_set.add(val)

    def byte_MAP_ADD(self, count):
        val, key = self.popn(2)
        the_map = self.peek(count)
        the_map[key] = val

    # The version of BUILD_MAP specialized for constant keys.
    # count values are consumed from the stack.
    # The top element on the stack contains a tuple of keys.
    def byte_BUILD_CONST_KEY_MAP(self, count):
        map_keys = self.pop()
        map_vals = self.popn(count)
        the_map = {k: v for k, v in zip(map_keys, map_vals)}
        self.push(the_map)

    # Concatenates count strings from the stack and
    # pushes the resulting string onto the stack.
    # what make f-string faster
    def byte_BUILD_STRING(self, count):
        string_arg = self.popn(count)
        self.push(''.join(string_arg))

    # Pops count iterables from the stack,
    # joins them in a single tuple, and pushes the result.
    # Implements iterable unpacking in tuple displays (*x, *y, *z).
    def byte_BUILD_TUPLE_UNPACK(self, count):
        elts = self.popn(count)
        self.push(tuple(j for i in elts for j in i))

    # This is similar to BUILD_TUPLE_UNPACK,
    # but is used for f(*x, *y, *z) call syntax.
    # The stack item at position count + 1
    # should be the corresponding callable f.
    def byte_BUILD_TUPLE_UNPACK_WITH_CALL(self, count):
        elts = self.popn(count)
        self.push(tuple(j for i in elts for j in i))

    # This is similar to BUILD_TUPLE_UNPACK,
    # but pushes a list instead of tuple.
    # Implements iterable unpacking in list displays [*x, *y, *z].
    def byte_BUILD_LIST_UNPACK(self, count):
        elts = self.popn(count)
        self.push(list(j for i in elts for j in i))

    # This is similar to BUILD_TUPLE_UNPACK,
    # but pushes a set instead of tuple.
    # Implements iterable unpacking in set displays {*x, *y, *z}.
    def byte_BUILD_SET_UNPACK(self, count):
        elts = self.popn(count)
        self.push(set(j for i in elts for j in i))

    # Pops count mappings from the stack,
    # merges them into a single dictionary, and pushes the result.
    # Implements dictionary unpacking in dictionary displays {**x, **y, **z}.
    def byte_BUILD_MAP_UNPACK(self, count):
        elts = self.popn(count)
        map_2_push = {}
        for i in elts:
            map_2_push.update(i)
        self.push(map_2_push)

    # This is similar to BUILD_MAP_UNPACK,
    # but is used for f(**x, **y, **z) call syntax.
    # The stack item at position count + 2
    # should be the corresponding callable f.
    def byte_BUILD_MAP_UNPACK_WITH_CALL(self, count):
        elts = self.popn(count)
        map_2_push = {}
        for i in elts:
            map_2_push.update(i)
        self.push(map_2_push)

    ## Printing

    # Only used in the interactive interpreter, not in modules.
    def byte_PRINT_EXPR(self):
        print(self.pop())

    def byte_PRINT_ITEM(self):
        item = self.pop()
        self.print_item(item)

    def byte_PRINT_ITEM_TO(self):
        to = self.pop()
        item = self.pop()
        self.print_item(item, to)

    def byte_PRINT_NEWLINE(self):
        self.print_newline()

    def byte_PRINT_NEWLINE_TO(self):
        to = self.pop()
        self.print_newline(to)

    def print_item(self, item, to=None):
        if to is None:
            to = sys.stdout
        if to.softspace:
            print(" ", end="", file=to)
            to.softspace = 0
        print(item, end="", file=to)
        if isinstance(item, str):
            if (not item) or (not item[-1].isspace()) or (item[-1] == " "):
                to.softspace = 1
        else:
            to.softspace = 1

    def print_newline(self, to=None):
        if to is None:
            to = sys.stdout
        print("", file=to)
        to.softspace = 0

    ## Jumps

    def byte_JUMP_FORWARD(self, jump):
        self.jump(jump)

    def byte_JUMP_ABSOLUTE(self, jump):
        self.jump(jump)

    # Not in py2.7
    def byte_JUMP_IF_TRUE(self, jump):
        val = self.top()
        if val:
            self.jump(jump)

    def byte_JUMP_IF_FALSE(self, jump):
        val = self.top()
        if not val:
            self.jump(jump)

    def byte_POP_JUMP_IF_TRUE(self, jump):
        val = self.pop()
        if val:
            self.jump(jump)

    def byte_POP_JUMP_IF_FALSE(self, jump):
        val = self.pop()
        if not val:
            self.jump(jump)

    def byte_JUMP_IF_TRUE_OR_POP(self, jump):
        val = self.top()
        if val:
            self.jump(jump)
        else:
            self.pop()

    def byte_JUMP_IF_FALSE_OR_POP(self, jump):
        val = self.top()
        if not val:
            self.jump(jump)
        else:
            self.pop()

    ## Blocks

    def byte_SETUP_LOOP(self, dest):
        self.push_block('loop', dest)

    def byte_FOR_ITER(self, jump):
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(jump)

    def byte_BREAK_LOOP(self):
        return 'break'

    def byte_CONTINUE_LOOP(self, dest):
        # This is a trick with the return value.
        # While unrolling blocks, continue and return both have to preserve
        # state as the finally blocks are executed.  For continue, it's
        # where to jump to, for return, it's the value to return.  It gets
        # pushed on the stack for both, so continue puts the jump destination
        # into return_value.
        self.return_value = dest
        return 'continue'

    def byte_SETUP_EXCEPT(self, dest):
        self.push_block('setup-except', dest)

    def byte_SETUP_FINALLY(self, dest):
        self.push_block('finally', dest)

    def byte_END_FINALLY(self):
        v = self.pop()
        if isinstance(v, str):
            why = v
            if why in ('return', 'continue'):
                self.return_value = self.pop()
            if why == 'silenced':       # PY3
                block = self.pop_block()
                assert block.type == 'except-handler'
                self.unwind_block(block)
                why = None
        elif v is None:
            why = None
        elif issubclass(v, BaseException):
            exctype = v
            val = self.pop()
            tb = self.pop()
            self.last_exception = (exctype, val, tb)
            why = 'reraise'
        else:       # pragma: no cover
            raise VirtualMachineError("Confused END_FINALLY")
        return why

    def byte_POP_BLOCK(self):
        self.pop_block()

        def byte_RAISE_VARARGS(self, argc):
            cause = exc = None
            if argc == 2:
                cause = self.pop()
                exc = self.pop()
            elif argc == 1:
                exc = self.pop()
            return self.do_raise(exc, cause)

    def do_raise(self, exc, cause):
        if exc is None:         # reraise
            exc_type, val, tb = self.last_exception
            if exc_type is None:
                return 'exception'      # error
            else:
                return 'reraise'

        elif type(exc) == type:
            # As in `raise ValueError`
            exc_type = exc
            val = exc()             # Make an instance.
        elif isinstance(exc, BaseException):
            # As in `raise ValueError('foo')`
            exc_type = type(exc)
            val = exc
        else:
            return 'exception'      # error

        # If you reach this point, you're guaranteed that
        # val is a valid exception instance and exc_type is its class.
        # Now do a similar thing for the cause, if present.
        if cause:
            if type(cause) == type:
                cause = cause()
            elif not isinstance(cause, BaseException):
                return 'exception'  # error

            val.__cause__ = cause

        self.last_exception = exc_type, val, val.__traceback__
        return 'exception'

    def byte_POP_EXCEPT(self):
        block = self.pop_block()
        if block.type != 'except-handler':
            raise Exception("popped block is not an except handler")
        self.unwind_block(block)

    def byte_SETUP_WITH(self, dest):
        ctxmgr = self.pop()
        self.push(ctxmgr.__exit__)
        ctxmgr_obj = ctxmgr.__enter__()
        self.push_block('finally', dest)
        self.push(ctxmgr_obj)

    def byte_WITH_CLEANUP(self):
        # The code here does some weird stack manipulation: the exit function
        # is buried in the stack, and where depends on what's on top of it.
        # Pull out the exit function, and leave the rest in place.
        v = w = None
        u = self.top()
        if u is None:
            exit_func = self.pop(1)
        elif isinstance(u, str):
            if u in ('return', 'continue'):
                exit_func = self.pop(2)
            else:
                exit_func = self.pop(1)
            u = None
        elif issubclass(u, BaseException):
            w, v, u = self.popn(3)
            tp, exc, tb = self.popn(3)
            exit_func = self.pop()
            self.push(tp, exc, tb)
            self.push(None)
            self.push(w, v, u)
            block = self.pop_block()
            assert block.type == 'except-handler'
            self.push_block(block.type, block.handler, block.level-1)
        else:       # pragma: no cover
            raise VirtualMachineError("Confused WITH_CLEANUP")
        exit_ret = exit_func(u, v, w)
        err = (u is not None) and bool(exit_ret)
        if err:
            # An error occurred, and was suppressed
            self.push('silenced')

    ## Functions

    # Pushes a new function object on the stack.
    # From bottom to top,
    # the consumed stack must consist of values
    # if the argument carries a specified flag value

    # 0x01 a tuple of default values for positional-only
    # and positional-or-keyword parameters in positional order
    # 0x02 a dictionary of keyword-only parameters’ default values
    # 0x04 an annotation dictionary
    # 0x08 a tuple containing cells for free variables, making a closure
    # the code associated with the function (at TOS1)
    # the qualified name of the function (at TOS)
    def byte_MAKE_FUNCTION(self, argc):
        name = self.pop()
        code = self.pop()
        globs = self.frame.f_globals
        closure = self.pop() if (argc & 0x8) else None
        ann = self.pop() if (argc & 0x4) else None
        kwdefaults = self.pop() if (argc & 0x02) else None
        defaults = self.pop() if (argc & 0x01) else None
        fn = Function(name, code, globs, defaults, kwdefaults, closure, self)
        self.push(fn)

    def byte_LOAD_CLOSURE(self, name):
        self.push(self.frame.cells[name])

    def byte_CALL_FUNCTION(self, argc):
        arg = self.popn(argc)
        return self.call_function(arg, [], {})

    def byte_CALL_FUNCTION_KW(self, argc):
        kwargs_keys = self.pop()
        kwargs_values = self.popn(len(kwargs_keys))
        kwargs = {kwargs_key: kwargs_val
                  for kwargs_key, kwargs_val in
                  zip(kwargs_keys, kwargs_values)}
        arg = self.pop(argc - len(kwargs_keys))
        return self.call_function(arg, [], kwargs)

    # 待修改，Python3中未绑定方法改为普通函数
    def call_function(self, arg, args, kwargs):
        posargs, namedargs = arg + args, kwargs

        func = self.pop()
        frame = self.frame
        # 属性
        if hasattr(func, 'im_func'):
            # Methods get self as an implicit first parameter.
            if func.im_self:
                posargs.insert(0, func.im_self)
            func = func.im_func
        retval = func(*posargs, **namedargs)
        self.push(retval)

    def byte_RETURN_VALUE(self):
        self.return_value = self.pop()
        if self.frame.generator:
            self.frame.generator.finished = True
        return "return"

    def byte_YIELD_VALUE(self):
        self.return_value = self.pop()
        return "yield"

    def byte_YIELD_FROM(self):
        u = self.pop()
        x = self.top()

        try:
            if not isinstance(x, Generator) or u is None:
                # Call next on iterators.
                retval = next(x)
            else:
                retval = x.send(u)
            self.return_value = retval
        except StopIteration as e:
            self.pop()
            self.push(e.value)
        else:
            # YIELD_FROM decrements f_lasti, so that it will be called
            # repeatedly until a StopIteration is raised.
            self.jump(self.frame.f_lasti - 1)
            # Returning "yield" prevents the block stack cleanup code
            # from executing, suspending the frame in its current state.
            return "yield"

    ## Importing

    def byte_IMPORT_NAME(self, name):
        level, fromlist = self.popn(2)
        frame = self.frame
        self.push(
            __import__(name, frame.f_globals, frame.f_locals, fromlist, level)
        )

    def byte_IMPORT_STAR(self):
        # TODO: this doesn't use __all__ properly.
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != '_':
                self.frame.f_locals[attr] = getattr(mod, attr)

    def byte_IMPORT_FROM(self, name):
        mod = self.top()
        self.push(getattr(mod, name))

    # Coroutine opcodes

    # Implements TOS = get_awaitable(TOS),
    # where get_awaitable(o) returns o
    # if o is a coroutine object or
    # a generator object with the CO_ITERABLE_COROUTINE flag,
    # or resolves o.__await__.
    def byte_GET_AWAITABLE(self):
        pass

    # Implements TOS = get_awaitable(TOS.__aiter__())
    def byte_GET_AITER(self):
        pass

    # Implements PUSH(get_awaitable(TOS.__anext__()))
    def byte_GET_ANEXT(self):
        pass

    # Resolves __aenter__ and __aexit__
    # from the object on top of the stack.
    # Pushes __aexit__ and result of __aenter__() to the stack.
    def byte_BEFORE_ASYNC_WITH(self):
        pass

    # Creates a new frame object.
    def byte_SETUP_ASYNC_WITH(self):
        pass


    ## And the rest...
    # Prefixes any opcode which has an argument too big
    # to fit into the default two bytes.
    # ext holds two additional bytes which,
    # taken together with the subsequent opcode’s argument,
    # comprise a four-byte argument,
    # ext being the two most-significant bytes.
    def byte_EXTENDED_ARG(self, ext):
        self.EXTENDED_ARG_ext = ext

    # Raises an exception.
    # argc indicates the number of arguments to the raise statement,
    # ranging from 0 to 3.
    # The handler will find the traceback as TOS2,
    # the parameter as TOS1, and the exception as TOS.
    def byte_RAISE_VARARGS(self, argc):
        pass

    def byte_EXEC_STMT(self):
        stmt, globs, locs = self.popn(3)
        six.exec_(stmt, globs, locs)

    def byte_LOAD_BUILD_CLASS(self):
        # New in py3
        self.push(build_class)

    def byte_STORE_LOCALS(self):
        self.frame.f_locals = self.pop()

    # Not in py2.7
    def byte_SET_LINENO(self, lineno):
        self.frame.f_lineno = lineno

    # Used for implementing formatted literal strings (f-strings).
    # Pops an optional fmt_spec from the stack,
    # then a required value.
    # Formatting is performed using PyObject_Format().
    # The result is pushed on the stack.

    # flags is interpreted as follows:
    # (flags & 0x03) == 0x00: value is formatted as-is.
    # (flags & 0x03) == 0x01: call str() on value before formatting it.
    # (flags & 0x03) == 0x02: call repr() on value before formatting it.
    # (flags & 0x03) == 0x03: call ascii() on value before formatting it.
    # (flags & 0x04) == 0x04: pop fmt_spec from the stack and use it, else use an empty fmt_spec.
    def byte_FORMAT_VALUE(self, flags):
        str_2_push = ''
        if (flags & 0x03) == 0x00:
            str_2_push = self.pop()
        elif (flags & 0x03) == 0x01:
            str_2_push = str(self.pop())
        elif (flags & 0x03) == 0x02:
            str_2_push = repr(self.pop())
        elif (flags & 0x03) == 0x03:
            str_2_push = ascii(self.pop())
        elif (flags & 0x04) == 0x04:
            fmt_spec = self.pop()
            str_2_push = fmt_spec.format(self.pop())
        self.push(str_2_push)


def build_class(func, name, *bases, **kwds):
    "Like __build_class__ in bltinmodule.c, but running in the byterun VM."
    if not isinstance(func, Function):
        raise TypeError("func must be a function")
    if not isinstance(name, str):
        raise TypeError("name is not a string")
    metaclass = type(bases[0]) if bases else type
    if isinstance(metaclass, type):
        metaclass = calculate_metaclass(metaclass, bases)

    try:
        prepare = metaclass.__prepare__
    except AttributeError:
        namespace = {}
    else:
        namespace = prepare(name, bases, **kwds)

    # Execute the body of func. This is the step that would go wrong if
    # we tried to use the built-in __build_class__, because __build_class__
    # does not call func, it magically executes its body directly, as we
    # do here (except we invoke our VirtualMachine instead of CPython's).
    frame = func._vm.make_frame(func.func_code,
                                f_globals=func.func_globals,
                                f_locals=namespace,
                                f_closure=func.func_closure)
    cell = func._vm.run_frame(frame)

    cls = metaclass(name, bases, namespace)
    if isinstance(cell, Cell):
        cell.set(cls)
    return cls


def calculate_metaclass(metaclass, bases):
    "Determine the most derived metatype."
    winner = metaclass
    for base in bases:
        t = type(base)
        if issubclass(t, winner):
            winner = t
        elif not issubclass(winner, t):
            raise TypeError("metaclass conflict", winner, t)
    return winner
