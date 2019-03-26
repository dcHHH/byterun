import inspect
import operator

from byterun.pyobj import Block, Function, Generator, Cell


class VirtualMachineError(Exception):
    """For raising errors in the operation of the VM."""
    pass


class VirtualMachine_instruction:
    def __init__(self):
        self.frames = []                # The call stack of frames.
        self.frame = None               # The current frame.
        self.return_value = None        # 在frame中传递的返回值
        self.last_exception = None      # 上一个异常状态; (type, value, traceback)

        self.EXTENDED_ARG_ext = 0       # EXTENDED_ARG指令使用

    def top(self):
        """
        Return the value at the top of the stack, with no changes.
        """
        return self.frame.stack[-1]

    def pop(self, i=0):
        """
        Pop a value from the stack.
        Default to the top of the stack,
        but `i` can be a count from the top instead.
        """
        return self.frame.stack.pop(-1-i)

    def push(self, *vals):
        """
        Push values onto the value stack.
        """
        self.frame.stack.extend(vals)

    def popn(self, n):
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n:
            ret = self.frame.stack[-n:]
            self.frame.stack[-n:] = []
            return ret
        else:
            return []

    def peek(self, n):
        """
        Get a value n entries down in the stack, without changing the stack.
        """
        return self.frame.stack[-n]

    def jump(self, jump):
        """
        Move the bytecode pointer to jump, so it will execute next.
        """
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
        """
        Manage a frame's block stack.
        Manipulate the block stack and data stack for looping,
        exception handling, or returning.

        why:
        continue;   break;  excption;   return
        yield;  silenced;   reraise
        """
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
        """
        Pushes co_consts[consti] onto the stack
        """
        self.push(const)

    def byte_POP_TOP(self):
        """
        Removes the top-of-stack (TOS) item.
        """
        self.pop()

    def byte_DUP_TOP(self):
        """
        Duplicates the reference on top of the stack
        """
        self.push(self.top())

    def byte_DUP_TOP_TWO(self):
        """
        Duplicates the two references on top of the stack,
        leaving them in the same order
        """
        a, b = self.popn(2)
        self.push(a, b, a, b)

    def byte_ROT_TWO(self):
        """
        Swaps the two top-most stack items
        """
        a, b = self.popn(2)
        self.push(b, a)

    def byte_ROT_THREE(self):
        """
        Lifts second and third stack item one position up,
        moves top down to position three
        """
        a, b, c = self.popn(3)
        self.push(c, a, b)

    ## Names

    def byte_LOAD_NAME(self, name):
        """
        Pushes the value associated with co_names[namei] onto the stack
        """
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
        """
        Implements name = TOS.
        namei is the index of name in the attribute co_names of the code object.
        The compiler tries to use STORE_FAST or STORE_GLOBAL if possible.
        """
        self.frame.f_locals[name] = self.pop()

    def byte_DELETE_NAME(self, name):
        """
        Implements del name, where namei is the index
        into co_names attribute of the code object.
        """
        del self.frame.f_locals[name]

    def byte_LOAD_FAST(self, name):
        """
        Pushes a reference to the local co_varnames[var_num] onto the stack.
        """
        if name in self.frame.f_locals:
            val = self.frame.f_locals[name]
        else:
            raise UnboundLocalError(
                f"local variable '{name}' referenced before assignment"
            )
        self.push(val)

    def byte_STORE_FAST(self, name):
        """
        Stores TOS into the local co_varnames[var_num]
        """
        self.frame.f_locals[name] = self.pop()

    def byte_DELETE_FAST(self, name):
        """
        Deletes local co_varnames[var_num]
        """
        del self.frame.f_locals[name]

    def byte_LOAD_GLOBAL(self, name):
        """
        Loads the global named co_names[namei] onto the stack
        """
        f = self.frame
        if name in f.f_globals:
            val = f.f_globals[name]
        elif name in f.f_builtins:
            val = f.f_builtins[name]
        else:
            raise NameError(f"name '{name}' is not defined")
        self.push(val)

    def byte_STORE_GLOBAL(self, name):
        """
        Works as STORE_NAME, but stores the name as a global
        """
        f = self.frame
        f.f_globals[name] = self.pop()

    def byte_DELETE_GLOBAL(self, name):
        """
        Works as DELETE_NAME, but deletes a global name
        """
        del self.frame.f_globals[name]

    def byte_LOAD_DEREF(self, name):
        """
        Loads the cell contained in slot i of the cell and free variable storage.
        Pushes a reference to the object the cell contains on the stack
        """
        self.push(self.frame.cells[name].get())

    def byte_STORE_DEREF(self, name):
        """
        Stores TOS into the cell contained in slot i of the cell
        and free variable storage
        """
        self.frame.cells[name].set(self.pop())

    ## Operators

    UNARY_OPERATORS = {
        'POSITIVE': operator.pos,
        'NEGATIVE': operator.neg,
        'NOT':      operator.not_,
        'INVERT':   operator.invert,
    }

    def unaryOperator(self, op):
        """
        Unary operations take the top of the stack,
        apply the operation, and push the result back on the stack.
        """
        x = self.pop()
        self.push(self.UNARY_OPERATORS[op](x))

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
        """
        Binary operations remove
        the top of the stack (TOS) and
        the second top-most stack item (TOS1) from the stack.
        They perform the operation, and put the result back on the stack.
        """
        x, y = self.popn(2)
        self.push(self.BINARY_OPERATORS[op](x, y))

    def inplaceOperator(self, op):
        """
        In-place operations are like binary operations,
        in that they remove TOS and TOS1,
        and push the result back on the stack,
        but the operation is done in-place when TOS1 supports it,
        and the resulting TOS may be (but does not have to be) the original TOS1.
        """
        x, y = self.popn(2)
        if op == 'POWER':
            x **= y
        elif op == 'MULTIPLY':
            x *= y
        elif op == 'MATRIX_MULTIPLY':
            x @= y
        elif op == 'FLOOR_DIVIDE':
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
        else:
            raise VirtualMachineError(f"Unknown in-place operator: {op}")
        self.push(x)

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

    def byte_COMPARE_OP(self, opnum):
        """
        Performs a Boolean operation.
        The operation name can be found in cmp_op[opname]
        """
        x, y = self.popn(2)
        self.push(self.COMPARE_OPERATORS[opnum](x, y))

    def byte_GET_ITER(self):
        """
        Implements TOS = iter(TOS)
        """
        self.push(iter(self.pop()))

    def byte_GET_YIELD_FROM_ITER(self):
        """
        If TOS is a generator iterator or coroutine object it is left as is.
        Otherwise, implements TOS = iter(TOS)
        """
        if not (inspect.isgenerator(self.top()) or
                inspect.iscoroutine(self.top())):
            self.push(iter(self.pop()))

    def byte_STORE_SUBSCR(self):
        """
        Implements TOS1[TOS] = TOS2
        """
        val, obj, subscr = self.popn(3)
        obj[subscr] = val

    def byte_DELETE_SUBSCR(self):
        """
        Implements del TOS1[TOS]
        """
        obj, subscr = self.popn(2)
        del obj[subscr]

    ## Attributes and indexing

    def byte_LOAD_ATTR(self, attr):
        """
        Replaces TOS with getattr(TOS, co_names[namei]).
        """
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def byte_STORE_ATTR(self, name):
        """
        Implements TOS.name = TOS1,
        where namei is the index of name in co_names
        """
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def byte_DELETE_ATTR(self, name):
        """
        Implements del TOS.name, using namei as index into co_names
        """
        obj = self.pop()
        delattr(obj, name)

    ## Building

    def byte_BUILD_TUPLE(self, count):
        """
        Creates a tuple consuming count items from the stack,
        and pushes the resulting tuple onto the stack
        """
        elts = self.popn(count)
        self.push(tuple(elts))

    def byte_BUILD_LIST(self, count):
        """
        Works as BUILD_TUPLE, but creates a list
        """
        elts = self.popn(count)
        self.push(elts)

    def byte_BUILD_SET(self, count):
        """
        Works as BUILD_TUPLE, but creates a set
        """
        elts = self.popn(count)
        self.push(set(elts))

    def byte_BUILD_MAP(self, count):
        """
        Pushes a new dictionary object onto the stack.
        Pops 2 * count items
        so that the dictionary holds count entries:
        {..., TOS3: TOS2, TOS1: TOS}
        """
        map_args = self.popn(2 * count)
        self.push(dict(zip(map_args[::2], map_args[1::2])))

    def byte_BUILD_SLICE(self, count):
        """
        Pushes a slice object on the stack.
        argc must be 2 or 3. If it is 2, slice(TOS1, TOS) is pushed;
        if it is 3, slice(TOS2, TOS1, TOS) is pushed
        """
        if count == 2:
            start, end = self.popn(2)
            self.push(slice(start, end))
        elif count == 3:
            start, end, step = self.popn(3)
            self.push(slice(start, end, step))
        else:
            raise VirtualMachineError(f"Strange BUILD_SLICE count: {count}")

    def byte_BUILD_CONST_KEY_MAP(self, count):
        """
        The version of BUILD_MAP specialized for constant keys.
        count values are consumed from the stack.
        The top element on the stack contains a tuple of keys.
        :param count:
        :return:
        """
        map_keys = self.pop()
        map_vals = self.popn(count)
        the_map = dict(zip(map_keys, map_vals))
        self.push(the_map)

    def byte_BUILD_STRING(self, count):
        """
        Concatenates count strings from the stack and
        pushes the resulting string onto the stack.
        what make f-string faster
        """
        string_arg = self.popn(count)
        self.push(''.join(string_arg))

    def byte_BUILD_TUPLE_UNPACK(self, count):
        """
        Pops count iterables from the stack,
        joins them in a single tuple, and pushes the result.
        Implements iterable unpacking in tuple displays (*x, *y, *z).
        """
        elts = self.popn(count)
        self.push(tuple(j for i in elts for j in i))

    def byte_BUILD_TUPLE_UNPACK_WITH_CALL(self, count):
        """
        This is similar to BUILD_TUPLE_UNPACK,
        but is used for f(*x, *y, *z) call syntax.
        The stack item at position count + 1
        should be the corresponding callable f
        """
        assert callable(self.peek(count + 1))
        elts = self.popn(count)
        self.push(tuple(j for i in elts for j in i))

    def byte_BUILD_LIST_UNPACK(self, count):
        """
        This is similar to BUILD_TUPLE_UNPACK,
        but pushes a list instead of tuple.
        Implements iterable unpacking in list displays [*x, *y, *z]
        """
        elts = self.popn(count)
        self.push(list(j for i in elts for j in i))

    def byte_BUILD_SET_UNPACK(self, count):
        """
        This is similar to BUILD_TUPLE_UNPACK,
        but pushes a set instead of tuple.
        Implements iterable unpacking in set displays {*x, *y, *z}
        """
        elts = self.popn(count)
        self.push(set(j for i in elts for j in i))

    def byte_BUILD_MAP_UNPACK(self, count):
        """
        Pops count mappings from the stack,
        merges them into a single dictionary, and pushes the result.
        Implements dictionary unpacking in dictionary displays {**x, **y, **z}
        """
        elts = self.popn(count)
        map_2_push = {}
        for i in elts:
            map_2_push.update(i)
        self.push(map_2_push)

    def byte_BUILD_MAP_UNPACK_WITH_CALL(self, count):
        """
        This is similar to BUILD_MAP_UNPACK,
        but is used for f(**x, **y, **z) call syntax.
        The stack item at position count + 2
        should be the corresponding callable f
        """
        assert callable(self.peek(count + 2))
        elts = self.popn(count)
        map_2_push = {}
        for i in elts:
            map_2_push.update(i)
        self.push(map_2_push)

    def byte_UNPACK_SEQUENCE(self, count):
        """
        Unpacks TOS into count individual values,
        which are put onto the stack right-to-left
        """
        seq = self.pop()
        if len(seq) == count:
            seq_2_push = seq
        else:
            seq_2_push = [seq[:-(count - 1)]] + \
                         seq[-(count - 1):]
        for x in reversed(seq_2_push):
            self.push(x)

    # For all of the SET_ADD, LIST_APPEND and MAP_ADD instructions,
    # while the added value or key/value pair is popped off,
    # the container object remains on the stack
    # so that it is available for further iterations of the loop.
    def byte_LIST_APPEND(self, count):
        """
        Calls list.append(TOS[-i], TOS).
        Used to implement list comprehensions
        """
        val = self.pop()
        the_list = self.peek(count)
        the_list.append(val)

    def byte_SET_ADD(self, count):
        """
        Calls set.add(TOS1[-i], TOS).
        Used to implement set comprehensions
        """
        val = self.pop()
        the_set = self.peek(count)
        the_set.add(val)

    def byte_MAP_ADD(self, count):
        """
        Calls dict.setitem(TOS1[-i], TOS, TOS1).
        Used to implement dict comprehensions
        """
        val, key = self.popn(2)
        the_map = self.peek(count)
        the_map[key] = val

    ## Jumps

    def byte_JUMP_FORWARD(self, jump):
        """
        Increments bytecode counter by delta
        """
        self.jump(jump)

    def byte_JUMP_ABSOLUTE(self, jump):
        """
        Set bytecode counter to target.
        """
        self.jump(jump)

    def byte_POP_JUMP_IF_TRUE(self, jump):
        """
        If TOS is true, sets the bytecode counter to target.
        TOS is popped
        """
        val = self.pop()
        if val:
            self.jump(jump)

    def byte_POP_JUMP_IF_FALSE(self, jump):
        """
        If TOS is false, sets the bytecode counter to target.
        TOS is popped
        """
        val = self.pop()
        if not val:
            self.jump(jump)

    def byte_JUMP_IF_TRUE_OR_POP(self, jump):
        """
        If TOS is true,
        sets the bytecode counter to target and leaves TOS on the stack.
        Otherwise (TOS is false), TOS is popped
        """
        val = self.top()
        if val:
            self.jump(jump)
        else:
            self.pop()

    def byte_JUMP_IF_FALSE_OR_POP(self, jump):
        """
        If TOS is false,
        sets the bytecode counter to target and leaves TOS on the stack.
        Otherwise (TOS is true), TOS is popped
        :param jump:
        :return:
        """
        val = self.top()
        if not val:
            self.jump(jump)
        else:
            self.pop()

    ## Blocks

    def byte_SETUP_LOOP(self, dest):
        """
        Pushes a block for a loop onto the block stack.
        The block spans from the current instruction with a size of delta bytes
        """
        self.push_block('loop', dest)

    def byte_FOR_ITER(self, jump):
        """
        TOS is an iterator. Call its __next__() method.
        If this yields a new value,
        push it on the stack (leaving the iterator below it).
        If the iterator indicates it is exhausted TOS is popped,
        and the byte code counter is incremented by delta
        """
        iterobj = self.top()
        try:
            v = next(iterobj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(jump)

    def byte_BREAK_LOOP(self):
        """
        Terminates a loop due to a break statement
        """
        return 'break'

    def byte_CONTINUE_LOOP(self, dest):
        """
        Continues a loop due to a continue statement.
        target is the address to jump to (which should be a FOR_ITER instruction)

        This is a trick with the return value.
        While unrolling blocks, continue and return both have to preserve
        state as the finally blocks are executed.  For continue, it's
        where to jump to, for return, it's the value to return.  It gets
        pushed on the stack for both, so continue puts the jump destination
        into return_value.
        """
        self.return_value = dest
        return 'continue'

    def byte_SETUP_EXCEPT(self, dest):
        """
        Pushes a try block from a try-except clause onto the block stack.
        delta points to the first except block.
        """
        self.push_block('setup-except', dest)

    def byte_SETUP_FINALLY(self, dest):
        """
        Pushes a try block from a try-except clause onto the block stack.
        delta points to the finally block
        """
        self.push_block('finally', dest)

    def byte_END_FINALLY(self):
        """
        Terminates a finally clause.
        The interpreter recalls whether the exception has to be re-raised,
        or whether the function returns, and continues with the outer-next block
        """
        v = self.pop()
        if isinstance(v, str):
            why = v
            if why in ('return', 'continue'):
                self.return_value = self.pop()
            if why == 'silenced':
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
        else:
            raise VirtualMachineError("Confused END_FINALLY")
        return why

    def byte_POP_BLOCK(self):
        """
        Removes one block from the block stack.
        Per frame, there is a stack of blocks,
        denoting nested loops, try statements, and such
        """
        self.pop_block()

    def byte_RAISE_VARARGS(self, argc):
        """
        Raises an exception.
        argc indicates the number of arguments to the raise statement,
        ranging from 0 to 3.
        The handler will find the traceback as TOS2,
        the parameter as TOS1, and the exception as TOS.
        """
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
        """
        Removes one block from the block stack.
        The popped block must be an exception handler block,
        as implicitly created when entering an except handler.
        In addition to popping extraneous values from the frame stack,
        the last three popped values are used to restore the exception state
        """
        block = self.pop_block()
        if block.type != 'except-handler':
            raise Exception("popped block is not an except handler")
        self.unwind_block(block)

    ## with statement

    def byte_SETUP_WITH(self, dest):
        """
        This opcode performs several operations before a with block starts.

        First      it loads __exit__() from the context manager and
                    pushes it onto the stack for later use by WITH_CLEANUP.
        Then       __enter__() is called, and a finally block pointing to delta is pushed.
        Finally    the result of calling the enter method is pushed onto the stack.

        The next opcode will either ignore it (POP_TOP), or store it in (a) variable(s)
        (STORE_FAST, STORE_NAME, or UNPACK_SEQUENCE)
        """
        ctxmgr = self.pop()
        self.push(ctxmgr.__exit__)
        ctxmgr_obj = ctxmgr.__enter__()
        self.push_block('finally', dest)
        self.push(ctxmgr_obj)

    def byte_WITH_CLEANUP_START(self):
        """(官方文档错误，摘抄至源码)
        At the top of the stack are 1-6 values indicating
       how/why we entered the finally clause:

       1. TOP = None
       2. (TOP, SECOND) = (WHY_{RETURN,CONTINUE}), retval
       3. TOP = WHY_*; no retval below it
       4. (TOP, SECOND, THIRD) = exc_info()
          (FOURTH, FITH, SIXTH) = previous exception for EXCEPT_HANDLER

       Below them is EXIT, the context.__exit__ bound method.

       In the first three cases, call EXIT(None, None, None),
       remove EXIT from the stack, leaving the rest in the same order.

       In the fourth case, call EXIT(TOP, SECOND, THIRD)
       we shift the bottom 3 values of the stack down,
       and replace the empty spot with NULL.

       In addition, if the stack represents an exception,
       *and* the function call returns a 'true' value, we
       push WHY_SILENCED onto the stack.  END_FINALLY will
       then not re-raise the exception
        """
        tos = self.top()
        top, second, third = None, None, None
        if tos is None:
            exit_method = self.pop(1)
        elif isinstance(tos, str):
            if tos in {'return', 'continue'}:
                exit_method = self.pop(2)
            else:
                exit_method = self.pop(1)
        elif issubclass(tos, BaseException):
            third, second, top = self.popn(3)
            sixth, fifth, fourth = self.popn(3)
            exit_method = self.pop()
            self.push(third, second, top)
            self.push(None)
            self.push(sixth, fifth, fourth)
            block = self.pop_block()
            assert block.type == 'except-handler'
            self.push_block(block.type, block.handler, block.level-1)
        else:
            raise Exception("WITH_CLEANUP_START error")

        res = exit_method(top, second, third)
        self.push(tos)
        self.push(res)

    def byte_WITH_CLEANUP_FINISH(self):
        """
        Pops exception type and result of ‘exit’ function call from the stack.
        If the stack represents an exception,
        and the function call returns a ‘true’ value,
        this information is “zapped” and replaced with a single
        WHY_SILENCED to prevent END_FINALLY from re-raising the exception.
        (But non-local gotos will still be resumed.)
        """
        res = self.pop()
        u = self.pop()
        if type(u) is type and issubclass(u, BaseException) and res:
            self.push("silenced")

    ## Functions

    def byte_MAKE_FUNCTION(self, argc):
        """
        Pushes a new function object on the stack.
        From bottom to top, the consumed stack must consist of values
        if the argument carries a specified flag value

        0x01    a tuple of default values for positional-only
                and positional-or-keyword parameters in positional order
        0x02    a dictionary of keyword-only parameters’ default values
        0x04    an annotation dictionary
        0x08    a tuple containing cells for free variables, making a closure

        the code associated with the function (at TOS1)
        the qualified name of the function (at TOS)
        """
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
        """
        Pushes a reference to the cell
        contained in slot i of the cell and free variable storage.
        The name of the variable is co_cellvars[i]
        if i is less than the length of co_cellvars.
        Otherwise it is co_freevars[i - len(co_cellvars)]
        """
        self.push(self.frame.cells[name])

    def byte_CALL_FUNCTION(self, argc):
        """
        Calls a callable object with positional arguments.
        argc indicates the number of positional arguments.
        The top of the stack contains positional arguments,
        with the right-most argument on top.
        Below the arguments is a callable object to call.
        CALL_FUNCTION pops all arguments and the callable object off the stack,
        calls the callable object with those arguments,
        and pushes the return value returned by the callable object
        """
        arg = self.popn(argc)
        return self.call_function(arg, [], {})

    def byte_CALL_FUNCTION_KW(self, argc):
        """
        Calls a callable object with positional (if any) and keyword arguments.
        argc indicates the total number of positional and keyword arguments.
        The top element on the stack contains a tuple of keyword argument names.
        Below that are keyword arguments in the order corresponding to the tuple.
        Below that are positional arguments, with the right-most parameter on top.
        Below the arguments is a callable object to call.
        CALL_FUNCTION_KW pops all arguments and the callable object off the stack,
        calls the callable object with those arguments,
        and pushes the return value returned by the callable object
        """
        kwargs_keys = self.pop()
        kwargs_values = self.popn(len(kwargs_keys))
        kwargs = dict(zip(kwargs_keys, kwargs_values))
        arg = self.popn(argc - len(kwargs_keys))
        return self.call_function(arg, [], kwargs)

    def byte_CALL_FUNCTION_EX(self, flags):
        """
        Calls a callable object with variable set of positional and keyword arguments.
        If the lowest bit of flags is set,
        the top of the stack contains a mapping object containing additional keyword arguments.
        Below that is an iterable object containing positional arguments
        and a callable object to call.
        BUILD_MAP_UNPACK_WITH_CALL and BUILD_TUPLE_UNPACK_WITH_CALL can be used for
        merging multiple mapping objects and iterables containing arguments.
        Before the callable is called,
        the mapping object and iterable object are each “unpacked” and
        their contents passed in as keyword and positional arguments respectively.
        CALL_FUNCTION_EX pops all arguments and the callable object off the stack,
        calls the callable object with those arguments,
        and pushes the return value returned by the callable object
        """
        kwargs = self.pop() if (flags & 0x01) else {}
        arg = list(self.pop())
        return self.call_function(arg, [], kwargs)

    def call_function(self, arg, args, kwargs):
        posargs, namedargs = arg + args, kwargs

        func = self.pop()
        # 属性（类方法）
        if hasattr(func, 'im_func'):
            # Methods get self as an implicit first parameter.
            if func.im_self:
                posargs.insert(0, func.im_self)
            func = func.im_func
        retval = func(*posargs, **namedargs)
        self.push(retval)

    def byte_RETURN_VALUE(self):
        """
        Returns with TOS to the caller of the function
        """
        self.return_value = self.pop()
        if self.frame.generator:
            self.frame.generator.finished = True
        return "return"

    def byte_YIELD_VALUE(self):
        """
        Pops TOS and yields it from a generator
        """
        self.return_value = self.pop()
        return "yield"

    def byte_YIELD_FROM(self):
        """
        Pops TOS and delegates to it as a subiterator from a generator
        """
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
            self.jump(self.frame.f_lasti - 2)
            # Returning "yield" prevents the block stack cleanup code
            # from executing, suspending the frame in its current state.
            return "yield"

    ## Importing

    def byte_IMPORT_NAME(self, name):
        """
        Imports the module co_names[namei].
        TOS and TOS1 are popped and provide the fromlist
        and level arguments of __import__().
        The module object is pushed onto the stack.
        The current namespace is not affected:
        for a proper import statement,
        a subsequent STORE_FAST instruction modifies the namespace
        """
        level, fromlist = self.popn(2)
        frame = self.frame
        self.push(
            __import__(name, frame.f_globals, frame.f_locals, fromlist, level)
        )

    def byte_IMPORT_STAR(self):
        """
        Loads all symbols not starting with '_'
        directly from the module TOS to the local namespace.
        The module is popped after loading all names.
        This opcode implements from module import *
        """
        mod = self.pop()
        for attr in dir(mod):
            if attr[0] != '_':
                self.frame.f_locals[attr] = getattr(mod, attr)

    def byte_IMPORT_FROM(self, name):
        """
        Loads the attribute co_names[namei]
        from the module found in TOS.
        The resulting object is pushed onto the stack,
        to be subsequently stored by a STORE_FAST instruction
        """
        mod = self.top()
        self.push(getattr(mod, name))

    # Coroutine

    def byte_GET_AWAITABLE(self):
        """
        Implements TOS = get_awaitable(TOS),
        where get_awaitable(o) returns o
        if o is a coroutine object or
        a generator object with the CO_ITERABLE_COROUTINE flag,
        or resolves o.__await__
        """
        pass

    def byte_GET_AITER(self):
        """
        Implements TOS = get_awaitable(TOS.__aiter__())
        """
        pass

    def byte_GET_ANEXT(self):
        """
        Implements PUSH(get_awaitable(TOS.__anext__()))
        """
        pass

    def byte_BEFORE_ASYNC_WITH(self):
        """
        Resolves __aenter__ and __aexit__
        from the object on top of the stack.
        Pushes __aexit__ and result of __aenter__() to the stack
        """
        pass

    def byte_SETUP_ASYNC_WITH(self):
        """
        Creates a new frame object
        """
        pass

    ## And the rest...

    def byte_PRINT_EXPR(self):
        """
        Implements the expression statement for the interactive mode.
        TOS is removed from the stack and printed.
        In non-interactive mode, an expression statement is terminated with POP_TOP.
        """
        print(self.pop())

    def byte_EXTENDED_ARG(self, ext):
        """
        Prefixes any opcode which has an argument too big
        to fit into the default two bytes.
        ext holds two additional bytes which,
        taken together with the subsequent opcode’s argument,
        comprise a four-byte argument,
        ext being the two most-significant bytes
        """
        self.EXTENDED_ARG_ext = ext

    def byte_FORMAT_VALUE(self, flags):
        """
        Used for implementing formatted literal strings (f-strings).
        Pops an optional fmt_spec from the stack,
        then a required value.
        Formatting is performed using PyObject_Format().
        The result is pushed on the stack.

        flags is interpreted as follows:
        (flags & 0x03) == 0x00: value is formatted as-is.
        (flags & 0x03) == 0x01: call str() on value before formatting it.
        (flags & 0x03) == 0x02: call repr() on value before formatting it.
        (flags & 0x03) == 0x03: call ascii() on value before formatting it.
        (flags & 0x04) == 0x04: pop fmt_spec from the stack and use it, else use an empty fmt_spec.
        """
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
            str_2_push = fmt_spec(self.pop())
        self.push(str_2_push)

    def byte_LOAD_BUILD_CLASS(self):
        """
        Pushes builtins.__build_class__() onto the stack.
        It is later called by CALL_FUNCTION to construct a class
        """
        self.push(build_class)


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
