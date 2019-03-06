[TOC]

#解释器结构

CPython 使用一个基于栈的虚拟机。

CPython 使用三种类型的栈：

##调用栈（call stack）
这是运行 Python 程序的主要结构。

栈底是程序的入口点。每个函数调用推送一个新的frame到调用栈，每当函数调用返回后，这个帧被销毁。

##数据栈（data stack）
在每个frame中，有一个数据栈（data stack）。

这个栈就是 Python 函数运行的地方。

##块栈（block stack）
在每个frame中，还有一个块栈（block stack）。

它被 Python 用于去跟踪某些类型的控制结构：循环、`try` / `except` 块、以及 `with` 块，全部推入到块栈中。当你退出这些控制结构时，块栈被销毁。

例:

```python
>>> def bar(y):
...     z = y + 3     # <--- (3) ... and the interpreter is here.
...     return z
...
>>> def foo():
...     a = 1
...     b = 2
...     return a + bar(b) # <--- (2) ... which is returning a call to bar ...
...
>>> foo()             # <--- (1) We're in the middle of a call to foo ...
3
```

![interpreter-callstack](/Users/hdc/Documents/learning/code/byterun/原理/interpreter-callstack.png)



#对象类型

##`VirtualMachine`类
它管理高层结构，frame调用栈，指令到操作的映射。

程序运行时只有一个`VirtualMachine`被创建。`VirtualMachine`保存调用栈，异常状态，在frame中传递的返回值。它的入口点是`run_code`方法，它以编译后的code object为参数，以创建一个frame为开始，然后运行这个frame。这个frame可能再创建出新的frame；调用栈随着程序的运行增长缩短。当第一个frame返回时，执行结束。

##`Frame`类
frame是一个属性的集合，它没有任何方法。这些属性包括由编译器生成的code object；局部，全局和内置命名空间；前一个frame的引用；一个数据栈；一个块栈；最后执行的指令。

Frame objects

Frame objects represent execution frames. They may occur in traceback objects (see below).

Special read-only attributes: `f_back` is to the previous stack frame (towards the caller), or `None` if this is the bottom stack frame; `f_code` is the code object being executed in this frame; `f_locals` is the dictionary used to look up local variables; `f_globals` is used for global variables; `f_builtins` is used for built-in (intrinsic) names; `f_lasti` gives the precise instruction (this is an index into the bytecode string of the code object).

Special writable attributes: `f_trace`, if not `None`, is a function called at the start of each source code line (this is used by the debugger); `f_lineno` is the current line number of the frame — writing to this from within a trace function jumps to the given line (only for the bottom-most frame). A debugger can implement a Jump command (aka Set Next Statement) by writing to f_lineno.

##`Function`类
函数对象

##`Block`类


##`Generator`类

生成器对象

## `Coroutine`类

协程对象