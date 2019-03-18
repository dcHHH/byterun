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

![interpreter-callstack](/Users/hdc/Documents/learning/code/byterun/documents/interpreter-callstack.png)

#对象类型

##`VirtualMachine`对象
它管理高层结构，frame调用栈，指令到操作的映射。

程序运行时只有一个`VirtualMachine`被创建。`VirtualMachine`保存调用栈，异常状态，在frame中传递的返回值。它的入口点是`run_code`方法，它以编译后的code object为参数，以创建一个frame为开始，然后运行这个frame。这个frame可能再创建出新的frame；调用栈随着程序的运行增长缩短。当第一个frame返回时，执行结束。

##`Frame`对象
frame是一个属性的集合，它没有任何方法。

这些属性包括由编译器生成的code object；局部，全局和内置命名空间；前一个frame的引用；一个数据栈；一个块栈；最后执行的指令。

**只读属性**：

- `f_back` ：调用栈中的上一个frame；如果当前frame位于栈顶，则为空。

- `f_code` ：当前frame中正在被执行的code object。

- `f_locals` ：本地命名空间

- `f_globals` ：全局命名空间。

- `f_builtins` ：内置命名空间

- `f_lasti` ：上一条字节码指令在`f_code`中的偏移位置。

**可变属性**：

- `f_trace`：异常调用时的句柄，或者为None。

- `f_lineno` ：当前字节码对应的源码行数。



##`Function`类
User-defined functions:

| Attribute                                                    | Meaning                                                      |           |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------- |
| `__doc__`                                                    | The function’s documentation string, or `None` if unavailable; not inherited by subclasses | Writable  |
| [`__name__`](https://docs.python.org/3.6/library/stdtypes.html#definition.__name__) | The function’s name                                          | Writable  |
| [`__qualname__`](https://docs.python.org/3.6/library/stdtypes.html#definition.__qualname__) | The function’s [qualified name](https://docs.python.org/3.6/glossary.html#term-qualified-name)*New in version 3.3.* | Writable  |
| `__module__`                                                 | The name of the module the function was defined in, or `None` if unavailable. | Writable  |
| `__defaults__`                                               | A tuple containing default argument values for those arguments that have defaults, or `None` if no arguments have a default value | Writable  |
| `__code__`                                                   | The code object representing the compiled function body.     | Writable  |
| `__globals__`                                                | A reference to the dictionary that holds the function’s global variables — the global namespace of the module in which the function was defined. | Read-only |
| [`__dict__`](https://docs.python.org/3.6/library/stdtypes.html#object.__dict__) | The namespace supporting arbitrary function attributes.      | Writable  |
| `__closure__`                                                | `None` or a tuple of cells that contain bindings for the function’s free variables. | Read-only |
| `__annotations__`                                            | A dict containing annotations of parameters. The keys of the dict are the parameter names, and `'return'` for the return annotation, if provided. | Writable  |
| `__kwdefaults__`                                             | A dict containing defaults for keyword-only parameters.      | Writable  |

##`Block`类


##`Generator`类

生成器对象

## `Coroutine`类

协程对象

# 前置知识

## 字节码

https://docs.python.org/3/reference/datamodel.html

`co_name` ：对象的名字。

 `co_argcount` ：位置参数的数目。

`co_nlocals` ：局部变量的数目。

 `co_consts` ：常量元组。

 `co_names` ：常量中的字符串对象元组。

 `co_varnames` ：局部变量元组。

`co_cellvars` ：嵌套函数所用局部变量元组。

 `co_freevars`：自由变量元组。

 `co_code` ：编译所得字节码。

 `co_filename` ：代码编译时的文件名。

 `co_firstlineno`：对应代码在源码的起始行。

 `co_lnotab` ：字节码与源码之间行号的对应关系（bytes）

 `co_stacksize` is the required stack size (including local variables);

 `co_flags` is an integer encoding a number of flags for the interpreter。

