"""classs tests for Byterun."""

from . import vmtest


class TestClasses(vmtest.VmTestCase):
    def test_classes(self):
        self.assert_ok("""\
            class Thing(object):
                def __init__(self, x):
                    self.x = x
                def meth(self, y):
                    return self.x * y
            thing1 = Thing(2)
            thing2 = Thing(3)
            print(thing1.x, thing2.x)
            print(thing1.meth(4), thing2.meth(5))
            """)

    def test_calling_methods_wrong(self):
        self.assert_ok("""\
            class Thing(object):
                def __init__(self, x):
                    self.x = x
                def meth(self, y):
                    return self.x * y
            thing1 = Thing(2)
            print(Thing.meth(14))
            """, raises=TypeError)

    def test_calling_subclass_methods(self):
        self.assert_ok("""\
            class Thing(object):
                def foo(self):
                    return 17
    
            class SubThing(Thing):
                pass
    
            st = SubThing()
            print(st.foo())
            """)

    def test_subclass_attribute(self):
        self.assert_ok("""\
            class Thing(object):
                def __init__(self):
                    self.foo = 17
            class SubThing(Thing):
                pass
            st = SubThing()
            print(st.foo)
            """)

    def test_subclass_attributes_not_shared(self):
        self.assert_ok("""\
            class Thing(object):
                foo = 17
            class SubThing(Thing):
                foo = 25
            st = SubThing()
            t = Thing()
            assert st.foo == 25
            assert t.foo == 17
            """)

    def test_object_attrs_not_shared_with_class(self):
        self.assert_ok("""\
            class Thing(object):
                pass
            t = Thing()
            t.foo = 1
            Thing.foo""", raises=AttributeError)

    def test_data_descriptors_precede_instance_attributes(self):
        self.assert_ok("""\
            class Foo(object):
                pass
            f = Foo()
            f.des = 3
            class Descr(object):
                def __get__(self, obj, cls=None):
                    return 2
                def __set__(self, obj, val):
                    raise NotImplementedError
            Foo.des = Descr()
            assert f.des == 2
            """)

    def test_instance_attrs_precede_non_data_descriptors(self):
        self.assert_ok("""\
            class Foo(object):
                pass
            f = Foo()
            f.des = 3
            class Descr(object):
                def __get__(self, obj, cls=None):
                    return 2
            Foo.des = Descr()
            assert f.des == 3
            """)

    def test_subclass_attributes_dynamic(self):
        self.assert_ok("""\
            class Foo(object):
                pass
            class Bar(Foo):
                pass
            b = Bar()
            Foo.baz = 3
            assert b.baz == 3
            """)

    def test_attribute_access(self):
        self.assert_ok("""\
            class Thing(object):
                z = 17
                def __init__(self):
                    self.x = 23
            t = Thing()
            print(Thing.z)
            print(t.z)
            print(t.x)
            """)

        self.assert_ok("""\
            class Thing(object):
                z = 17
                def __init__(self):
                    self.x = 23
            t = Thing()
            print(t.xyzzy)
            """, raises=AttributeError)

    def test_staticmethods(self):
        self.assert_ok("""\
            class Thing(object):
                @staticmethod
                def smeth(x):
                    print(x)
                @classmethod
                def cmeth(cls, x):
                    print(x)
    
            Thing.smeth(1492)
            Thing.cmeth(1776)
            """)

    def test_unbound_methods(self):
        self.assert_ok("""\
            class Thing(object):
                def meth(self, x):
                    print(x)
            m = Thing.meth
            m(Thing(), 1815)
            """)

    def test_bound_methods(self):
        self.assert_ok("""\
            class Thing(object):
                def meth(self, x):
                    print(x)
            t = Thing()
            m = t.meth
            m(1815)
            """)

    def test_multiple_classes(self):
        # Making classes used to mix together all the class-scoped values
        # across classes.  This test would fail because A.__init__ would be
        # over-written with B.__init__, and A(1, 2, 3) would complain about
        # too many arguments.
        self.assert_ok("""\
            class A(object):
                def __init__(self, a, b, c):
                    self.sum = a + b + c

            class B(object):
                def __init__(self, x):
                    self.x = x

            a = A(1, 2, 3)
            b = B(7)
            print(a.sum)
            print(b.x)
            """)
