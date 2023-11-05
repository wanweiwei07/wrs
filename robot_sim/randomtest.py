class A(object):
    def __init__(self):
        pass

    def a_func(self, test_param, b, **kwargs):
        pass


class ASub(A):
    def __init__(self):
        super().__init__()

    def a_func(self, test_param, **kwargs):
        pass


te = ASub()
t