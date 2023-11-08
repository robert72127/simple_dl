
'''
Numpy wrapper with few extra methods
'''
import operator
from functools import reduce

from numpy import *

# math.prod not in Python 3.7
def prod(array, x):
    return reduce(operator.mul, x, 1)

def permute(a, axes=None):
    return transpose(a,axes)


def randn(*shape, dtype="float32"):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton
    return random.randn(*shape).astype(dtype)

def rand(*shape, dtype="float32"):
    # note: numpy doesn't support types within standard random routines, and
    # .astype("float32") does work if we're generating a singleton
    return random.rand(*shape).astype(dtype)

def one_hot(n, i, dtype="float32"):
    return eye(n, dtype=dtype)[i]




