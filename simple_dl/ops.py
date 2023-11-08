from numbers import Number
from typing import Optional, List, Tuple, Union
from .tensor import Tensor, State, TensorTuple, TensorOp, TensorTupleOp, Op
from . import ndarray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])

def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> State:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(ndarray.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: ndarray.array, b: ndarray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: ndarray.array):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: ndarray.array, b: ndarray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: ndarray.array):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: ndarray.array) -> ndarray:
        return ndarray.array.power(a, self.scalar)        

    def gradient(self, out_grad, node):
        return  out_grad *   power_scalar(node.inputs[0], self.scalar-1) * self.scalar


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return ndarray.array.divide(a, b)        

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * -(lhs) / (rhs * rhs)        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return ndarray.array.divide(a, self.scalar)        

    def gradient(self, out_grad, node):
        return out_grad / self.scalar        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axis1, axis2 = self.axes if self.axes else [len(a.shape)-2, len(a.shape)-1]
        return ndarray.array.swapaxes(a,axis1,axis2)        

    def gradient(self, out_grad, node):
        return transpose(out_grad, axes=self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return ndarray.array.reshape(a, newshape=self.shape)        

    def gradient(self, out_grad, node):
        prev_shape = node.inputs[0].cached_data.shape
        return reshape(out_grad, prev_shape)        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return ndarray.array.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
    
        original_shape = node.inputs[0].cached_data.shape
        sum_by = []
        for i in range(len(self.shape)):
            if (i >= len(original_shape)) or self.shape[i] != original_shape[i]:
                sum_by += [i]

        tmp_shape = [ out_grad.shape[i] if not i in sum_by else 1 for i in range(len(out_grad.shape)) ] 

        return  broadcast_to( reshape(summation(out_grad, tuple(sum_by)), tmp_shape ), original_shape )
                


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return ndarray.array.sum(a, axis=self.axes)        

    def gradient(self, out_grad, node):
        original_shape = list(node.inputs[0].shape)
        out_shape = list(out_grad.cached_data.shape)
        if self.axes and original_shape:
            if not isinstance(self.axes, tuple):
                self.axes = (self.axes,)
            out_shape = [original_shape[i] if i not in self.axes else 1 for i in range(len(original_shape))]
        out_grad = reshape(out_grad, out_shape)
        return broadcast_to(out_grad, original_shape)
            


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @  b        

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs

        return   matmul(out_grad, transpose(rhs) ), matmul( transpose(lhs), out_grad)        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a        

    def gradient(self, out_grad, node):
        return  negate(out_grad)        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return ndarray.array.log(a)        

    def gradient(self, out_grad, node):
        deriv = Tensor((1/node.inputs[0].cached_data))
        return multiply(out_grad, deriv)        

def log(a):
    return Log()(a)

class Exp(TensorOp):
    def compute(self, a):
        a = ndarray.array.clip(a, -709, 709)
        return ndarray.array.exp(a)        

    def gradient(self, out_grad, node):
        return multiply(out_grad, exp(node.inputs[0]))        

def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        relu = lambda x: x * (x > 0)
        return relu(a)        
        

    def gradient(self, out_grad, node):
        relu_deriv = lambda x : 1 * (x > 0)
        deriv = Tensor(relu_deriv(node.inputs[0].cached_data))
        return multiply(out_grad, deriv)


def relu(a):
    return ReLU()(a)

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        max_Z = ndarray.array.max(Z, axis=self.axes) 
        if self.axes:
            new_shape = [Z.shape[i] if i not in self.axes else 1 for i in range(len(Z.shape))]
            max_Z = ndarray.array.reshape(max_Z, new_shape)
         
        lse = ndarray.array.log(ndarray.sum( ndarray.exp(Z - max_Z), axis=self.axes ))
       
        return lse + ndarray.array.reshape(max_Z, lse.shape)

    def gradient(self, out_grad, node):
        exp_input = exp(node.inputs[0])
        sum = summation(exp_input, axes=self.axes)
        
        if self.axes is None:
            self.axes = []
        if  not (isinstance(self.axes, tuple) or isinstance(self.axes, list)):
                 self.axes = (self.axes,)
     
        new_shape = [ exp_input.shape[i] if not i in self.axes else 1  for i in range(len(exp_input.shape))]
        
        if self.axes != []:
            sum  = reshape(sum,  new_shape)
            out_grad = reshape(out_grad, new_shape)

        exp_input = exp_input /  broadcast_to(sum, exp_input.shape)
        
        
        return exp_input* broadcast_to(out_grad, exp_input.shape)

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
