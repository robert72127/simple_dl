"""
Neural Network modules
"""
from typing import List, Callable, Any
import simple_dl
from .import ops
from .tensor import Tensor
from . import ndarray, init


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""

def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.use_bias = bias
        self.weight =Parameter(init.kaiming_uniform(fan_in=self.in_features, fan_out=self.out_features))
        if self.use_bias:
            self.bias = Parameter(init.kaiming_uniform(fan_in=out_features, fan_out=1))
            self.bias = ops.transpose(self.bias)

    def forward(self, X: Tensor) -> Tensor:
        result = X @ self.weight
        if self.use_bias:
            result += self.bias
        return result



class Flatten(Module):
    def forward(self, X):
        shape = list(X.shape)
        new_shape = 1
        for s in shape[1:]:
            new_shape *= s
        
        new_shape = [shape[0], new_shape]

        return ops.reshape(X, new_shape)


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        one_hot = init.one_hot(logits.shape[-1], y) * logits
        return ops.summation(ops.logsumexp(logits, axes=(1,) ) - ops.summation(one_hot,axes=1)) / logits.shape[0]



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        
        self.weight = Parameter(init.ones(self.dim,device=device,dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim,device=device,dtype=dtype, requires_grad=True))

        self.running_mean = init.zeros(self.dim,device=device,dtype=dtype, requires_grad=False)
        self.running_var = init.ones(self.dim,device=device,dtype=dtype, requires_grad=False)


    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            mean = ops.summation(x, axes=0) / x.shape[0]
            self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean
            new_shape = [1] + list(mean.shape) 
            mean = ops.reshape(mean, new_shape)
            mean = ops.broadcast_to(mean, x.shape)

            var =  ops.summation((x - mean)**2, axes=0) / (x.shape[0])
            self.running_var = (1-self.momentum) * self.running_var + self.momentum * var
            var = ops.reshape(var, new_shape)
            var = ops.broadcast_to(var, x.shape)

            return self.weight * ((x-mean) / (var + self.eps)**(1/2) ) + self.bias
        else:

            return (x - self.momentum * self.running_mean) / (self.running_var  + self.eps)**(1/2)


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(self.dim,device=device,dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim,device=device,dtype=dtype, requires_grad=True))
        self.bias = ops.transpose(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        mean = ops.summation(x, axes=1) / x.shape[1]
        new_shape =  list(mean.shape) + [1]
        mean = ops.reshape(mean, new_shape)
        mean = ops.broadcast_to(mean, x.shape)

        var =  ops.summation((x - mean)**2, axes=1) / (x.shape[1])
        var = ops.reshape(var, new_shape)
        var = ops.broadcast_to(var, x.shape)

        return self.weight * ((x-mean) / (var + self.eps)**(1/2) ) + self.bias


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        mask = init.rand(*x.shape, low=0,high=1)
        f = lambda x: (x<self.p) / (1-self.p)
        mask = f(mask.cached_data)
        return  x * mask


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(x) + x



