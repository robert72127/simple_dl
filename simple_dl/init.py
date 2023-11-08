import math
from . import Tensor
from . import ndarray

def rand(*shape, low=0.0, high=1.0, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    array = ndarray.rand(*shape) * (high - low) + low
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)
    
def randn(*shape, mean=0.0, std=1.0, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    array = ndarray.randn(*shape) * std + mean
    return Tensor(array, dtype=dtype, requires_grad=requires_grad)

def constant(*shape, c=1.0,  dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    array = ndarray.ones(*shape, dtype=dtype) * c # note: can change dtype
    return Tensor(array,  dtype=dtype, requires_grad=requires_grad)

def ones(*shape,  dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape, c=1.0, dtype=dtype, requires_grad=requires_grad)

def zeros(*shape,  dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape, c=0.0,  dtype=dtype, requires_grad=requires_grad)

def randb(*shape, p=0.5,  dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    array = ndarray.rand(*shape) <= p
    return Tensor(array,  dtype=dtype, requires_grad=requires_grad)

def one_hot(n, i,  dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    return Tensor(ndarray.one_hot(n,i.numpy(), dtype=dtype),  requires_grad=requires_grad)

def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    a = gain * math.sqrt(6/(fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, requires_grad=False)

def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    std = gain * math.sqrt(2/(fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, requires_grad=False) 

def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    bound = gain * math.sqrt(3/(fan_in))
    return rand(fan_in, fan_out, low=-bound, high=bound, requires_grad=False)

def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, requires_grad=False) 
