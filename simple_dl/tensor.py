from . import ndarray
import simple_dl
from typing import List, Optional, NamedTuple, Tuple, Union, Dict
from collections import namedtuple

# lazy or eager
LAZY_MODE = True
TENSOR_COUNTER = 0

class State:
    """A value in the computational graph."""

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        self.cached_data
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def __init__(self,op, inputs: List["Tensor"],*,num_outputs: int = 1,cached_data: List[object] = None,requires_grad: Optional[bool] = None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op    # from what op was it created
        self.inputs = inputs # parent tensors
        self.num_outputs = num_outputs
        self.cached_data = cached_data # we can just track computational graph and compute this only when needed
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value.__init__(None,[],cached_data=data,requires_grad=requires_grad,)
        return value

    @classmethod
    def make_from_op(cls, op, inputs: List["State"]):
        value = cls.__new__(cls)
        value.__init__(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value

class Tensor(State):
    grad: "Tensor"

    def __init__(self,array,*,dtype=None,requires_grad=True,**kwargs):
        if isinstance(array, Tensor):
            if dtype is None:
                dtype = array.dtype
            if  dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = ndarray.array(
                    array, dtype=dtype
                )
        else:
            dtype = dtype
            cached_data = ndarray.array(array)

        super(Tensor, self).__init__(None,[],cached_data=cached_data,requires_grad=requires_grad,)

    @staticmethod
    def make_from_op(op, inputs: List["State"]):
        tensor = Tensor.__new__(Tensor)
        super(Tensor, tensor).__init__(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        super(Tensor, tensor).__init__(
            None,
            [],
            cached_data=data if not isinstance(data, Tensor) else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the computational graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    #this is called on last node in computational grad
    def backward(self, out_grad=None):
        #if has out_grad call it else init
        out_grad = out_grad if out_grad else  Tensor(ndarray.ones(*self.shape, dtype=self.dtype), dtype=self.dtype, requires_grad=False)
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        self.realize_cached_data()
        return self.cached_data

    def __add__(self, other):
        if isinstance(other, Tensor):
            return simple_dl.EWiseAdd()(self, other)
        else:
            return simple_dl.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return simple_dl.EWiseMul()(self, other)
        else:
            return simple_dl.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return simple_dl.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return simple_dl.EWiseAdd()(self, simple_dl.Negate()(other))
        else:
            return simple_dl.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return simple_dl.EWiseDiv()(self, other)
        else:
            return simple_dl.DivScalar(other)(self)

    def __matmul__(self, other):
        return simple_dl.MatMul()(self, other)

    def matmul(self, other):
        return simple_dl.MatMul()(self, other)

    def sum(self, axes=None):
        return simple_dl.Summation(axes)(self)

    def broadcast_to(self, shape):
        return simple_dl.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return simple_dl.Reshape(shape)(self)

    def __neg__(self):
        return simple_dl.Negate()(self)

    def transpose(self, axes=None):
        return simple_dl.Transpose(axes)(self)

    def log(self):
        return simple_dl.Log()(self)
    
    def relu(self):
        return simple_dl.ReLU()(self)
    
    def exp(self):
        return simple_dl.Exp()(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


class TensorTuple(State):
    '''
    Tuple of tensors
    '''

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return simple_dl.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return simple_dl.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data())

################### Gradient computation ##################################

def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for i in reverse_topo_order:

        v_i = None
        for node in node_to_output_grads_list[i]:
            v_i = node if v_i is None else v_i + node

        i.grad = v_i 

        if i.inputs :
            partial_derivs = i.op.gradient(v_i, i)
            if not isinstance(partial_derivs, tuple):
                partial_derivs = (partial_derivs,)

            for (k, v_k_i) in zip(i.inputs, partial_derivs):
                if not k in node_to_output_grads_list:
                    node_to_output_grads_list[k] = [v_k_i]
                else:
                    node_to_output_grads_list[k] += [v_k_i]
    

def find_topo_sort(node_list: List[State]) -> List[State]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    """
    visited = ()
    stack = []

    for node in node_list:
        if not node in visited:
            visited, stack = topo_sort_dfs(node,visited, stack) 

    return stack    


def topo_sort_dfs(node, visited, stack):
    """Post-order DFS"""

    visited += (node,)
    for innode in node.inputs:
        if not innode in visited:
            visited, stack = topo_sort_dfs(innode,visited, stack)

    stack = stack + [node]     
    return visited, stack      
    
def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)

