from . import tensor

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if not p in self.u:
                self.u[p] = p.grad.data
            else:
                self.u[p] = self.u[p] * self.momentum + (1-self.momentum)* (p.grad)
            p = (1 - self.lr * self.weight_decay) * p.data - self.lr * ( self.u[p] )
    

class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.u = {}
        self.v = {}

    def step(self):
        for p in self.params:
            if not p in self.u:
                self.u[p] =  (1-self.beta1) * tensor.Tensor(p.grad.data)
            if not p in self.v:
                self.v[p] =  (1-self.beta2) * tensor.Tensor(p.grad.data)**2
            else:
                self.u[p] = self.beta1*self.u[p]  + (1-self.beta1)* tensor.Tensor(p.grad.data)
                self.v[p] = self.beta2*self.v[p] + (1-self.beta2)* tensor.Tensor(p.grad.data)**2
            
            v_hat_t = self.v[p] / (1- self.beta2 ** (self.t + 1))
            u_hat_t = self.u[p] / (1- self.beta1 ** (self.t + 1))
            
            p.data = (1 - self.lr * self.weight_decay) * p.data - self.lr * ( u_hat_t / (v_hat_t ** (1/2) + self.eps) )
            
            self.t += 1
