
import numpy as np

class Linear:
    def __init__(self,in_size,out_size,w_init="xavier"):
        self.in_size=in_size
        self.out_size=out_size
        if w_init=="zeros":  #all weights initialized to zero
            self.W=np.zeros((in_size,out_size))
        elif w_init=="xavier": #Xavier initialization where weights are initialized from a specific distribution
            limit=np.sqrt(2.0/(in_size+out_size))
            self.W=np.random.randn(in_size,out_size)*limit
        else:
            raise ValueError("Incorrect init method")
        self.b=np.zeros((1,out_size))
        self.grad_W=None
        self.grad_b=None
        self.input=None

    def forward(self,X):
        self.input=X
        return X@self.W+self.b

    def backward(self,dZ):
        batch_size=self.input.shape[0]
        self.grad_W=(self.input.T@dZ)/batch_size
        self.grad_b=np.sum(dZ,axis=0,keepdims=True)/batch_size
        dX=dZ@self.W.T
        return dX