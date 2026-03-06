import numpy as np

class SGD: #stochastic gradient descent 
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,layer):
        layer.W-=self.lr*layer.grad_W
        layer.b-=self.lr*layer.grad_b

class Momentum: #momentum-based optimization which helps accelerate gradients in the right direction, leading to faster convergence
    def __init__(self,lr=0.01,beta=0.9):
        self.lr=lr
        self.beta=beta
        self.vW={}
        self.vb={}
    def update(self,layer):
        if layer not in self.vW:
            self.vW[layer]=np.zeros_like(layer.W)
            self.vb[layer]=np.zeros_like(layer.b)
        self.vW[layer]=self.beta*self.vW[layer]+layer.grad_W
        self.vb[layer]=self.beta*self.vb[layer]+layer.grad_b
        layer.W-=self.lr*self.vW[layer]
        layer.b-=self.lr*self.vb[layer]

class NAG: #Nesterov Accelerated Gradient, a variant of momentum that looks ahead to where the parameters will be after the momentum step, and calculates gradients at that point, leading to better convergence
    def __init__(self,lr=0.01,gamma=0.9):
        self.lr=lr
        self.gamma=gamma
        self.vW={}
        self.vb={}
    def update(self,layer):
        if layer not in self.vW:
            self.vW[layer]=np.zeros_like(layer.W)
            self.vb[layer]=np.zeros_like(layer.b)
        v_prev_W=self.vW[layer]
        v_prev_b=self.vb[layer]
        self.vW[layer]=self.gamma*self.vW[layer]+self.lr*layer.grad_W
        self.vb[layer]=self.gamma*self.vb[layer]+self.lr*layer.grad_b
        layer.W-=self.gamma*v_prev_W+self.vW[layer]
        layer.b-=self.gamma*v_prev_b+self.vb[layer]

class RMSProp: #Root Mean Square Propagation, an adaptive learning rate method that divides the learning rate by an exponentially decaying average of squared gradients
    def __init__(self,lr=0.001,beta=0.9,eps=1e-8):
        self.lr=lr
        self.beta=beta
        self.eps=eps
        self.sW={}
        self.sb={}
    def update(self,layer):
        if layer not in self.sW:
            self.sW[layer]=np.zeros_like(layer.W)
            self.sb[layer]=np.zeros_like(layer.b)
        self.sW[layer]=self.beta*self.sW[layer]+layer.grad_W**2
        self.sb[layer]=self.beta*self.sb[layer]+layer.grad_b**2
        layer.W-=self.lr*layer.grad_W/(np.sqrt(self.sW[layer])+self.eps)
        layer.b-=self.lr*layer.grad_b/(np.sqrt(self.sb[layer])+self.eps)
'''extra
class Adam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.mW={}
        self.mb={}
        self.vW={}
        self.vb={}
        self.t=0
    def update(self,layer):
        if layer not in self.mW:
            self.mW[layer]=np.zeros_like(layer.W)
            self.mb[layer]=np.zeros_like(layer.b)
            self.vW[layer]=np.zeros_like(layer.W)
            self.vb[layer]=np.zeros_like(layer.b)
        self.t+=1
        self.mW[layer]=self.beta1*self.mW[layer]+layer.grad_W
        self.mb[layer]=self.beta1*self.mb[layer]+layer.grad_b
        self.vW[layer]=self.beta2*self.vW[layer]+layer.grad_W**2
        self.vb[layer]=self.beta2*self.vb[layer]+layer.grad_b**2
        mW_hat=self.mW[layer]/(1-self.beta1**self.t)
        mb_hat=self.mb[layer]/(1-self.beta1**self.t)
        vW_hat=self.vW[layer]/(1-self.beta2**self.t)
        vb_hat=self.vb[layer]/(1-self.beta2**self.t)
        layer.W-=self.lr*mW_hat/(np.sqrt(vW_hat)+self.eps)
        layer.b-=self.lr*mb_hat/(np.sqrt(vb_hat)+self.eps)

class Nadam:
    def __init__(self,lr=0.001,beta1=0.9,beta2=0.999,eps=1e-8):
        self.lr=lr
        self.beta1=beta1
        self.beta2=beta2
        self.eps=eps
        self.mW={}
        self.mb={}
        self.vW={}
        self.vb={}
        self.t=0
    def update(self,layer):
        if layer not in self.mW:
            self.mW[layer]=np.zeros_like(layer.W)
            self.mb[layer]=np.zeros_like(layer.b)
            self.vW[layer]=np.zeros_like(layer.W)
            self.vb[layer]=np.zeros_like(layer.b)
        self.t+=1
        self.mW[layer]=self.beta1*self.mW[layer]+layer.grad_W
        self.mb[layer]=self.beta1*self.mb[layer]+layer.grad_b
        self.vW[layer]=self.beta2*self.vW[layer]+layer.grad_W**2
        self.vb[layer]=self.beta2*self.vb[layer]+layer.grad_b**2
        mW_hat=self.mW[layer]/(1-self.beta1**self.t)
        mb_hat=self.mb[layer]/(1-self.beta1**self.t)
        vW_hat=self.vW[layer]/(1-self.beta2**self.t)
        vb_hat=self.vb[layer]/(1-self.beta2**self.t)
        layer.W-=self.lr*(self.beta1*mW_hat+layer.grad_W)/(np.sqrt(vW_hat)+self.eps)
        layer.b-=self.lr*(self.beta1*mb_hat+layer.grad_b)/(np.sqrt(vb_hat)+self.eps)
'''