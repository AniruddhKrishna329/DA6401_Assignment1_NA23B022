import numpy as np
class ReLU: #implements the ReLU activation function
    def __init__(self):
        self.Z=None
        self.output=None
    def forward(self,Z):
        self.Z=Z
        self.output=np.maximum(0,Z)
        return self.output
    def backward(self,dA):
        dZ=dA*(self.Z>0) #derivative
        return dZ
    
class Sigmoid: #Sigmoid activation function
    def __init__(self):
        self.out=None
        self.output=None
    def forward(self,Z):
        self.out=1/(1+np.exp(-Z))
        self.output=self.out
        return self.output
    def backward(self,dA):
        dZ=dA*self.out*(1-self.out) #derivative
        return dZ

class Tanh: #Tanh activation function
    def __init__(self):
        self.out=None
        self.output=None
    def forward(self,Z):
        self.out=np.tanh(Z)
        self.output=self.out
        return self.output
    def backward(self,dA):
        dZ=dA*(1-self.out**2) #derivative
        return dZ
