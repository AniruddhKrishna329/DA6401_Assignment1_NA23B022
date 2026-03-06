import numpy as np

class MSE: #calculates mean squared error, and its derivative
    def __init__(self):
        self.y_actual=None
        self.y_pred=None
        self.probabs=None
    def forward(self,y_actual,logits):
        self.y_actual=y_actual
        normalized=logits-np.max(logits,axis=1,keepdims=True)
        exps=np.exp(normalized)
        self.probabs=exps/np.sum(exps,axis=1,keepdims=True)
        self.y_pred=self.probabs
        mse=np.mean((y_actual-self.y_pred)**2)
        return mse
    def backward(self):
        dY=2*(self.y_pred-self.y_actual)
        return dY

class CrossEntropy: #calculates cross entropy loss and its derivative
    def __init__(self):
        self.probabs=None
        self.y_actual=None
    def forward(self,y_actual,logits):
        self.y_actual=y_actual
        normalized=logits-np.max(logits,axis=1,keepdims=True)
        exps=np.exp(normalized)
        self.probabs=exps/np.sum(exps,axis=1,keepdims=True)
        crossent=-np.mean(np.sum(y_actual*np.log(self.probabs+1e-15),axis=1))
        return crossent
    def backward(self):
        dZ=self.probabs-self.y_actual
        return dZ
