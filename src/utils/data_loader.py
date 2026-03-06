import torch
import torchvision
import numpy as np

class DataLoader:
    def __init__(self,dataset_name,seed=42):
        self.dataset_name=dataset_name
        self.seed=seed
        self.X_train=None
        self.y_train=None
        self.X_test=None
        self.y_test=None

    def load_data(self): #we load botht he datasets here and create training and test sets.
        if self.dataset_name=="mnist":
            trainset=torchvision.datasets.MNIST(root="./data",train=True,download=True)
            testset=torchvision.datasets.MNIST(root="./data",train=False,download=True)
        elif self.dataset_name=="fashion_mnist":
            trainset=torchvision.datasets.FashionMNIST(root="./data",train=True,download=True)
            testset=torchvision.datasets.FashionMNIST(root="./data",train=False,download=True)
        else:
            raise ValueError("Invalid dataset")
        X_train=trainset.data.numpy().astype(np.float32)
        y_train=trainset.targets.numpy()
        X_test=testset.data.numpy().astype(np.float32)
        y_test=testset.targets.numpy()
        X_train/=255.0
        X_test/=255.0
        X_train=X_train.reshape(X_train.shape[0],-1)
        X_test=X_test.reshape(X_test.shape[0],-1)
        num_classes=10
        y_train=np.eye(num_classes,dtype=np.float32)[y_train]
        y_test=np.eye(num_classes,dtype=np.float32)[y_test]
        self.X_train=X_train
        self.y_train=y_train
        self.X_test=X_test
        self.y_test=y_test
        
    def get_batches(self,X,y,batch_size):
        np.random.seed(self.seed)
        n=X.shape[0]
        perm=np.random.permutation(n)
        X=X[perm]
        y=y[perm]
        for i in range(0,n,batch_size):
            yield X[i:i+batch_size],y[i:i+batch_size]
