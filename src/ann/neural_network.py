import numpy as np
from .neural_layer import Linear
from .activations import ReLU,Tanh,Sigmoid
from .objective_functions import MSE,CrossEntropy
from .optimizers import SGD,Momentum,NAG,RMSProp

class NeuralNetwork:

    def __init__(self,cli_args): #initialize all CLI args to process specific configurations

        self.layers=[]
        self.loss=None
        self.optimizer=None
        input_size=784
        output_size=10
        hidden_sizes=cli_args.hidden_size
        prev_size=input_size
        for h in hidden_sizes:
            self.layers.append(Linear(prev_size,h,cli_args.weight_init))
            if cli_args.activation=="relu":
                self.layers.append(ReLU())
            elif cli_args.activation=="tanh":
                self.layers.append(Tanh())
            elif cli_args.activation=="sigmoid":
                self.layers.append(Sigmoid())
            prev_size=h
        self.layers.append(Linear(prev_size,output_size,cli_args.weight_init))

        if cli_args.loss=="cross_entropy":
            self.loss=CrossEntropy()
        elif cli_args.loss=="mse":
            self.loss=MSE()
        lr=cli_args.learning_rate
        if cli_args.optimizer=="sgd":
            self.optimizer=SGD(lr)
        elif cli_args.optimizer=="momentum":
            self.optimizer=Momentum(lr,0.9)
        elif cli_args.optimizer=="nag":
            self.optimizer=NAG(lr,0.9)
        elif cli_args.optimizer=="rmsprop":
            self.optimizer=RMSProp(lr)
        elif cli_args.optimizer=="adam":
            self.optimizer=Adam(lr)
        elif cli_args.optimizer=="nadam":
            self.optimizer=Nadam(lr)

    def forward(self,X): #the forward pass leading to calcuation of loss function
        output=X
        for layer in self.layers:
            output=layer.forward(output)
        return output
    
    def backward(self,y_true,y_pred): #backward pass calculating gradients
        gradient=self.loss.backward()
        for layer in reversed(self.layers):
            if hasattr(layer,"backward"):
                gradient=layer.backward(gradient)

    def update_weights(self): #weight update step
        for layer in self.layers:
            if hasattr(layer,"W"):
                self.optimizer.update(layer)

    def train(self,X_train,y_train,epochs,batch_size): #training loop
        n=X_train.shape[0]
        for epoch in range(epochs):
            perm=np.random.permutation(n)
            X_train=X_train[perm]
            y_train=y_train[perm]
            epoch_loss=0
            for i in range(0,n,batch_size):
                X_batch=X_train[i:i+batch_size]
                y_batch=y_train[i:i+batch_size]
                y_pred=self.forward(X_batch)
                loss=self.loss.forward(y_batch,y_pred)
                epoch_loss+=loss
                self.backward(y_batch,y_pred)
                self.update_weights()
            epoch_loss/=(n//batch_size)
            print("Epoch:",epoch+1,"Loss:",epoch_loss)

    def evaluate(self,X,y): #eval step
        y_pred=self.forward(X)
        preds=np.argmax(y_pred,axis=1)
        labels=np.argmax(y,axis=1)
        acc=np.mean(preds==labels)
        return acc   
    
    def get_weights(self): 
        weights=[]
        for layer in self.layers:
            if hasattr(layer,"W"):
                weights.append({"W":layer.W,"b":layer.b})
        return weights
    
    def set_weights(self,weights):
        idx=0
        for layer in self.layers:
            if hasattr(layer,"W"):
                layer.W=weights[idx]["W"]
                layer.b=weights[idx]["b"]
                idx+=1

