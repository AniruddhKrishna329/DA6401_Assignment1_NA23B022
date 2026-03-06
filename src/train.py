import argparse
import numpy as np
import wandb
import json
from src.utils.data_loader import DataLoader
from src.ann.neural_network import NeuralNetwork
from sklearn.metrics import f1_score

def parse_arguments(): #args we pass to train the model
    parser=argparse.ArgumentParser(description="Train a neural network")
    parser.add_argument("-d","--dataset",type=str,required=True,choices=["mnist","fashion_mnist"])
    parser.add_argument("-e","--epochs",type=int,default=10)
    parser.add_argument("-hl","--hidden_layers",type=int,default=2,choices=[1,2,3,4,5,6])
    parser.add_argument("-b","--batch_size",type=int,default=64)
    parser.add_argument("-l","--loss",type=str,default="cross_entropy",choices=["cross_entropy","mse"])
    parser.add_argument("-o","--optimizer",type=str,default="sgd",choices=["sgd","momentum","nag","rmsprop","adam","nadam"])
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001)
    parser.add_argument("-nhl","--num_layers",type=int,default=2)
    parser.add_argument("-sz","--hidden_size",nargs="+",type=int,default=[64])
    parser.add_argument("-a","--activation",type=str,default="relu",choices=["relu","tanh","sigmoid"])
    parser.add_argument("-wi","--weight_init",type=str,default="xavier",choices=["zeros","xavier"])
    parser.add_argument("--wandb_project",type=str,default="da6401_assignment1")
    parser.add_argument("--model_save_path",type=str,default="best_model_fashion.npy")
    return parser.parse_args()

def compute_accuracy(y_true,y_pred): #acc calculator
    preds=np.argmax(y_pred,axis=1)
    labels=np.argmax(y_true,axis=1)
    return np.mean(preds==labels)

def save_model(model,path):
    weights=[]
    for layer in model.layers:
        if hasattr(layer,"W"):
            weights.append({"W":layer.W,"b":layer.b})
    np.save(path,weights)

def main():
    args=parse_arguments()
    wandb.init(project=args.wandb_project,config=vars(args)) #wandb logs at every step
    config=dict(wandb.config)
    hidden_size=config["hidden_size"]
    if isinstance(hidden_size,int):
        config["hidden_size"]=[hidden_size]
    class ArgsObject:
        pass
    args_obj=ArgsObject()
    for k,v in config.items():
        setattr(args_obj,k,v)
    data_loader=DataLoader(config["dataset"])
    data_loader.load_data()
    X_train=data_loader.X_train
    y_train=data_loader.y_train
    X_test=data_loader.X_test
    y_test=data_loader.y_test
    val_size=int(0.1*X_train.shape[0])
    X_val=X_train[:val_size]
    y_val=y_train[:val_size]
    X_train=X_train[val_size:]
    y_train=y_train[val_size:]

    model=NeuralNetwork(args_obj)

    best_test_f1=0
    for epoch in range(config["epochs"]):
        model.train(X_train,y_train,1,config["batch_size"])
        for layer in model.layers:
            if hasattr(layer,"grad_W") and layer.grad_W is not None:
                for i in range(min(5,layer.grad_W.shape[1])):
                    wandb.log({f"grad_neuron_{i+1}":np.mean(np.abs(layer.grad_W[:,i]))})
                break

        train_logits=model.forward(X_train)
        val_logits=model.forward(X_val)
        test_logits=model.forward(X_test)
        train_acc=compute_accuracy(y_train,train_logits)
        val_acc=compute_accuracy(y_val,val_logits)
        test_acc=compute_accuracy(y_test,test_logits)
        train_loss=model.loss.forward(y_train,train_logits)
        val_loss=model.loss.forward(y_val,val_logits)
        train_preds=np.argmax(train_logits,axis=1)
        test_preds=np.argmax(test_logits,axis=1)
        train_labels=np.argmax(y_train,axis=1)
        test_labels=np.argmax(y_test,axis=1)
        test_f1=f1_score(test_labels,test_preds,average="macro")

        wandb.log({
            "epoch":epoch+1,
            "train_accuracy":train_acc,
            "val_accuracy":val_acc,
            "test_accuracy":test_acc,
            "train_loss":train_loss,
            "val_loss":val_loss,
            "test_f1":test_f1
        })

        print("Epoch:",epoch+1,
              "Train Acc:",train_acc,
              "Val Acc:",val_acc,
              "Test F1:",test_f1)

        if test_f1>best_test_f1:
            best_test_f1=test_f1
            weights=model.get_weights()
            np.save("src/best_model_fashion.npy",weights)
            with open("src/best_model_config_fashion.json","w") as f:
                json.dump(config,f)

    wandb.log({"best_test_f1":best_test_f1})
    wandb.finish()
    print("Training complete!")

if __name__=="__main__":
    main()