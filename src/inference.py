
import argparse
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import DataLoader
from src.ann.neural_network import NeuralNetwork

def parse_arguments(): #all the args that we would be passing in the command line for inference
    parser=argparse.ArgumentParser(description="Run inference")
    parser.add_argument("-mp","--model_path",type=str,required=True)
    parser.add_argument("-d","--dataset",type=str,required=True,
                        choices=["mnist","fashion_mnist"])
    parser.add_argument("-b","--batch_size",type=int,default=64)
    parser.add_argument("-sz","--hidden_size",nargs="+",type=int,default=[64])
    parser.add_argument("-a","--activation",type=str,default="relu",
                        choices=["relu","tanh","sigmoid"])
    parser.add_argument("-l","--loss",type=str,default="cross_entropy",
                        choices=["cross_entropy","mse"])
    parser.add_argument("-o","--optimizer",type=str,default="adam",
                        choices=["sgd","momentum","nag","rmsprop"])
    parser.add_argument("-lr","--learning_rate",type=float,default=0.001)
    parser.add_argument("-wi","--weight_init",type=str,default="xavier",
                        choices=["zeros","xavier"])
    return parser.parse_args()

def load_model(model_path,args):
    params=np.load(model_path,allow_pickle=True)
    model=NeuralNetwork(args)
    param_index=0
    for layer in model.layers:
        if hasattr(layer,"W"):
            layer.W=params[param_index]["W"]
            layer.b=params[param_index]["b"]
            param_index+=1
    return model

def evaluate_model(model,X_test,y_test): #eval loop
    logits=model.forward(X_test)
    loss=model.loss.forward(y_test,logits)
    y_pred_classes=np.argmax(logits,axis=1)
    y_true_classes=np.argmax(y_test,axis=1)
    accuracy=np.mean(y_pred_classes==y_true_classes)
    num_classes=y_test.shape[1]
    precisions=[]
    recalls=[]
    f1s=[]
    conf_matrix=np.zeros((num_classes,num_classes),dtype=int)

    for i in range(len(y_true_classes)):
        conf_matrix[y_true_classes[i],y_pred_classes[i]]+=1

    for c in range(num_classes): #all the metrics we want to know
        TP=np.sum((y_pred_classes==c)&(y_true_classes==c))
        FP=np.sum((y_pred_classes==c)&(y_true_classes!=c))
        FN=np.sum((y_pred_classes!=c)&(y_true_classes==c))
        precision=TP/(TP+FP+1e-15)
        recall=TP/(TP+FN+1e-15)
        f1=2*precision*recall/(precision+recall+1e-15)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precision=np.mean(precisions)
    recall=np.mean(recalls)
    f1=np.mean(f1s)
    misclassified=(y_pred_classes!=y_true_classes)
    return{
        "logits":logits,
        "loss":loss,
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1":f1,
        "confusion_matrix":conf_matrix,
        "misclassified":misclassified,
        "y_pred":y_pred_classes,
        "y_true":y_true_classes
    }

def plot_confusion_matrix(cm): #code to plot the confusion matrix
    plt.figure(figsize=(6,6))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center",color="white")
    plt.savefig("confusion_matrix.png")
    plt.close()

def plot_failure_heatmap(cm): #code to know misclassified points
    failure_matrix=cm.copy()
    for i in range(failure_matrix.shape[0]):
        failure_matrix[i,i]=0
    plt.figure(figsize=(6,6))
    plt.imshow(failure_matrix)
    plt.title("Misclassification Heatmap")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.savefig("misclassification_heatmap.png")
    plt.close()


def plot_misclassified(X,misclassified,y_true,y_pred):
    idx=np.where(misclassified)[0][:16]
    plt.figure(figsize=(6,6))
    for i in range(len(idx)):
        plt.subplot(4,4,i+1)
        plt.imshow(X[idx[i]].reshape(28,28),cmap="gray")
        plt.title(str(y_true[idx[i]])+"->"+str(y_pred[idx[i]]))
        plt.axis("off")
    plt.suptitle("Misclassified Samples")
    plt.savefig("misclassified_samples.png")
    plt.close()

def main():
    args=parse_arguments()
    data=DataLoader(args.dataset)
    data.load_data()
    model=load_model(args.model_path,args)
    results=evaluate_model(model,data.X_test,data.y_test)
    print("Accuracy:",results["accuracy"])
    print("Precision:",results["precision"])
    print("Recall:",results["recall"])
    print("F1 Score:",results["f1"])
    print("Confusion Matrix:")
    print(results["confusion_matrix"])
    plot_confusion_matrix(results["confusion_matrix"])
    plot_failure_heatmap(results["confusion_matrix"])
    plot_misclassified(
        data.X_test,
        results["misclassified"],
        results["y_true"],
        results["y_pred"]
    )

if __name__=="__main__":
    main()

