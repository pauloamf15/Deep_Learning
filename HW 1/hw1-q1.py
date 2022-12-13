#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def ReLU(x):
    return x * (x > 0)

def dReLU(x, grad_x):
    return grad_x * (x > 0)

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a

        # Is learning rate necessary? Does not make big diff
        # Bias was already added to inputs -> See loading of data
        # Results not very good
        
        y_hat=self.predict(x_i)
        if y_hat != y_i:
            self.W[y_i, :] +=  x_i
            self.W[y_hat, :] -=   x_i
        



class LogisticRegression(LinearModel):

    
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b

        # Better results

        # Defining the probabilities for each class
        exps=np.exp(np.dot(self.W, x_i))
        Z=np.sum(exps)
        probs=exps/Z
        
        # Creating e_y
        ey=np.zeros_like(probs)
        ey[y_i]=1
        
        # Creating grad_L
        grad_L=np.einsum('i,j->ij',probs-ey,x_i)
        
        # Updating weights stochastically
        self.W -= learning_rate*grad_L



class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size,layers=1):
        # Initialize an MLP with a single hidden layer.

        # Initialize weights matrices
        w1 = np.random.normal(0.1,0.1,[hidden_size,n_features])
        w2 = np.random.normal(0.1,0.1,[n_classes,hidden_size])
        self.w=[w1,w2]  # still need to initialize values
        
        # Initialize biases vectors
        b1 = np.zeros(hidden_size)
        b2 = np.zeros(n_classes)
        self.b=[b1,b2] #check later

        # Initializing zs ( necessary? )
        self.h=[np.zeros(n_features), np.zeros(hidden_size),np.zeros(n_classes)] #check later
    
    
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        pred_tot=np.zeros(X.shape[0])
        num_layers=len(self.w)
        for n in np.arange(X.shape[0]):
            x_i=X[n]
            x_i = (x_i-np.mean(x_i))/np.std(x_i)
            self.h[0] = x_i
            for j in range(num_layers):
                lin = (self.w[j]).dot(self.h[j]) + self.b[j]    # Linear part
                if j < num_layers - 1: self.h[j+1] = ReLU(lin)  # ReLU activation
                else: self.h[j+1] = lin
            self.h[-1]=np.exp(self.h[-1])/np.sum(np.exp(self.h[-1]))
            pred = np.argmax(self.h[-1])
            pred_tot[n]=pred
        return pred_tot


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
     
     #from sklearn.model_selection import train_test_split
     #X_new = (X-np.mean(X))/np.std(X)
     #X_train, _, y_train, _ = train_test_split(X, y, test_size=0.000001, random_state=33)
    
        num_layers=len(self.w)
        for x_i,y_i in zip(X,y):
            x_i = (x_i-np.mean(x_i))/np.std(x_i)
            # Forward
            self.h[0] = x_i
            z=[]
            z.append(x_i)
            for j in range(num_layers):
                lin = (self.w[j]).dot(self.h[j]) + self.b[j]    # Linear part
                z.append(lin)
                if j < num_layers - 1: self.h[j+1] = ReLU(lin)  # ReLU activation
                else: self.h[j+1] = np.exp(lin)/np.sum(np.exp(lin)) # output activation(softmax)
                
            
            ey = np.zeros_like(self.h[-1])
            ey[y_i] = 1 # One hot
            grad_z=self.h[-1]-ey
            #grad_z=probs-ey
            for k in range(len(self.h)-1, 0, -1):
                gradw = np.outer(grad_z,self.h[k-1])

                gradb = grad_z[:]  #python is sometimes a moron and takes a=b as b=a, as in afterwards changing b changes a and vice versa, the [:] makes sure this doesnt happen.

                # Gradient of hidden layer below
                gradh = np.dot(self.w[k-1].T,grad_z)
        
                # Gradient of hidden layer below before activation
                grad_z = dReLU(z[k-1],gradh)

                #Update parameters
                self.w[k-1] -= learning_rate*gradw
                self.b[k-1] -= learning_rate*gradb
       
def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=5, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]
    

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size, opt.layers)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
