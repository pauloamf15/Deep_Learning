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

def Softmax_normalized(x):
    return np.exp(x - np.max(x,axis=0)) / np.sum(np.exp(x - np.max(x,axis=0)))

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

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
    def update_weight(self, x_i, y_i, learning_rate=0.001, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        
        y_hat=self.predict(x_i)
        if y_hat != y_i:
            self.W[y_i, :] +=  learning_rate*x_i
            self.W[y_hat, :] -=   learning_rate*x_i
        

class LogisticRegression(LinearModel):

    
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b

        # Defining the probabilities for each class
        probs=Softmax_normalized(np.dot(self.W, x_i))
        
        # Creating one hot array
        ey=np.zeros_like(probs)
        ey[y_i]=1
        
        # Creating grad_L
        grad_L=np.einsum('i,j->ij',probs-ey,x_i)
        
        # Updating weights stochastically
        self.W -= learning_rate*grad_L



class MLP(object):
    # Q1.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size, layers=1):
        # Initialize an MLP with a single hidden layer.

        # Initializing weights matrices
        w1 = np.random.normal(0.1,0.1,[hidden_size,n_features])
        w2 = np.random.normal(0.1,0.1,[n_classes,hidden_size])
        self.w=[w1,w2]  # still need to initialize values
        
        # Initializing biases vectors
        b1 = np.zeros(hidden_size)
        b2 = np.zeros(n_classes)
        self.b=[b1,b2] 

        # Initializing vector of input, hiddens and outputs
        self.vec=[np.zeros(n_features), np.zeros(hidden_size),np.zeros(n_classes)] 
    
    
    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.

        # Initializing local variables
        pred_tot = np.zeros(X.shape[0]) # Predictions array
        num_layers = len(self.w)        # Number of layers - 1

        # For cycle for each sample
        for n in np.arange(X.shape[0]):
            
            x = X[n]

            for j in range(num_layers):
                lin = (self.w[j]).dot(x) + self.b[j]    # Linear part
                if j < num_layers - 1: x = ReLU(lin)    # ReLU activation
                else: x = Softmax_normalized(lin)       # Output has softmax activation
            
            # Prediction consists on the class that maximizes the probability function softmax
            pred = np.argmax(x)
            pred_tot[n] = pred

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
    
        num_layers=len(self.w)
        for x_i,y_i in zip(X,y):

            # Forward
            self.vec[0] = x_i
            z=[]
            z.append(x_i)
            for j in range(num_layers):
                lin = (self.w[j]).dot(self.vec[j]) + self.b[j]    # Linear part
                z.append(lin)
                if j < num_layers - 1: self.vec[j+1] = ReLU(lin)  # ReLU activation
                else: self.vec[j+1] = Softmax_normalized(lin)
            
            # Backward
            
            # One hot array
            ey = np.zeros_like(self.vec[-1])
            ey[y_i] = 1   

            # Grad_z wrt to last z
            grad_z = self.vec[-1] - ey

            for k in range(len(self.vec)-1, 0, -1):
                
                # Gradients of weights and biases
                gradw = np.einsum('i,j->ij', grad_z, self.vec[k-1]) # Outer product
                gradb = grad_z[:]  # [:] avoids bad assignment

                # Gradient of hidden layer below
                gradh = np.einsum('ki,k->i', self.w[k-1], grad_z)   # Inner product of transposed matrix with array
                
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
    parser.add_argument('-epochs', default=20, type=int,
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
    

    # Initializing the model
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

    # Plot
    plot(epochs, valid_accs, test_accs)
    # Last accuracies
    print(valid_accs[-1], test_accs[-1])


if __name__ == '__main__':
    main()
