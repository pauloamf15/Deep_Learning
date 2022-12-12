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

class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        print(X[:][-1])
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
        # Results not very good
        # print(x_i)
        # self.W=np.concatenate([ np.zeros((self.W.shape[0],1)), self.W], axis=1)
        # x_i=np.concatenate([ [1], x_i])
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
    def __init__(self, n_classes, n_features, hidden_size,layers):
        # Initialize an MLP with a single hidden layer.
        self.w=[np.random.normal(0.1,0.1,[hidden_size,n_features]),np.random.normal(0.1,0.1,[n_classes,hidden_size])]  # still need to initialize values
        self.b=[np.zeros(hidden_size),np.zeros(n_classes)] #check later
        self.z=[np.zeros(n_features), np.zeros(hidden_size),np.zeros(n_classes)] #check later
        self.gradz=[np.zeros(n_features), np.zeros(hidden_size),np.zeros(n_classes)] #check later

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        pred_tot=np.zeros(X.shape[0])
        for n in np.arange(X.shape[0]):
            x_i=X[n]
            self.z[0] = x_i
            for j in range(1,len(self.z)):
                # print(self.w[j-1].shape)
                # print(self.z[j-1].T.shape)
                # a=self.w[j-1].dot(self.z[j-1].T)
                # b=np.einsum('i,j->ij', self.b[j-1],np.ones(self.z[j-1].shape[0]))
                # a=a.T
                # b=b.T
                # print(a.shape)
                # print(b.shape)
                # lin=a+b
                # lin = (self.w[j-1]).dot(self.z[j-1].T) + np.einsum('i,j->ij', self.b[j-1],np.ones(self.z[j-1].shape[0])) # Linear part of a layer
                # print(self.z[j].shape, lin.shape)
                # self.z[j] = np.where(lin>0,lin,0) # RELU of linear part
                # print(self.z[j].shape,lin.shape)
                # self.z[-1]=np.exp(self.z[-1])/np.sum(np.exp(self.z[-1]))
                lin=(self.w[j-1]).dot(self.z[j-1])+self.b[j-1] # Linear part
                self.z[j]=np.where(lin>0,lin,0) # RELU
            self.z[-1]=np.exp(self.z[-1])/np.sum(np.exp(self.z[-1]))
            pred = np.argmax(self.z[-1])
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
        # n=int(np.random.rand()*y.shape[0])
        for x_i,y_i in zip(X,y):
            self.z[0] = x_i
            for j in range(1,len(self.z)):
                lin = (self.w[j-1]).dot(self.z[j-1]) + self.b[j-1] # Linear part
                self.z[j] = np.where(lin>0,lin,0) # RELU
            print(np.exp(self.z[-1]))
            self.z[-1] = np.exp(self.z[-1])/np.sum(np.exp(self.z[-1]))
            
            e = np.zeros((self.z[-1]).shape[0])
            e[y_i] = 1
            self.gradz[-1] = self.z[-1] - e
            for k in range(len(self.z)-1,0,-1):
                gradw = np.outer(self.gradz[k],self.z[k-1])
                gradb = self.gradz[k]
                gradh = np.dot(self.w[k-1].T,self.gradz[k])
                self.gradz[k-1] = np.where(self.z[k-1]<=0,0,gradh)
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
