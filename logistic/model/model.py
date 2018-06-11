import os, sys
sys.path.append("../data")
from datahelper import *
import numpy as np
from mylib.texthelper import *
import random
import math

def exp(a):
    if isinstance(a, np.ndarray):
        return np.array([math.exp(i) for i in a])
    elif isinstance(a, np.float64):
        return np.float64(math.exp(a))

class logistic(object):
    def __init__(self,train_set,test_set,sizes,learning_rate,epoachs,minibatch_size, regluar_rate):
        self.train_set = train_set
        self.test_set = test_set
        self.learning_rate = learning_rate
        self.epoachs = epoachs
        self.minibatch_size = minibatch_size
        self.w_ = np.zeros(size)
        self.b_ = np.random.randn()
        self.regluar_rate = regluar_rate
    def linear(self, x):
        return np.dot(x,self.w_) + self.b_

    def sigmoid(self,x):
        return 1/(1+exp(-x))

    def sigmoid_premire(self,l):
        return self.sigmoid(l)*(1-self.sigmoid(l))

    def feedforward(self,x):
        return 1 if self.sigmoid(self.linear(x))>0.5 else 0

    def backprop(self, x, y_label):
        epro = 0.00001
        l = self.linear(x)
        y = self.sigmoid(l)
        l2 = np.dot(self.w_, self.w_) * regluar_rate
        sig_line = self.sigmoid_premire(l)
        loss_sig = -1*(y_label/(y+epro)*sig_line+(1-y_label)/(1-y+epro)*(-sig_line)) + l2
        delta_w = np.dot(loss_sig,x)/self.minibatch_size*self.learning_rate
        delta_b = sum(loss_sig)/self.minibatch_size*self.learning_rate
        return (delta_w,delta_b)

    def update_mini_batch(self,mini_batch):
        x = np.array([i[0] for i in mini_batch])
        y_label = np.array([i[1] for i in mini_batch])
        delta_w,delta_b = self.backprop(x,y_label)
        self.b_ -= delta_b
        self.w_ -= delta_w

    def SGD(self):
        n = len(self.train_set)
        for i in range(self.epoachs):
            random.shuffle(self.train_set)
            mini_batches = [self.train_set[j:j+self.minibatch_size] for j in range(0,n,self.minibatch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            print("Epoch: %d/%d Accuarcy: %f"%(i+1,self.epoachs,self.evaluate()))

    def evaluate(self):
        test_result = [(self.feedforward(x)==y) for (x,y) in self.test_set]
        return sum(test_result)/len(test_result)

if __name__ == "__main__":
    mini_batch = 8
    learning_rate = 0.05
    epochs = 10
    # regluar_rate = 0.000001
    regluar_rate = 0
    size,train_set,test_set = dataSetGenerate()
    model = logistic(train_set,test_set,size, learning_rate, epochs, mini_batch, regluar_rate)
    model.SGD()
