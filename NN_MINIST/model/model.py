# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import random
import os,re,sys
sys.path.append('../loader')
sys.path.append('../tools')
from loader import *
from tools import *

class NerualNet(object):
    def __init__(self,train_set,test_set,sizes,learning_rate,epoachs,minibatch_size):
        self.train_set = train_set
        self.test_set = test_set
        self.learning_rate = learning_rate
        self.epoachs = epoachs
        self.minibatch_size = minibatch_size
        self.sizes = sizes
        self.L2 = 0

        self.w_ = [np.random.randn(i,j) for i,j in zip(sizes[1:],sizes[:-1])]
        self.b_ = [np.random.randn(j,1) for j in sizes[1:]]
    def sigmoid(self,x):
        return 1.0/(1.0+np.exp(-x))

    def sigmoid_premire(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def feedforward(self,x):
        for w,b in zip(self.w_,self.b_):
            x = self.sigmoid(np.dot(w,x)+b)
        return x

    def softmax_premire(self,x,y):
        exp = [np.exp(i) for i in x]
        total   = sum(exp)
        softmax = [e/total for e in exp]
        delta   = np.zeros(y.shape)
        y_label = np.argmax(y)
        for j in range(len(exp)):
            if j==y_label:
                delta[j] = exp[j] * softmax[j] - 1
            else:
                delta[j] = exp[j] * softmax[j]

        return delta


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    # for training_set
    def backprop(self,x,y):
        delta_w = np.array([np.zeros(w.shape) for w in self.w_])
        delta_b = np.array([np.zeros(b.shape) for b in self.b_])

        zs = []
        activations = []
        activation = x
        activations.append(activation)

        for i in range(len(self.sizes)-1):
            z = np.dot(self.w_[i],activation) + self.b_[i]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.softmax_premire(activations[-1],y) * self.sigmoid_premire(zs[-1]) + self.L2
        
        delta_w[-1] = np.dot(delta,activations[-2].transpose())
        delta_b[-1] = delta

        for i in range(2,len(self.sizes)):
            delta =np.dot(self.w_[-i+1].transpose(),delta)*self.sigmoid_premire(zs[-i])
            delta_w[-i] = np.dot(delta,activations[-i-1].transpose())
            delta_b[-i] = delta

        return (delta_w,delta_b)

    def update_mini_batch(self,mini_batch):
        delta_w = [np.zeros(w.shape) for w in self.w_]
        delta_b = [np.zeros(b.shape) for b in self.b_]

        # print np.array(mini_batch).shape
        self.L2 = 0.5*0.000001*sum([np.sum(w) for w in self.w_])
        for x,y in mini_batch:
            # print "x.shape:",x.shape," y_label.shape", y.shape
            delta_w_train,delta_b_train = self.backprop(x,y)
            delta_b = [sub_delta_b+sub_delta_b_train for sub_delta_b,sub_delta_b_train in zip(delta_b,delta_b_train)]
            delta_w = [sub_delta_w+sub_delta_w_train for sub_delta_w,sub_delta_w_train in zip(delta_w,delta_w_train)]
        self.b_ = [sub_self_b_ - self.learning_rate * sub_delta_b/len(mini_batch) for sub_self_b_,sub_delta_b in zip(self.b_,delta_b)]
        self.w_ = [sub_self_w_ - self.learning_rate * sub_delta_w/len(mini_batch) for sub_self_w_,sub_delta_w in zip(self.w_,delta_w)]

    def SGD(self):
        n = len(train_set)
        for i in range(self.epoachs):
            random.shuffle(self.train_set)
            mini_batches = [self.train_set[j:j+self.minibatch_size] for j in range(0,n,self.minibatch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            print("Epoch: %d/%d Accuarcy: %f"%(i+1,self.epoachs,self.evaluate()))

    def evaluate(self):
        test_result = [(np.argmax(self.feedforward(x)),y[0]) for (x,y) in self.test_set]
        return sum([int(x == y) for (x, y) in test_result])/len(test_result)




if __name__ == "__main__":
    image_size = 28*28
    size = [image_size,40,10]
    train_set = load_samples(dataset='training_data')
    test_set = load_samples(dataset='testing_data')
    nerualnet = NerualNet(train_set,test_set,size,3,13,100)
    nerualnet.SGD()
