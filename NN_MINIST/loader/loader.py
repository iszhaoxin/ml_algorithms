# -*- coding: utf-8 -*-
import os, struct
from array import array as pyarray
import numpy as np
np.set_printoptions(threshold='nan')
from numpy import append, array, int8, uint8, zeros
import random

def load_mnist(dataset="training_data", digits=np.arange(10), path="."):
    if dataset == "training_data":
        fname_image = '../database/train-images-idx3-ubyte/data'
        fname_label = '../database/train-labels-idx1-ubyte/data'
    elif dataset == "testing_data":
        fname_image = '../database/t10k-images-idx3-ubyte/data'
        fname_label = '../database/t10k-labels-idx1-ubyte/data'
    else:
        raise ValueError("dataset must be 'training_data' or 'testing_data'")

    flbl = open(fname_label, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_image, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def load_samples(dataset="training_data"):
    image,label = load_mnist(dataset)
    #print(image[0].shape, image.shape)   # (28, 28) (60000, 28, 28)
    #print(label[0].shape, label.shape)   # (1,) (60000, 1)
    #print(label[0])   # 5

    # 把28*28二维数据转为一维数据
    X = [np.reshape(x,(28*28, 1)) for x in image]
    X = [x/255.0 for x in X]   # 灰度值范围(0-255)，转换为(0-1)
    #print(X.shape)

    # 5 -> [0,0,0,0,0,1.0,0,0,0]      1 -> [0,1.0,0,0,0,0,0,0,0]
    def vectorized_Y(y):
        e = np.zeros((10, 1))
        e[y] = 1.0
        return e
    # 把Y值转换为神经网络的输出格式
    if dataset == "training_data":
        Y = [vectorized_Y(y) for y in label]
        pair = list(zip(X, Y))
        return pair
    elif dataset == 'testing_data':
        pair = list(zip(X, label))
        return pair
    else:
        print('Something wrong')


if __name__ == '__main__':
    training_data = list(load_mnist())
    random.shuffle(training_data)
    # training_data = [[1,1],[2,2],[3,3],[4,4],[5,5]]
    # random.shuffle(training_data)
    # print(len(training_data[1][4]))
        # break
    # print training_data
