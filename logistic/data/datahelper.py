import os,sys
import numpy as np

class datahelper:
    def __init__(self, posfile, negfile, testfile):
        self.posfile = posfile
        self.negfile = negfile
        self.testfile = testfile
        with open(self.posfile,'r') as f:
            for line in f:
                num = [int(i.split(':')[0]) for i in line.split()[:-1]]
            self.max = max(num)

    def get_dataset(self, file, set):
        with open(file,'r') as f:
            for line in f:
                line_split = line.split()
                num = [(int(i.split(':')[0]),int(i.split(':')[1])) for i in line_split[:-1]]
                input = np.zeros(self.max)
                for i in num:
                    if i[0]<=self.max:
                        input[i[0]-1] = i[1]
                set.append((input,0) if line_split[-1] == "neg" else (input,1))

    def trainSet(self):
        train_set = []
        self.get_dataset(self.posfile, train_set)
        self.get_dataset(self.negfile, train_set)
        return train_set

    def testSet(self):
        test_set = []
        self.get_dataset(self.testfile, test_set)
        return test_set

def dataSetGenerate():
    dir = "/home/dreamer/codes/algorithm_code/logistic/data/data/books"
    negfile = dir+'/negative.review'
    posfile = dir+'/positive.review'
    testfile = dir+'/unlabeled.review'
    data = datahelper(posfile, negfile, testfile)
    train_set = data.trainSet()
    test_set = data.testSet()
    return (data.max,train_set,test_set)

if __name__ == "__main__":
    dataSetGenerate()
