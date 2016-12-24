from numpy import *
seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + exp(-x))

def dsigmoid(x):
    return x * (1. - x)
