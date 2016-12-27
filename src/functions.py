from numpy import *
seterr(all='ignore')

def sigmoid(x):
    return 1. / (1 + exp(-x))

def dsigmoid(x):
    return x * (1. - x)

def gaussian(x, mean=0.0, scale=1.0):
    s = 2 * power(scale, 2)
    e = exp( - power((x - mean), 2) / s )
    return e / square(pi * s)

def tanh(x):
    return tanh(x)

def dtanh(x):
    return 1. - x * x

def RELU(x):
    return x * (x > 0)

def dRELU(x):
    return 1. * (x > 0)

def softmax(x):
    e = exp(x - max(x))  # prevent overflow
    if e.ndim == 1:
        return e / sum(e, axis=0)
    else:  
        return e / array([sum(e, axis=1)]).T
