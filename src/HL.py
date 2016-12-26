import sys
from random import random
from numpy import *
from functions import *


class HiddenLayer :
    
    def output(self, input=None):
        if input is not None:
            self.x = input
        
        linear_output = dot(self.x, self.W) + self.b
        return self.activation(linear_output)


    def __init__(self, input, n_input, n_output, W=None, b=None, rng=None, activation=tanh):
        
        if rng is None:
            rng = random.RandomState(1234)

        if W is None:
            a = 1. / n_input
            W = array(rng.uniform(low=-a, high=a, size=(n_input, n_output)))

        if b is None:
            b = zeros(n_output) 

        self.rng = rng
        self.x = input

        self.W = W
        self.b = b

        if activation == tanh:
            self.dactivation = dtanh

        elif activation == sigmoid:
            self.dactivation = dsigmoid

        elif activation == RELU:
            self.dactivation = dRELU

        else:
            raise ValueError('activation function not supported.')

        
        self.activation = activation
        


    def forward(self, input=None):
        return self.output(input=input)


    def backward(self, prev_layer, learning_rate=0.1, input=None, dropout=False, mask=None):
        if input is not None:
            self.x = input

        d_y = self.dactivation(prev_layer.x) * dot(prev_layer.d_y, prev_layer.W.T)

        if dropout == True:
            d_y *= mask

        self.W += learning_rate * dot(self.x.T, d_y)
        self.b += learning_rate * mean(d_y, axis=0)
        self.d_y = d_y


    def dropout(self, input, p, rng=None):
        if rng is None:
            rng = random.RandomState(123)

        mask = rng.binomial(size=input.shape, n=1, p=1-p)  # where p is the property of dropping
        return mask
                     

    def sample_h_given_v(self, input=None):
        if input is not None:
            self.x = input

        v_mean = self.output()
        hidden_sample = self.rng.binomial(size=v_mean.shape, n=1, p=v_mean)
        return hidden_sample


