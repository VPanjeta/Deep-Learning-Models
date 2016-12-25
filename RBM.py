import sys
from random import random
from numpy import *
from functions import *

class RBMachine :
    
    def div(self, learning_rate=0.1, k=1, input=None):
        if input is not None:
            self.input = input

        ph_mean, ph_sample = self.sample_h_given_v(self.input)

        chain_start = ph_sample

        for step in xrange(k):
            if step == 0:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(chain_start)
            else:
                nv_means, nv_samples, nh_means, nh_samples = self.gibbs_hvh(nh_samples)

        # chain_end = nv_samples


        self.W += learning_rate * (dot(self.input.T, ph_mean)
                        - dot(nv_samples.T, nh_means))
        self.visible_unit_bias += learning_rate * mean(self.input - nv_samples, axis=0)
        self.hidden_unit_bias += learning_rate * mean(ph_mean - nh_means, axis=0)

    def __init__(self, input=None, visible_units=2, hidden_units=3, W=None, hidden_unit_bias=None, visible_unit_bias=None, rng=None):
       
        self.visible_units = visible_units  # Units in input layer
        self.hidden_units = hidden_units    # units in hidden layer

        if hidden_unit_bias is None:
            hidden_unit_bias = zeros(hidden_units)  # Hidden units' bias is 0

        if visible_unit_bias is None:
            visible_unit_bias = zeros(visible_units)  # Visible units' bias is 0

        if rng is None:
            rng = random.RandomState(1234)


        if W is None:
            a = 1. / visible_units
            initial_W = array(rng.uniform(  
                low=-a,
                high=a,
                size=(visible_units, hidden_units)))

            W = initial_W


        self.rng = rng
        self.input = input
        self.W = W
        self.hidden_unit_bias = hidden_unit_bias
        self.visible_unit_bias = visible_unit_bias


    def sample_h_given_v(self, v0_sample):
        hidden_mean = self.propup(v0_sample)
        hidden1_sample = self.rng.binomial(size=hidden_mean.shape, n=1, p=hidden_mean)

        return [hidden_mean, hidden1_sample]


    def sample_v_given_h(self, h0_sample):
        visible1_mean = self.propdown(h0_sample)
        visible1_sample = self.rng.binomial(size=visible1_mean.shape, n=1, p=visible1_mean)
        
        return [visible1_mean, visible1_sample]

    def propup(self, v):
        pre_sigmoid_activation = dot(v, self.W) + self.hidden_unit_bias
        return sigmoid(pre_sigmoid_activation)

    def propdown(self, h):
        pre_sigmoid_activation = dot(h, self.W.T) + self.visible_unit_bias
        return sigmoid(pre_sigmoid_activation)


    def gibbs_hvh(self, h0_sample):
        visible1_mean, visible1_sample = self.sample_v_given_h(h0_sample)
        hidden1_mean, hidden1_sample = self.sample_h_given_v(visible1_sample)

        return [visible1_mean, visible1_sample,
                hidden1_mean, hidden1_sample]
    

    def get_executeion_cross_entropy(self):
        pre_sigmoid_activation_h = dot(self.input, self.W) + self.hidden_unit_bias
        sigmoid_activation_h = sigmoid(pre_sigmoid_activation_h)
        
        pre_sigmoid_activation_v = dot(sigmoid_activation_h, self.W.T) + self.visible_unit_bias
        sigmoid_activation_v = sigmoid(pre_sigmoid_activation_v)

        cross_entropy =  - mean(
            sum(self.input * log(sigmoid_activation_v) +
            (1 - self.input) * log(1 - sigmoid_activation_v),
                      axis=1))
        
        return cross_entropy

    def execute(self, v):
        h = sigmoid(dot(v, self.W) + self.hidden_unit_bias)
        executeed_v = sigmoid(dot(h, self.W.T) + self.visible_unit_bias)
        return executeed_v





def execute_machine(learning_rate, k, training_period):
    TData = array([
    					[1,1,1,0,0,0],
                        [1,0,1,0,0,0],
                        [1,1,1,0,0,0],
                        [0,0,1,1,1,0],
                        [0,0,1,1,0,0],
                        [0,0,1,1,1,0]
                ])


    rng = random.RandomState(123)

    # construct RBMachine object
    rbm = RBMachine(input=TData, visible_units=6, hidden_units=2, rng=rng)

    # Train RBM
    for epoch in range(training_period):
        rbm.div(learning_rate=learning_rate, k=k)


    # test
    data = array([
    				 [1, 0, 1, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0],
                     [1, 0, 1, 0, 0, 0],
                ])

    print "\nFor data set of: "
    for arr in data:
    	print arr
    print "\nReconstruction is: "
    for arr in rbm.execute(data):
    	print arr



if __name__ == "__main__":
    execute_machine(learning_rate=.1,k=1,training_period=1000)