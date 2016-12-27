import sys
from numpy import *
from functions import *


class LogisticRegression :
    def train(self, learning_rate=0.1, input=None, L2_reg=0.00):        
        if input is not None:
            self.x = input


    def __init__(self, input, label, n_input, n_output):
        self.x = input
        self.y = label

        self.W = zeros((n_input, n_output))  # initialize W 0
        self.b = zeros(n_output)  # initialize bias 0

        p_y_given_x = self.output(self.x)
        d_y = self.y - p_y_given_x

        self.W += learning_rate * dot(self.x.T, d_y) - learning_rate * L2_reg * self.W
        self.b += learning_rate * mean(d_y, axis=0)
        self.d_y = d_y

    def output(self, x):
        # return sigmoid(dot(x, self.W) + self.b)
        return softmax(dot(x, self.W) + self.b)

    def predict(self, x):
        return self.output(x)


    def neg_log_likelihood(self):
        # sigmoid_activation = sigmoid(dot(self.x, self.W) + self.b)
        sigmoid_activation = softmax(dot(self.x, self.W) + self.b)
        cross_entropy = - mean(sum(self.y * log(sigmoid_activation) + (1 - self.y) * log(1 - sigmoid_activation), axis=1))
        return cross_entropy


def test_learning_rate(learning_rate=0.1, n_period=500):
    rng = random.RandomState(123)
    # data to train
    d = 2
    N = 10
    x1 = rng.randn(N, d) + array([0, 0])
    x2 = rng.randn(N, d) + array([20, 10])
    y1 = [[1, 0] for i in xrange(N)]
    y2 = [[0, 1] for i in xrange(N)]
    x = r_[x1.astype(int), x2.astype(int)]
    y = r_[y1, y2]


    # construct LogisticRegression
    classifier = LogisticRegression(input=x, label=y, n_input=d, n_output=2)

    # train
    for epoch in xrange(n_period):
        classifier.train(learning_rate=learning_rate)
        learning_rate *= 0.995


    # test
    result = classifier.predict(x)
    for i in xrange(N):
        print result[i]
    print
    for i in xrange(N):
        print result[N+i]



if __name__ == "__main__":
    test_learning_rate()
