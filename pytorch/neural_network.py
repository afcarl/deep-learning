# neural_network.py
# implementation of neural network with Numpy (tutorial from PyTorch)

import numpy as np
import matplotlib.pyplot as plt

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

_t = []     # iteration
_loss = []  # loss

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.dot(w1)               # input sum of hidden layer
    h_relu = np.maximum(h, 0)   # activation of hidden layer
    y_pred = h_relu.dot(w2)     # output layer (prediction)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print('%d: %f' % (t, loss))

    _t.append(t)
    _loss.append(loss)

    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2. * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

# plot a line chart
plt.plot(_t, _loss, lw=1, color='r')
plt.title('Numpy Neural Network')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
