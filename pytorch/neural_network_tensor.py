# neural_network_tensor.py
# implementation of neural network with PyTorch tensor (tutorial from PyTorch)

import torch
from matplotlib import pyplot as plt

dtype = torch.FloatTensor

# N is batch size; D_in is input_dimensions;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in).type(dtype)
y = torch.randn(N, D_out).type(dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H).type(dtype)
w2 = torch.randn(H, D_out).type(dtype)

_t = []     # iteration
_loss= []   # loss

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1)            # matrix multiplication (dot product)
    h_relu = h.clamp(min=0) # negative values -> 0
    y_pred = h_relu.mm(w2)  # output

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print('%d: %s' % (t, loss))
    
    _t.append(t)
    _loss.append(loss)

    grad_y_pred = 2. * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0 
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

plt.plot(_t, _loss, lw=1, color='r')
plt.title('PyTorch Neural Network')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()
 
