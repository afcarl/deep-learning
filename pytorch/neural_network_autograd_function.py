# neural_network_autograd_function.py
# implementation of neural network with PyTorch Autograd Function
# (tutorial from PyTorch)

import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

class ReLU(torch.autograd.Function):
    def forward(self, input):
        self.save_for_backward(input)
        return input.clamp(min=0)

    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

tensor = torch.FloatTensor

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random Tensors to hold input and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in).type(tensor), requires_grad=False)
y = Variable(torch.randn(N, D_out).type(tensor), requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
w1 = Variable(torch.randn(D_in, H).type(tensor), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(tensor), requires_grad=True)

_t = []     # iteration
_loss = []  # loss

learning_rate = 1e-6
for t in range(500):
    # Construct an instance of our ReLU class to use in out network
    relu = ReLU()

    # Forward pass: compute predicted y using operations on Variables; we
    # compute ReLU using out custom autograd operation.
    y_pred = relu(x.mm(w1)).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print('%d: %f' % (t, loss.data[0]))

    _t.append(t)
    _loss.append(loss.data[0])

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients after updating weights
    w1.grad.data.zero_()
    w2.grad.data.zero_()

plt.plot(_t, _loss, lw=1, color='b')
plt.title('PyTorch Autograd Function Neural Network')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

