# neural_network_nn.py
# implementation of neural network with PyTorch NN (tutorial from PyTorch)

import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create ranom Tensors to hold inputs and outputs, and wrap them in Variables.
x = Variable(torch.randn(N, D_in))
y = Variable(torch.randn(N, D_out), requires_grad=False)

# Use the nn package to define out model as a sequence of layers. nn.Sequential
# is a Module which contains other Modules, and applies them in sequence to
# produce its output. Each linear Module computes output from input using a
# linear function, and holds internal Variables for its weights and bias.
model = torch.nn.Sequential(
            torch.nn.Linear(D_in, H),
            torch.nn.ReLU(),
            torch.nn.Linear(H, D_out),
        )

# The nn package also contains definitions of popular loss functions; in this
# case we will use Mean Squared Error (MSE) as our loss function.
loss_fn = torch.nn.MSELoss(size_average=False)

_t = []     # iteration
_loss = []  # loss

learning_rate = 1e-4
for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module
    # objects override the __call__ operator so you can call them like
    # functions. When doing so you pass a Variable of input data to the
    # Module and it produces a Variable of output data.
    y_pred = model(x)

    # Compute and print loss. We pass Variables containing the predicted and
    # true values of y, and the loss function returns a Variable containing the
    # loss.
    loss = loss_fn(y_pred, y)
    print('%d: %f' % (t, loss.data[0]))

    _t.append(t)
    _loss.append(loss.data[0])

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the
    # learnable parameters of the model. Internally, the parameters of each
    # Module are stored in Variables with requires_grad=True, so this call will
    # compute gradients for all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Variable,
    # so we can access its data and gradients like we did before.
    for param in model.parameters():
        param.data -= learning_rate * param.grad.data

plt.plot(_t, _loss, lw=1, color='r')
plt.title('PyTorch NN Neural Network')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()

