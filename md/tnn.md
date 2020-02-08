# Training Neural Networks

[Back](README.md)

The network we built in the previous part isn't so smart, it doesn't know anything about our handwritten digits. Neural networks with non-linear activations work like universal function approximators.

There is some function that maps your input to the output. For example, images of handwritten digits to class probabilities. The power of neural networks is that we can train them to approximate this function, and basically any function given enough data and compute time.

![fuction](../img/fa.png)

At first the network is naive, it doesn't know the function mapping the inputs to the outputs.

We train the network by showing it examples of real data, then adjusting the network parameters such that it approximates this function.

To find these parameters, we need to know how poorly the network is predicting the real outputs.

For this we calculate a loss function (also called the cost), a measure of our prediction error. For example, the mean squared loss is often used in regression and binary classification problems

By minimizing this loss with respect to the network parameters, we can find configurations where the loss is at a minimum and the network is able to predict the correct labels with high accuracy.

We find this minimum using a process called gradient descent. The gradient is the slope of the loss function and points in the direction of fastest change.

To get to the minimum in the least amount of time, we then want to follow the gradient (downwards). You can think of this like descending a mountain by following the steepest slope to the base.

![gard dis](../img/grde.png)

---

## Backpropagation

For single layer networks, gradient descent is straightforward to implement. However, it's more complicated for deeper, multilayer neural networks like the one we've built.

Complicated enough that it took about 30 years before researchers figured out how to train multilayer networks.

Training multilayer networks is done through backpropagation which is really just an application of the chain rule from calculus.

It's easiest to understand if we convert a two layer network into a graph representation

![fp bp](../img/fpbp.png)

In the forward pass through the network, our data and operations go from bottom to top here.

We pass the input  𝑥  through a linear transformation  𝐿1  with weights  𝑊1  and biases  𝑏1 . The output then goes through the sigmoid operation  𝑆  and another linear transformation  𝐿2 . Finally we calculate the loss  ℓ .

We use the loss as a measure of how bad the network's predictions are. The goal then is to adjust the weights and biases to minimize the loss.

To train the weights with gradient descent, we propagate the gradient of the loss backwards through the network.

Each operation has some gradient between the inputs and outputs. As we send the gradients backwards, we multiply the incoming gradient with the gradient for the operation.

Mathematically, this is really just calculating the gradient of the loss with respect to the weights using the chain rule.

---

### Losses in PyTorch

Let's start by seeing how we calculate the loss with PyTorch. Through the nn module, PyTorch provides losses such as the cross-entropy loss (nn.CrossEntropyLoss).

You'll usually see the loss assigned to criterion. As noted in the last part, with a classification problem such as MNIST, we're using the softmax function to predict class probabilities. With a softmax output, you want to use cross-entropy as the loss.

To actually calculate the loss, you first define the criterion then pass in the output of your network and the correct labels.

Something really important to note here. Looking at the documentation for [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss),

This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class.

The input is expected to contain scores for each class.

This means we need to pass in the raw output of our network into the loss, not the output of the softmax function. This raw output is usually called the logits or scores.

We use the logits because softmax gives you probabilities which will often be very close to zero or one but floating-point numbers can't accurately represent values near zero or one [(read more here)](https://docs.python.org/3/tutorial/floatingpoint.html). It's usually best to avoid doing calculations with probabilities, typically we use log-probabilities.

```py

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                              ])
# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

```

```py

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10))

# Define the loss
criterion = nn.CrossEntropyLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our logits
logits = model(images)
# Calculate the loss with the logits and the labels
loss = criterion(logits, labels)

print(loss)

```

In my experience it's more convenient to build the model with a log-softmax output using nn.LogSoftmax or F.log_softmax [(documentation)](https://pytorch.org/docs/stable/nn.html#torch.nn.LogSoftmax). Then you can get the actual probabilites by taking the exponential torch.exp(output). With a log-softmax output, you want to use the negative log likelihood loss, nn.NLLLoss [(documentation)](https://pytorch.org/docs/stable/nn.html#torch.nn.NLLLoss).

Exercise: Build a model that returns the log-softmax as the output and calculate the loss using the negative log likelihood loss.

```py

## Solution

# Build a feed-forward network
model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()

# Get our data
images, labels = next(iter(trainloader))
# Flatten images
images = images.view(images.shape[0], -1)

# Forward pass, get our log-probabilities
logps = model(images)
# Calculate the loss with the logps and the labels
loss = criterion(logps, labels)

print(loss)

```

### Autograd

Now that we know how to calculate a loss, how do we use it to perform backpropagation? Torch provides a module, autograd, for automatically calculating the gradients of tensors.

We can use it to calculate the gradients of all our parameters with respect to the loss. Autograd works by keeping track of operations performed on tensors, then going backwards through those operations, calculating gradients along the way.

To make sure PyTorch keeps track of operations on a tensor and calculates the gradients, you need to set requires_grad = True on a tensor. You can do this at creation with the requires_grad keyword, or at any time with x.requires_grad_(True).

You can turn off gradients for a block of code with the torch.no_grad() content:

```py

x = torch.zeros(1, requires_grad=True)
>>> with torch.no_grad():
...     y = x * 2
>>> y.requires_grad
False

```

Also, you can turn on or off gradients altogether with torch.set_grad_enabled(True|False).

The gradients are computed with respect to some variable z with z.backward(). This does a backward pass through the operations that created z

```py

x = torch.randn(2,2, requires_grad=True)
print(x)

tensor([[ 0.7652, -1.4550],
        [-1.2232,  0.1810]])


y = x**2
print(y)

tensor([[ 0.5856,  2.1170],
        [ 1.4962,  0.0328]])

# Below we can see the operation that created y, a power operation PowBackward0

## grad_fn shows the function that generated this variable
print(y.grad_fn)

<PowBackward0 object at 0x10b508b70>

"""The autograd module keeps track of these operations and knows how to calculate the gradient for each one.

In this way, it's able to calculate the gradients for a chain of operations, with respect to any one tensor.

Let's reduce the tensor y to a scalar value, the mean."""

z = y.mean()
print(z)

None

"""To calculate the gradients, you need to run the .backward method on a Variable, z for example.

This will calculate the gradient for z with respect to x

∂𝑧∂𝑥=∂∂𝑥[1𝑛∑𝑖𝑛𝑥2𝑖]=𝑥2"""

z.backward()
print(x.grad)
print(x/2)

tensor([[ 0.3826, -0.7275],
        [-0.6116,  0.0905]])
tensor([[ 0.3826, -0.7275],
        [-0.6116,  0.0905]])

```

These gradients calculations are particularly useful for neural networks. For training we need the gradients of the weights with respect to the cost.

With PyTorch, we run data forward through the network to calculate the loss, then, go backwards to calculate the gradients with respect to the loss.

Once we have the gradients we can make a gradient descent step.

---
