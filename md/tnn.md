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