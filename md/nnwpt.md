# Neural networks with PyTorch

[Back](../README.md)

Deep learning networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from.

You can build one of these deep networks using only weight matrices as we did in the previous notebook, but in general it's very cumbersome and difficult to implement.

PyTorch has a nice module nn that provides a nice way to efficiently build large neural networks.

---

Now we're going to build a larger network that can solve a (formerly) difficult problem, identifying text in an image.

Here we'll use the MNIST dataset which consists of greyscale handwritten digits. Each image is 28x28 pixels, you can see a sample below

![numbers](../img/numbers.png)

---

```py

# Import necessary packages

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np
import torch

import helper

import matplotlib.pyplot as plt

```