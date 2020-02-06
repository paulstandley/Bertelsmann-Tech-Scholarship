# Neural networks with PyTorch

[Back](../README.md)

## Deep learning networks tend to be massive with dozens or hundreds of layers, that's where the term "deep" comes from.

You can build one of these deep networks using only weight matrices as we did in the previous notebook, but in general it's very cumbersome and difficult to implement.

PyTorch has a nice module nn that provides a nice way to efficiently build large neural networks.

---

Now we're going to build a larger network that can solve a (formerly) difficult problem, identifying text in an image.

Here we'll use the MNIST dataset which consists of greyscale handwritten digits. Each image is 28x28 pixels, you can see a sample below

![numbers](../img/numbers.png)

Our goal is to build a neural network that can take one of these images and predict the digit in the image.

First up, we need to get our dataset. This is provided through the torchvision package.

The code below will download the MNIST dataset, then create training and test datasets for us.

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

```py

### Run this cell

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

```

We have the training data loaded into trainloader and we make that an iterator with iter(trainloader). Later, we'll use this to loop through the dataset for training, like

```py

for image, label in trainloader:

  # do things with images and labels
  
```

You'll notice I created the trainloader with a batch size of 64, and shuffle=True.

The batch size is the number of images we get in one iteration from the data loader and pass through our network, often called a batch.

And shuffle=True tells it to shuffle the dataset every time we start going through the data loader again. But here I'm just grabbing the first batch so we can check out the data.

We can see below that images is just a tensor with size (64, 1, 28, 28). So, 64 images per batch, 1 color channel, and 28x28 images.

```py

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

```

This is what one of the images looks like.

```py

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')

```
