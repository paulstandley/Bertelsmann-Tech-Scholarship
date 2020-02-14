# Loading Image Data

So far we've been working with fairly artificial datasets that you wouldn't typically be using in real projects.

Instead, you'll likely be dealing with full-sized images like you'd get from smart phone cameras.

In this notebook, we'll look at how to load images and use them to train neural networks.

We'll be using a [dataset of cat and dog](https://www.kaggle.com/c/dogs-vs-cats) photos available from Kaggle. Here are a couple example images

![cats and dogs](../img/dog_cat.png)

We'll use this dataset to train a neural network that can differentiate between cats and dogs.

These days it doesn't seem like a big accomplishment, but five years ago it was a serious challenge for computer vision systems.

```py

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms

import helper

```
