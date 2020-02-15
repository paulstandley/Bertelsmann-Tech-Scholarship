# Loading Image Data

## [Back](../README.md)

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

The easiest way to load image data is with datasets.ImageFolder from torchvision [(documentation)](http://pytorch.org/docs/master/torchvision/datasets.html#imagefolder).

In general you'll use ImageFolder like so:

```py

dataset = datasets.ImageFolder('path/to/data', transform=transform)

```

where __'path/to/data'__ is the file path to the data directory and transform is a sequence of processing steps built with the transforms module from torchvision.

ImageFolder expects the files and directories to be constructed like so:

```py

root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png

```

where each class has it's own directory __(cat and dog)__ for the images.

The images are then labeled with the class taken from the directory name. So here, the image __123.png__ would be loaded with the class label cat.

You can download the dataset already structured like this from [here](https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip).

I've also split it into a training set and test set.

---

### Transforms

When you load in the data with ImageFolder, you'll need to define some transforms.

For example, the images are different sizes but we'll need them to all be the same size for training.

You can either resize them with transforms.Resize() or crop with

transforms.CenterCrop(),

transforms.RandomResizedCrop(), etc.

We'll also need to convert the images to PyTorch tensors with transforms.ToTensor().

Typically you'll combine these transforms into a pipeline with transforms.Compose(),

which accepts a list of transforms and runs them in sequence.

It looks something like this to scale, then crop, then convert to a tensor

```py

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

```

There are plenty of transforms available, I'll cover more in a bit and you can read through the [documentation](http://pytorch.org/docs/master/torchvision/transforms.html).

### Data Loaders

With the ImageFolder loaded, you have to pass it to a DataLoader.

The DataLoader takes a dataset (such as you would get from ImageFolder) and returns batches of images and the corresponding labels.

You can set various parameters like the batch size and if the data is shuffled after each epoch.

```py

dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

```

Here dataloader is a [generator](https://jeffknupp.com/blog/2013/04/07/improve-your-python-yield-and-generators-explained/). To get data out of it, you need to loop through it or convert it to an iterator and call next().

```py

# Looping through it, get a batch on each loop
for images, labels in dataloader:
    pass

# Get one batch
images, labels = next(iter(dataloader))

```

Exercise: Load images from the Cat_Dog_data/train folder, define a few transforms, then build the dataloader.

```py

data_dir = 'Cat_Dog_data/train'

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)

```

If you loaded the data correctly, you should see something like this (your image will be different):

![cat](../img/cat_cropped.png)

---

### Data Augmentation

A common strategy for training neural networks is to introduce randomness in the input data itself.

For example, you can randomly rotate, mirror, scale, and/or crop your images during training.

This will help your network generalize as it's seeing the same images but in different locations, with different sizes, in different orientations, etc.

To randomly rotate, scale and crop, then flip your images you would define your transforms like this:

```py

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])]))

```

You'll also typically want to normalize images with transforms.Normalize.

You pass in a list of means and list of standard deviations, then the color channels are normalized like so

```py

input[channel] = (input[channel] - mean[channel]) / std[channel]

```

Subtracting mean centers the data around zero and dividing by std squishes the values to be between -1 and 1.

Normalizing helps keep the network weights near zero which in turn makes backpropagation more stable.

Without normalization, networks will tend to fail to learn.

You can find a list of all the available transforms here.

When you're testing however, you'll want to use images that aren't altered other than normalizing.

So, for validation/test images, you'll typically just resize and crop.

Exercise: Define transforms for training data and testing data below.

Leave off normalization for now.

```py

data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()]) 

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# change this to the trainloader or testloader 
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)

```

Your transformed images should look something like this.

#### Training examples:

![trainning](../img/train_examples.png)

#### Testing examples:

![test](../img/test_examples.png)

At this point you should be able to load data for training and testing.

Now, you should try building a network that can classify cats vs dogs.

This is quite a bit more complicated than before with the MNIST and Fashion-MNIST datasets.

To be honest, you probably won't get it to work with a fully-connected network, no matter how deep.

These images have three color channels and at a higher resolution (so far you've seen 28x28 images which are tiny).

In the next part, I'll show you how to use a pre-trained network to build a model that can actually solve this problem.

### Optional TODO: Attempt to build a network to classify cats vs dogs from this dataset

[Back](../README.md)
