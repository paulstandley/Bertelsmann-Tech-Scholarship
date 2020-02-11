# Inference and Validation

## [Back](../README.md)

Now that you have a trained network, you can use it for making predictions. This is typically called inference, a term borrowed from statistics.

However, neural networks have a tendency to perform too well on the training data and aren't able to generalize to data that hasn't been seen before.

This is called overfitting and it impairs inference performance. To test for overfitting while training, we measure the performance on data not in the training set called the validation set.

We avoid overfitting through regularization such as dropout while monitoring the validation performance during training. In this notebook, I'll show you how to do this in PyTorch.

As usual, let's start by loading the dataset through torchvision. You'll learn more about torchvision and loading data in a later part.

This time we'll be taking advantage of the test set which you can get by 

setting train=False here:

```py

testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

```

The test set contains images just like the training set. Typically you'll see 10-20% of the original dataset held out for testing and validation with the rest being used for training.

```py

import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
# Download and load the training data
trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# Create a model

from torch import nn, optim
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

```

The goal of validation is to measure the model's performance on data that isn't part of the training set.

Performance here is up to the developer to define though. Typically this is just accuracy, the percentage of classes the network predicted correctly.

Other options are precision and recall and top-5 error rate. We'll focus on accuracy here.

First I'll do a forward pass with one batch from the test set.

```py

model = Classifier()

images, labels = next(iter(testloader))
# Get the class probabilities
ps = torch.exp(model(images))
# Make sure the shape is appropriate, we should get 10 class probabilities for 64 examples
print(ps.shape)

```

With the probabilities, we can get the most likely class using the ps.topk method. This returns the  𝑘  highest values.

Since we just want the most likely class, we can use ps.topk(1). This returns a tuple of the top- 𝑘  values and the top- 𝑘  indices.

If the highest value is the fifth element, we'll get back 4 as the index.

```py

top_p, top_class = ps.topk(1, dim=1)
# Look at the most likely classes for the first 10 examples
print(top_class[:10,:])

```

Now we can check if the predicted classes match the labels. This is simple to do by equating top_class and labels, but we have to be careful of the shapes.

Here top_class is a 2D tensor with shape (64, 1) while labels is 1D with shape (64).

To get the equality to work out the way we want, top_class and labels must have the same shape.

### If we do

```py

equals = top_class == labels


```

equals will have shape (64, 64), try it yourself.

What it's doing is comparing the one element in each row of top_class with each element in labels which returns 64 True/False boolean values for each row.

```py

equals = top_class == labels.view(*top_class.shape)

```

Now we need to calculate the percentage of correct predictions. equals has binary values, either 0 or 1.

This means that if we just sum up all the values and divide by the number of values, we get the percentage of correct predictions.

This is the same operation as taking the mean, so we can get the accuracy with a call to torch.mean.

If only it was that simple. If you try torch.mean(equals), you'll get an error

RuntimeError: mean is not implemented for type torch.ByteTensor

This happens because equals has type torch.ByteTensor but torch.mean isn't implement for tensors with that type.

So we'll need to convert equals to a float tensor. Note that when we take torch.mean it returns a scalar tensor, to get the actual value as a float we'll need to do accuracy.item().

```py

model = Classifier()
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

epochs = 30
steps = 0

train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        
        optimizer.zero_grad()
        
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    else:
        test_loss = 0
        accuracy = 0
        
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for images, labels in testloader:
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))
                
        train_losses.append(running_loss/len(trainloader))
        test_losses.append(test_loss/len(testloader))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False

```

![tlvl](../img/tlvl.png)

---
