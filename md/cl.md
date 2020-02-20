# Convolutional Layer

## [Back](../README.md)

In this notebook, we visualize four filtered outputs (a.k.a. activation maps) of a convolutional layer.

In this example, we are defining four filters that are applied to an input image by initializing the weights of a convolutional layer, but a trained CNN will learn the values of these weights.

![cl](../img/cl.gif)

---

### Import the image

```py

import cv2
import matplotlib.pyplot as plt
%matplotlib inline

# TODO: Feel free to try out your own images here by changing img_path
# to a file path to another image on your computer!
img_path = 'data/udacity_sdc.png'

# load color image 
bgr_img = cv2.imread(img_path)
# convert to grayscale
gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

# normalize, rescale entries to lie in [0,1]
gray_img = gray_img.astype("float32")/255

# plot image
plt.imshow(gray_img, cmap='gray')
plt.show()

```

### Define and visualize the filters

```py

import numpy as np

## TODO: Feel free to modify the numbers here, to try out another filter!
filter_vals = np.array([[-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1], [-1, -1, 1, 1]])

print('Filter shape: ', filter_vals.shape)

```

```py

# Defining four different filters, 
# all of which are linear combinations of the `filter_vals` defined above

# define four filters
filter_1 = filter_vals
filter_2 = -filter_1
filter_3 = filter_1.T
filter_4 = -filter_3
filters = np.array([filter_1, filter_2, filter_3, filter_4])

# For an example, print out the values of filter 1
print('Filter 1: \n', filter_1)

```

```py

# visualize all four filters
fig = plt.figure(figsize=(10, 5))
for i in range(4):
    ax = fig.add_subplot(1, 4, i+1, xticks=[], yticks=[])
    ax.imshow(filters[i], cmap='gray')
    ax.set_title('Filter %s' % str(i+1))
    width, height = filters[i].shape
    for x in range(width):
        for y in range(height):
            ax.annotate(str(filters[i][x][y]), xy=(y,x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if filters[i][x][y]<0 else 'black')

```

### Define a convolutional layer

The various layers that make up any neural network are documented, here. For a convolutional neural network, we'll start by defining a:

__.__ Convolutional layer

Initialize a single convolutional layer so that it contains all your created filters.

Note that you are not training this network;

you are initializing the weights in a convolutional layer so that you can visualize what happens after a forward pass through this network!

### __init__ and forward

To define a neural network in PyTorch, you define the layers of a model in the function \_\_init\_\_ and define the forward behavior

of a network that applyies those initialized layers to an input (x) in the function forward.

In PyTorch we convert all inputs into the Tensor datatype, which is similar to a list data type in Python.

Below, I define the structure of a class called Net that has a convolutional layer that can contain four 4x4 grayscale filters.
