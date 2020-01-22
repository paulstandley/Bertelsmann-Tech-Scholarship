# Coding-Perceptron-Algorithm

## [Back](../README.md)

---

![points](../img/points.png)

---

### Coding the Perceptron Algorithm

Time to code! In this quiz,

you'll have the chance to implement the perceptron algorithm

to separate the following data (given in the file __data.csv__)

#### Recall that the perceptron step works as follows. For a point with coordinates

(p,q)(p,q), label yy, and prediction given by the equation

\hat{y} = step(w_1x_1 + w_2x_2 + b) y ^ =step(w 1 x 1 +w 2 x 2 +b):

__1__ If the point is correctly classified, do nothing.

__2__ If the point is classified positive, but it has a negative label, subtract \alpha p, \alpha q,αp,αq, and \alphaα from w_1, w_2,w 1 ,w 2 , and bb respectively.

__3__ If the point is classified negative, but it has a positive label, add \alpha p, \alpha q,αp,αq, and \alphaα to w_1, w_2,w 1 ,w 2 , and bb respectively.
