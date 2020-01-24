import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    # get the exp of L
    expl = np.exp(L)
    # sum the exp of l
    sum_expl = sum(expl)
    ret_val = []
    # loop to get return values
    for i in expl:
        ret_val.append(i * 1.0 / sum_expl)
    return ret_val
    # Note: The function np.divide can also be used here, as follows:
    # def softmax(L):
    #     expL = np.exp(L)
    #     return np.divide (expL, expL.sum())