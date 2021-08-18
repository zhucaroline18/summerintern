#### importing data stuff i don't understand

import numpy as np
#import pandas as pd
from matplotlib import pyplot as plt

#### importing data stuff i don't understand

def init_params():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1)-0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1)-0.5
    
    return W1, b1, W2, b2

def ReLu(z):
    return np.maximum(0,z)

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

def forward_prop(W1, b1, W2, b2, x):
    Z1 = W1.dot(x)+b1
    A1 = ReLu(Z1)
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1)+b2
    A2 = softmax(A1)
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y=np.zeros((Y.size, Y.max()+1))
    one_hot_Y[np.arange(Y.size), Y]=1

#def back_prop(Z1, A1, Z2, A2, W2, Y):
