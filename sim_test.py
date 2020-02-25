import numpy as np
import datetime
from matplotlib import pyplot as plt
import pickle
import collections
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("MC")
args = parser.parse_args()
MC = args.MC

def normalize(array):
    ## normalize an input array, each column of which represents a data point
    num = np.shape(array)[1]
    mean = np.sum(array, axis=1)/num
    var_sqr = 0
    for i in range(num):
        a = array[:,i]
        k = np.linalg.norm(a-mean, ord = 2)
        var_sqr = var_sqr +  k * k
    var = np.sqrt(var_sqr)
    m = np.repeat(mean, num)
    m = m.reshape(np.size(mean),num)
    array = array - m
    array = array/var
    return array

def ReLu(x):
    return x * (x > 0)

def feed_forward_once(params, input, gamma):
    ## params: the parameter  assigned to current layer of the neural network
    ## input: the input to current layer of NN
    ## gamma: a scalar parameter of batch norm
    ## b: an ndarray parameter of batch norm
    ## first conduct whitening
    for k in range(np.shape(input)[1]):
        input[:,k] = ReLu(input[:,k])
    input = normalize(input)
    ## add in bias term
    ones = np.ones(np.shape(input)[1])
    input = np.vstack((input, ones))
    input = params.dot(input)
    output = gamma * input
    return output

def count(a, b):
    ## a: an array
    ## b: a matrix
    ## the function counts the appearance of a as a row of b
    count = 0
    for i in range(np.shape(b)[0]):
        if np.array_equal(a, b[i,:]):
            count = count + 1
    return count

##initialize parameters:
## parameter for NN
n = 7             ## dimension of input data, user-defined
m = 2**n          ## number of data points
k = 2**m           ## number of possible output functions
layer_num = 2      ## number of layers of the neural network, user-defined
neu = 40           ## neurons per layer
times = int(1e6)       ## the times to run the program
var_w = 2.5 * (np.sqrt(n))   ## variance of weights
var_b = 2.5                  ## variance of bias terms

## parameter for batch normalization
Gamma = np.zeros(layer_num)        ## an array containing scalar multiple for each layer
for i in range(layer_num):
    Gamma[i] = 1                  ## need to fill in this one

## initialize data sets as binary strings
data = np.zeros([n,m])
for i in range(m):
    bin = np.binary_repr(i,n)
    data[:,i] = np.array(list(bin), dtype=int)

## run the program
h = 0
P_f = np.zeros(times)

while (h < times):

    if h % 10000 == 0:
        print(f'{datetime.datetime.now()} Test No.{h} Complete!')

    ## initialize parameters:
    params_start = np.random.normal(loc=0.0, scale=var_w, size=(neu,n))  ## variance = var_w
    params_start_b = np.random.normal(loc=0.0, scale=var_b, size=(neu,1))  ## variance = var_b
    params_start = np.hstack((params_start,params_start_b))
    params = np.zeros([neu,neu+1,layer_num-1])     ## a three-dimensional matrix containing parameter for each layer
    for i in range(layer_num-1):
        par = np.random.normal(loc=0.0, scale=var_w, size=(neu,neu))
        bias = np.random.normal(loc=0.0, scale=var_b, size=(neu,1))
        params[:, :, i] = np.hstack((par,bias))

    ## do feedforward
    ones = np.ones(np.shape(data)[1])
    output = np.vstack((data,ones))
    output = params_start.dot(output)
    for j in range(layer_num-1):
        output = feed_forward_once(params[:,:,j], output, Gamma[j])

    ## the perceptron at the end
    ones = np.ones(m)
    output = np.vstack((output, ones))
    wb = np.zeros(neu + 1)
    for i in range(neu):
        wb[i] = np.random.normal(loc=0.0, scale=var_w, size=None)
    wb[neu] = np.random.normal(loc=0.0, scale=var_b, size=None)
    output = wb.dot(output)
    output = ReLu(output)
    for k in range(len(output)):
        if output[k] > 0:
            output[k] = 1

    ## read function complexity
    P_f[h] = np.sum(output)
    h = h + 1

## Plot P_f
T = collections.Counter(P_f)
Y = np.array(list(T.values()))/times
Y = np.sort(Y)
Y = Y[::-1]
t = len(Y)
X = np.arange(t)
Z = (np.log(k) * X) ** (-1)
np.save(f'X{MC}.npy', X)
np.save(f'Y{MC}.npy', Y)
np.save(f'Z{MC}.npy', Z)

plt.plot(X,Y)
plt.plot(X,Z)
plt.xscale('log')
plt.yscale('log')
plt.show()
