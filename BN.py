import numpy as np
import datetime
from matplotlib import pyplot as plt
import concurrent.futures
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
    mean = np.sum(array, axis=1) / num
    var_sqr = 0
    for i in range(num):
        a = array[:, i]
        k = np.linalg.norm(a - mean, ord=2)
        var_sqr = var_sqr + k * k
    var = np.sqrt(var_sqr)
    m = np.repeat(mean, num)
    m = m.reshape(np.size(mean), num)
    array = array - m
    array = array / var
    return array


def ReLu(x):
    return x * (x > 0)


def feed_forward_once(params, input, gamma, b, BN):
    ## params: the parameter  assigned to current layer of the neural network
    ## input: the input to current layer of NN
    ## gamma: a scalar parameter of batch norm
    ## b: an ndarray parameter of batch norm
    ## BN: True of False
    ## first conduct whitening
    for k in range(np.shape(input)[1]):
        input[:, k] = ReLu(input[:, k])
    ## if conducting batch normalization, then do normalization
    if BN == True:
        input = normalize(input)
    ## add in bias term
    ones = np.ones(np.shape(input)[1])
    input = np.vstack((input, ones))
    output = params.dot(input)
    ## if conducting Batch normalization then conduct rescaling
    if BN == True:
        output = gamma * output + np.tile(b, (1, np.shape(output)[1]))
    return output


def get_freq(x):
    T = collections.Counter(x)
    Y = np.array(list(T.values())) / times
    Y = np.sort(Y)
    Y = Y[::-1]
    return Y


##initialize parameters:
## parameter for NN
n = 7  ## dimension of input data, user-defined
m = 2 ** n  ## number of data points
## k = m * np.log(2)          ## number of possible output functions
k = 2 ** m  ## number of possible output functions
layer_num = 2  ## number of layers of the neural network, user-defined
neu = 40  ## neurons per layer
times = int(1e6)  ## the times to run the program
var_w = 2.5 * (np.sqrt(n))  ## variance of weights
var_b = 2.5  ## variance of bias terms
loc_g = 0.0  ## actually the variance of dataset
var_g = 1.0  ## variance of Gamma
loc_B = 0.0  ## actually the mean of dataset
var_B = 1.0  ## variance of B
BN = [0, 0]  ## indicator for batch normalization
BN[0] = True
BN[1] = False

## initialize data sets as binary strings
data = np.zeros([n, m])
for i in range(m):
    bin = np.binary_repr(i, n)
    data[:, i] = np.array(list(bin), dtype=int)

## run the program
Times = list(range(times))
P_f = np.zeros(times)
P_f_BN = np.zeros(times)
h = 0

##def process_func(h):
while (h < times):
    if h % (times / 10) == 0:
        print(f'{datetime.datetime.now()} Test No.{h} Complete!')

    ## initialize parameters:
    params_start = np.random.normal(loc=0.0, scale=var_w, size=(neu, n))  ## variance = var_w
    params_start_b = np.random.normal(loc=0.0, scale=var_b, size=(neu, 1))  ## variance = var_b
    params_start = np.hstack((params_start, params_start_b))
    params = np.zeros([neu, neu + 1, layer_num - 1])  ## a three-dimensional matrix containing parameter for each layer
    for i in range(layer_num - 1):
        par = np.random.normal(loc=0.0, scale=var_w, size=(neu, neu))
        bias = np.random.normal(loc=0.0, scale=var_b, size=(neu, 1))
        params[:, :, i] = np.hstack((par, bias))

    ## parameter for batch normalization
    Gamma = np.random.normal(loc=loc_g, scale=var_g,
                             size=layer_num)  ## an array containing scalar multiple for each layer
    B = np.zeros([layer_num, neu])
    for i in range(layer_num):
        B[i, :] = np.random.normal(loc=loc_B, scale=var_B, size=neu)

    for Bn in BN:
        ## do feedforward
        if Bn is True:
            data = normalize(data)
        ones = np.ones(np.shape(data)[1])
        output = np.vstack((data, ones))
        output = params_start.dot(output)
        if Bn is True:
            C = np.transpose(B[0, :])
            C = np.tile(C, (m, 1))
            output = Gamma[0] * output + np.transpose(C)
            for j in range(layer_num - 1):
                output = feed_forward_once(params[:, :, j], output, Gamma[j + 1], B[j + 1, :].reshape(neu, 1), Bn)
        else:
            for j in range(layer_num - 1):
                output = feed_forward_once(params[:, :, j], output, 0, 0, Bn)

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
        if Bn is True:
            P_f_BN[h] = np.sum(output)
        else:
            P_f[h] = np.sum(output)
        h = h + 1

##multiprocessing:
## with concurrent.futures.ProcessPoolExecutor() as executor:
## executor.map(process_func, Times)

## Plot P_f
Y = get_freq(P_f)
Z = get_freq(P_f_BN)
t = max(len(Y), len(Z))
# np.save(f'X{MC}.npy', X)
np.save(f'BNY{MC}.npy', Y)
np.save(f'BNZ{MC}.npy', Z)
Min = np.min(np.min(Y), np.min(Z))
X = np.arange(t)
np.save(f'BNX{MC}.npy', X)
U = (np.log(k) * X) ** (-1)
plt.plot(Y, label="no BatchNorm")
plt.plot(Z, label="BatchNorm")
plt.plot(X, U, label="Zipf's Law")
plt.xlabel('Rank')
plt.ylabel('Probability')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc="upper right")
plt.xlim(1, t)
plt.ylim(Min, 1)
plt.show()
