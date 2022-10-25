import numpy as np


#TODO: normalize input data
def data_normalize(raw_data):
    return [] #TODO: implement

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

#TODO: implement relu function
def relu(Z):
    return []#TODO:implement

#TODO: complete implementation of function for single layer forward propagation
#it should compute linear propagation based on input values and parameters
#based on activation parameter should choose the proper non-linear activation function and apply calculations
#return: activation A and linear Z matrix
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    # TODO: calculation of linear propagation

    # selection of activation function
    if activation is "relu":
        activation_func = relu
    elif activation is "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
    #TODO: calculate non-linear activation and return proper values
    # return of calculated activation A and the intermediate Z matrix

#TODO: implement full forward propagation for all layers return last lyaer A3 value and
# all intermidiate matrixes A1, Z1... Z3 as memory list
def full_forward_propagation(X, params_values):
    #TODO: single_layer_forward_propagation with proper parameters 3 times for 3 layers
    #TODO: create memory list with all intermidiate values A1, Z1... Z3
    return []#TODO: return A3 and memory list

def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[1]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

#TODO:implement backward relu function return dZ
def relu_backward(dA, Z):
    return [] #TODO:implement

#TODO: complete implementation of function for single layer backward propagation
def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]

    # selection of activation function
    if activation is "relu":
        backward_activation_func = relu_backward
    elif activation is "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    # TODO: calculation of the activation function derivative
    dZ_curr=[] #TODO:complete, make calculations instead of empty declaration
    # TODO: calculate dW_curr derivative of the matrix W as (dZ*A_prev.T)/m

    #derivative of the vector b
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    #TODO: calculate a dA_prev derivative of the matrix A_prev as W_curr.T*dZ_curr

    #TODO: return dA_prev, dW_curr, db_curr

