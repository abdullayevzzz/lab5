import numpy as np


def data_normalize(raw_data):
    max_val = max([max(row) for row in raw_data])
    return raw_data/max_val

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

#it should compute linear propagation based on input values and parameters
#based on activation parameter should choose the proper non-linear activation function and apply calculations
#return: activation A and linear Z matrix
def single_layer_forward_propagation(A_prev, W_curr, b_curr, activation="relu"):
    Z = W_curr.dot(A_prev) + b_curr
    # selection of activation function
    if activation == "relu":
        activation_func = relu
    elif activation == "sigmoid":
        activation_func = sigmoid
    else:
        raise Exception('Non-supported activation function')
    A = activation_func(Z)
    # return of calculated activation A and the intermediate Z matrix
    return A, Z


def full_forward_propagation(X, params_values):
    A1, Z1 = single_layer_forward_propagation(X, params_values['W1'], params_values['b1'], activation="relu")
    A2, Z2 = single_layer_forward_propagation(A1, params_values['W2'], params_values['b2'], activation="relu")
    A3, Z3 = single_layer_forward_propagation(A2, params_values['W3'], params_values['b3'], activation="sigmoid")
    return A1, A2, A3, Z1, Z2, Z3

def get_cost_value(Y_hat, Y):
    # number of examples
    m = Y_hat.shape[0]
    # calculation of the cost according to the formula
    cost = -1 / m * (np.dot(Y, np.log(Y_hat).T) + np.dot(1 - Y, np.log(1 - Y_hat).T))
    return np.squeeze(cost)


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    return dA * (Z >= 0)  # elementwise multiply with boolean value

def single_layer_backward_propagation(dA_curr, W_curr, b_curr, Z_curr, A_prev, activation="relu"):
    # number of examples
    m = A_prev.shape[1]
    # selection of activation function
    if activation == "relu":
        backward_activation_func = relu_backward
    elif activation == "sigmoid":
        backward_activation_func = sigmoid_backward
    else:
        raise Exception('Non-supported activation function')

    dZ_curr = backward_activation_func(dA_curr, Z_curr)
    dW_curr = dZ_curr.dot(A_prev.T)/m
    db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
    dA_prev = W_curr.T.dot(dZ_curr)
    return dA_prev, dW_curr, db_curr

