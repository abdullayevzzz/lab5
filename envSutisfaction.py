import numpy as np
import matplotlib.pyplot as plt
from dnn_lib import *

# test22
learning_rate = 0.075
num_iterations = 10
# TODO: generate more data
raw_x = np.array([[20, 40, 30],
                  [45, 35, 25]])
raw_y = np.array([[1],
                  [0]])

INPUT_SIZE = 3
HID_LAYER1 = 5
HID_LAYER2 = 4
OUTPUT_SIZE = 1
np.random.seed(10)
W1 = np.random.randn(HID_LAYER1, INPUT_SIZE) * 0.1
W2 = np.random.randn(HID_LAYER2, HID_LAYER1) * 0.1
W3 = np.random.randn(OUTPUT_SIZE, HID_LAYER2) * 0.1

b1 = np.zeros((HID_LAYER1, 1))
b2 = np.zeros((HID_LAYER2, 1))
b3 = np.zeros((OUTPUT_SIZE, 1))

print(raw_x)
print(raw_y)
print(raw_x.shape)
train_x = data_normalize(raw_x)
print(train_x)

# appends every iteration cost value, will use for making a plot
cost_history = []
# collection of all parameters
param_values = {}
param_values["W1"] = W1
param_values["b1"] = b1
param_values["W2"] = W2
param_values["b2"] = b2
param_values["W3"] = W3
param_values["b3"] = b3

# train
for i in range(num_iterations):
    A1, A2, A3, Z1, Z2, Z3 = full_forward_propagation(train_x.T, param_values)

    cost = get_cost_value(A3.squeeze(), raw_y.squeeze())
    cost_history.append(cost)

    m = m = A2.shape[1]
    dZ3 = A3 - raw_y.T
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = np.dot(param_values["W3"].T, dZ3)

    dA1, dW2, db2 = single_layer_backward_propagation(dA2, W2, b2, Z2, A1, activation="relu")
    dA0, dW1, db1 = single_layer_backward_propagation(dA1, W1, b1, Z1, train_x.T, activation="relu")
    # TODO: update parameters W1, W2, W3, b1, b2, b3
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1


print("Z3:", Z3)
print("A3:", A3)

print("W3", W3)
print("b3", b3)
print("W2", W2)
print("b2", b2)
print("W1", W1)
print("b1", b1)
print(cost)

A_prediction = []
print("A prediction", A_prediction)

plt.plot(cost_history)
plt.show()
