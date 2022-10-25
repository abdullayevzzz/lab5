import numpy as np
import matplotlib.pyplot as plt
from dnn_lib import *

learning_rate = 0.075
num_iterations = 10
#TODO: generate more data
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
W3=[]#TODO: generate random data for W2 and W3 weights

b1 = np.zeros((HID_LAYER1, 1))
#TODO: generate b2 and b3 with zeros

print(raw_x)
print(raw_y)
print(raw_x.shape)
train_x = data_normalize(raw_x)
print(train_x)

#appends every iteration cost value, will use for making a plot
cost_history = []
#collection of all parameters
param_values = {}
param_values["W1"] = W1
param_values["b1"] = b1
#TODO: add also W2, b2, W3, b3

# train
for i in range(num_iterations):
    #TODO: call full_forward_propagation with proper input values and param_values
    A1=[], A2=[], A3=[], Z1=[], Z2=[] #TODO: get A3, A2, A1, Z2, Z1 from memory returned by full_forward_propagation

    cost = 0 #TODO: call get_cost_value function for calculating the cost
    cost_history.append(cost)

    m = m = A2.shape[1]
    dZ3 = A3 - raw_y.T
    dW3 = np.dot(dZ3, A2.T) / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = np.dot(param_values["W3"].T, dZ3)
    #TODO: call single_layer_backward_propagation 2 times for remaining 2 layers and get dA1, dW2, db2, dA0, dW1, db1

    #TODO: update parameters W1, W2, W3, b1, b2, b3


print("Z3:", Z3)
print("A3:", A3)

print("W3", W3)
print("b3", b3)
print("W2", W2)
print("b2", b2)
print("W1", W1)
print("b1", b1)
print(cost)

#TODO: create an input data for prediction
#TODO: make prediction
A_prediction=[]
print("A prediction", A_prediction)

plt.plot(cost_history)
plt.show()