import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

training_inputs = np.array([[1,0,1],
                            [0,1,0],
                            [1,1,1],
                            [1,0,0]])

training_outputs = np.array([[1,0,1,1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,1)) - 1

for i in range (10000):
    
    inputs = training_inputs
    outputs = sigmoid(np.dot(inputs,synaptic_weights))

    Error = training_outputs - outputs

    adjustement  = np.dot(inputs.T, Error * (outputs * (1 - outputs)) )

    synaptic_weights += adjustement

new_situation = np.array([1,0,1])
output = np.dot(new_situation,synaptic_weights)

print(output)
