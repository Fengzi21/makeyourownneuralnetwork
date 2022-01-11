import numpy as np
from scipy.special import expit  # sigmoid function


# neural network class definition
class NeuralNetwork:

    # initialize the neural network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set number of nodes in each input, hidden and output layers
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        # link weight matrices, wih and who
        # weights inside the arrays are wij,
        # where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learning_rate

        # activation function is the sigmoid function
        self.activation_function = lambda x: expit(x)

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2D array
        inputs = np.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


if __name__ == '__main__':
    input_nodes, hidden_nodes, output_nodes = 3, 3, 3
    learning_rate = 0.3

    # create instance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    input_list = [1.0, 0.5, -1.5]
    final_outputs = n.query(input_list)
    print(final_outputs)
