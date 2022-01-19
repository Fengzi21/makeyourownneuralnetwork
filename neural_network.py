import numpy as np
from scipy.special import expit, logit  # sigmoid function and it's inverse


# neural network class definition
class NeuralNetwork:

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """initialize the neural network"""
        # set number of nodes in each input, hidden and output layers
        self.inodes, self.hnodes, self.onodes = input_nodes, hidden_nodes, output_nodes

        # random initial link weight matrices, wih and who
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
        self.inverse_activation_function = lambda x: logit(x)

        
    def _forward(self, inputs, return_hidden_outputs=False):
        """feed the input signal forward to get outputs"""
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        if return_hidden_outputs:
            return final_outputs, hidden_outputs
        else:
            return final_outputs
    
        
    def train(self, inputs_list, targets_list):
        """train the neural network"""
        # convert inputs and targets lists to 2D arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        #== feed forward ==#
        final_outputs, hidden_outputs = self._forward(inputs, return_hidden_outputs=True)

        #== backpropagation ==#
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        
    def query(self, inputs_list):
        """query the neural network"""
        # convert inputs list to 2D arrays
        inputs = np.array(inputs_list, ndmin=2).T

        return self._forward(inputs)
    

    def backquery(self, targets_list):
        """backquery the neural network.
        we'll use the same termnimology to each item, 
        eg target are the values at the right of the network, albeit used as input
        eg hidden_output is the signal to the right of the middle nodes
        """
        def _scale(signal):
            # scale signal to 0.01 to 0.99
            signal -= np.min(signal)
            signal /= np.max(signal)
            signal *= 0.98
            signal += 0.01
            return signal
        
        # transpose the targets list to a vertical array
        final_outputs = np.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)
        # calculate the signal out of the hidden layer and scale them back to 0.01 to 0.99
        hidden_outputs = _scale(p.dot(self.who.T, final_inputs))
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        # calculate the signal out of the input layer and scale them back to 0.01 to .99
        inputs = _scale(np.dot(self.wih.T, hidden_inputs))
        
        return inputs

    
if __name__ == '__main__':
    input_nodes, hidden_nodes, output_nodes = 3, 3, 3
    learning_rate = 0.3

    # create instance of neural network
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    input_list = [1.0, 0.5, -1.5]
    final_outputs = n.query(input_list)
    print(final_outputs)
