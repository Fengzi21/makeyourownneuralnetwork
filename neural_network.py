# neural_network.py


import numpy as np
from scipy.special import expit, logit  # sigmoid function and it's inverse
import dill as pickle                   # for pickling the trained neural network
import pandas as pd                     # for plotting progress
from rich import print                  # comment this out if rich is not installed


def to_col_vec(row_list):
    """Convert a non-nested list to a column vector.
    i.e. with shape (len(row_list, 1))
    """
    return np.array(row_list, ndmin=2).T


def scale(signal):
    """Scale signal to range 0.01 ~ 0.99.
    """
    signal -= signal.min()
    signal /= signal.max()
    signal *= 0.98
    signal += 0.01

    return signal


class Classifier:
    """A MNIST dataset classifier building from scratch with pure python, numpy and scipy.
    """
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """Initialize the neural network.
        """
        # set number of nodes in input, hidden and output layers
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
        # inverse activation function is the logit function
        self.inverse_activation_function = lambda x: logit(x)

        # alias of `forward` for querying the neural network
        self.query = self.forward

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

    def forward(self, inputs_list, *, return_hidden_outputs=False):
        """Feed the input signal forward to get outputs.
        """
        # convert inputs list to 2D arrays
        inputs = to_col_vec(inputs_list)

        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return (final_outputs, hidden_outputs) if return_hidden_outputs else final_outputs

    def train(self, inputs_list, targets_list, print_counter=False):
        """Train the neural network.
        """
        # convert inputs and targets lists to 2D arrays
        inputs = to_col_vec(inputs_list)
        targets = to_col_vec(targets_list)

        # == feed forward == #
        final_outputs, hidden_outputs = self.forward(inputs_list, return_hidden_outputs=True)

        # == backpropagation == #
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs

        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = np.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)

        # increase counter
        self.counter += 1

        # accumulate error every 10 loops
        if self.counter % 10 == 0:
            self.progress.append(np.sum(output_errors**2) / output_errors.size)

        # print counter every 10000 loops
        if print_counter and (self.counter % 10000 == 0):
            print('counter = ', self.counter)

    def backquery(self, targets_list):
        """Backquery the neural network.

        We'll use the same termnimology to each item,
        e.g. target are the values at the right of the network, albeit used as input
        e.g. hidden_output is the signal to the right of the middle nodes
        """

        # transpose the targets list to a vertical array
        final_outputs = to_col_vec(targets_list)

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer and scale them back to 0.01 to 0.99
        hidden_outputs = scale(np.dot(self.who.T, final_inputs))

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer and scale them back to 0.01 to .99
        inputs = scale(np.dot(self.wih.T, hidden_inputs))

        return inputs

    def plot_progress(self):
        """Plot classifier error.
        """
        df = pd.DataFrame(self.progress, columns=['loss'])
        plt_kwargs = {
            'figsize': (16, 8),
            # 'ylim':  (0, 1.0),
            'ylim': (0, max(self.progress)),
            'alpha': 0.1,
            'marker': '.',
            'grid': True,
            # 'yticks':  (0, 0.25, 0.5)
        }
        df.plot(**plt_kwargs)

    def pickle(self, filename='classifier.pkl'):
        """Pickle and save the trained classifier.
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
