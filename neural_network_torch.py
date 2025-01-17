import torch
import torch.nn as nn
from functools import partial
from torch.autograd import Variable

try:
    from rich import print
except Exception:
    pass


class Classifier(nn.Module):
    def __init__(self, inodes, hnodes, onodes, learning_rate, device=None):
        # call the base class's initialisation too
        super().__init__()

        # using cpu by default
        if device is None:
            device = torch.device('cpu')

        # dimensions
        self.inodes, self.hnodes, self.onodes = inodes, hnodes, onodes

        # learning rate
        self.lr = learning_rate

        # define the layers and their sizes, turn off bias
        self.linear_ih = nn.Linear(inodes, hnodes, bias=False)
        self.linear_ho = nn.Linear(hnodes, onodes, bias=False)

        # define activation function
        self.activation = nn.Sigmoid()

        # define inverse activation function
        self.inverse_activation_function = torch.logit

        # create error function
        # self.error_function = torch.nn.MSELoss(size_average=False)  # size_average is deprecated
        self.error_function = torch.nn.MSELoss(reduction='sum')

        # create optimiser, using simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), self.lr)

        # use GPU if it is available
        # self.FloatTensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.FloatTensor  # on longer recommended
        self.FloatTensor = partial(torch.tensor, dtype=torch.float32, device=device.type)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

    def forward(self, inputs_list):
        # convert list to a 2-D FloatTensor then wrap in Variable
        inputs = Variable(self.FloatTensor(inputs_list).view(1, self.inodes))

        # combine input layer signals into hidden layer
        hidden_inputs = self.linear_ih(inputs)

        # apply sigmiod activation function
        hidden_outputs = self.activation(hidden_inputs)

        # combine hidden layer signals into output layer
        final_inputs = self.linear_ho(hidden_outputs)

        # apply sigmiod activation function
        final_outputs = self.activation(final_inputs)

        return final_outputs

    def train(self, inputs_list, targets_list, print_counter=False):
        # calculate the output of the network
        output = self.forward(inputs_list)

        # create a Variable out of the target vector, doesn't need gradients calculated
        target_variable = Variable(self.FloatTensor(targets_list).view(1, self.onodes), requires_grad=False)

        # calculate error
        loss = self.error_function(output, target_variable)

        # increase counter
        self.counter += 1

        # accumulate error every 10 loops
        if self.counter % 10 == 0:
            self.progress.append(loss.item())

        # print counter every 10000 loops
        if print_counter and (self.counter % 10000 == 0):
            print('counter = ', self.counter)

        # zero gradients, perform a backward pass, and update the weights.
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def backquery(self, targets_list):
        """Backquery the neural network.
        """

        def _scale(signal):
            # scale signal to 0.01 to 0.99
            signal -= signal.min().item()
            signal /= signal.max().item()
            signal *= 0.98
            signal += 0.01
            return signal

        # create a Variable out of the target vector, doesn't need gradients calculated
        target_variable = Variable(self.FloatTensor(targets_list).view(1, self.onodes), requires_grad=False).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(target_variable)

        # calculate the signal out of the hidden layer and scale them back to 0.01 to 0.99
        hidden_outputs = _scale(torch.matmul(self.linear_ho.weight.T, final_inputs))

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer and scale them back to 0.01 to .99
        inputs = _scale(torch.matmul(self.linear_ih.weight.T, hidden_inputs))

        return inputs

    def plot_progress(self):
        """Plot classifier error.
        """
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError('pandas needs to be installed to use this method.')
        else:
            df = pd.DataFrame(self.progress, columns=['loss'])
            plt_kwargs = {
                'figsize': (16, 8),
                # 'ylim':  (0, 1.0),
                'ylim': (0, max(self.progress)),
                'alpha': 0.1,
                'marker': '.',
                'grid': True,
            }
            df.plot(**plt_kwargs)

    def pickle(self, filename='classifier.pkl'):
        """Pickle and save the trained classifier.

        Note:
        In case of this method failed, you can use `torch.save` and `torch.load`
        """
        try:
            import dill as pickle
        except Exception:
            raise RuntimeError('dill needs to be installed to use this method.')
        else:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)