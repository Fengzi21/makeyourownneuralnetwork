import pytest
import numpy as np
import matplotlib.pyplot as plt

import dill as pickle

from neural_network import Classifier


class TestClassifier:

    # shared resources
    input_nodes, hidden_nodes, output_nodes = 28 * 28, 200, 10
    learning_rate = 0.1

    # create instance of neural network
    n = Classifier(input_nodes, hidden_nodes, output_nodes, learning_rate)

    @classmethod
    def setup_class(cls):
        print(f'Setting up class: {cls.__name__}')

    def test_query(self):
        input_list = np.random.rand(self.input_nodes)
        final_outputs = self.n.query(input_list)
        print(f'{final_outputs = }')
        assert final_outputs.shape == (self.output_nodes, 1)

    @pytest.mark.plot
    def test_backquery(self):
        label = 0  # label to test
        # create the output signals for this label
        targets = np.zeros(self.output_nodes) + 0.01
        targets[label] = 0.99
        print(f'{targets = }')

        # get image data
        image_data = self.n.backquery(targets)
        assert image_data.shape == (self.input_nodes, 1)

        # plot image data
        plt.figure()
        plt.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
        plt.colorbar()
        plt.tight_layout()
        plt.show()
        plt.close()

    @pytest.mark.plot
    def test_plot_progress(self):
        self.n.progress = np.random.rand(20000)
        self.n.plot_progress()
        plt.show()
        plt.close()

    @pytest.mark.pickle
    def test_pickle(self):
        filename = 'test_classifier.pkl'
        self.n.pickle(filename)

        with open(filename, 'rb') as f:
            nn = pickle.load(f)

        assert isinstance(nn, Classifier)

        for attr in dir(self.n):
            if not attr.startswith('__'):
                assert hasattr(nn, attr)

    @classmethod
    def teardown_class(cls):
        print(f'Tearing down class: {cls.__name__}')
