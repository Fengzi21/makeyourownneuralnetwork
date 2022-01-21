import time
import itertools
import glob

import numpy as np
import scipy.ndimage
import imageio
from caput import mpiutil as mpi
from rich import print

from neural_network import Classifier


with open("mnist_dataset/mnist_train.csv", 'r') as training_data_file:
    training_data_list = training_data_file.readlines()

with open("mnist_dataset/mnist_test.csv", 'r') as test_data_file:
    test_data_list = test_data_file.readlines()

angle_list = [3, 5, 7, 10, 12, 15, 20, 25]
epochs_list = [5, 7, 10, 13, 15]

ini_args = list(itertools.product(angle_list, epochs_list))
# print(len(ini_args))

input_nodes, hidden_nodes, output_nodes = 784, 200, 10
learning_rate = 0.1

for args in mpi.mpilist(ini_args):
    # print(f"Start: {args} by rank {mpi.rank}")

    angle, epochs = args
    n = Classifier(input_nodes, hidden_nodes, output_nodes, learning_rate)

    start = time.perf_counter()
    
    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

            inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), angle, cval=0.01, order=1, reshape=False)
            n.train(inputs_plusx_img.reshape(784), targets)
            # rotated clockwise by x degrees
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -angle, cval=0.01, order=1, reshape=False)
            n.train(inputs_minusx_img.reshape(784), targets)

    training_time = time.perf_counter() - start

    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = outputs.argmax()
        scorecard.append(1) if (label==correct_label) else scorecard.append(0)
            
    scorecard_array = np.asarray(scorecard)
    performance = scorecard_array.sum() / scorecard_array.size

    print(f"{args}: {training_time = :.2f} seconds, {performance = }.")

    filename = f"trained_classifiers/classifier_rotation_{learning_rate}_{epochs}.pkl"
    n.performance = performance
    n.training_time = training_time
    n.epochs = epochs
    n.pickle(filename)