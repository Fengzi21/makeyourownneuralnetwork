import time
import itertools
from pathlib import Path

import numpy as np
import mpiutil as mpi
from rich import print

from neural_network import Classifier


save_dir = Path('./trained_classifiers')
save_dir.mkdir(parents=True, exist_ok=True)

with open('mnist_dataset/mnist_train.csv', 'r') as training_data_file:
    training_data_list = training_data_file.readlines()

with open('mnist_dataset/mnist_test.csv', 'r') as test_data_file:
    test_data_list = test_data_file.readlines()

hidden_nodes_list = [10, 20, 50, 100, 200, 300, 400, 500]
learning_rate_list = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
epochs_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

ini_args = list(itertools.product(hidden_nodes_list, learning_rate_list, epochs_list))

input_nodes, output_nodes = 784, 10

for args in mpi.mpilist(ini_args):
    # print(f"Start: {args} by rank {mpi.rank}")

    hidden_nodes, learning_rate, epochs = args
    n = Classifier(input_nodes, hidden_nodes, output_nodes, learning_rate)

    start = time.perf_counter()

    for e in range(epochs):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    training_time = time.perf_counter() - start

    scorecard = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asarray(all_values[1:], dtype=float) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = outputs.argmax()
        scorecard.append(1) if (label == correct_label) else scorecard.append(0)

    scorecard_array = np.asarray(scorecard, dtype=float)
    performance = scorecard_array.sum() / scorecard_array.size

    print(f'{args}: {training_time = :.2f} seconds, {performance = :.4f}.')

    filename = save_dir / f'classifier_{hidden_nodes}_{learning_rate}_{epochs}.pkl'
    n.performance = performance
    n.training_time = training_time
    n.epochs = epochs
    n.pickle(filename)
    print(f'Saved {filename}')
