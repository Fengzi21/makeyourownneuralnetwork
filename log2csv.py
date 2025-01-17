import pandas as pd
import numpy as np


perfile = 'perf.log'

with open(perfile, 'r') as f:
    lines = sorted(f.readlines())

# print(len(lines))

hidden_nodes, learning_rate, epochs, performance_list, training_time_list = [], [], [], [], []

for line in lines:
    if not line.startswith('('):
        continue
    hlepart, tfpart = line[:-2].split(':')
    tfpart = tfpart.replace('seconds,', ';')
    hn, lr, e = eval(hlepart)
    hidden_nodes.append(hn)
    learning_rate.append(lr)
    epochs.append(e)
    exec(tfpart.strip())
    performance_list.append(performance)  # type: ignore
    training_time_list.append(training_time)  # type: ignore

columns_name = ['hidden_nodes', 'learning_rate', 'epochs', 'performance', 'training_time']
data = np.asarray([np.array(i) for i in (hidden_nodes, learning_rate, epochs, performance_list, training_time_list)])

table = pd.DataFrame(data.T, columns=columns_name)
table.to_csv('performance.csv')


perfile = 'perf_rotation.log'

with open(perfile, 'r') as f:
    lines = sorted(f.readlines())

angle, epochs, performance_list, training_time_list = [], [], [], []

for line in lines:
    if not line.startswith('('):
        continue
    aepart, tfpart = line[:-2].split(':')
    tfpart = tfpart.replace('seconds,', ';')
    a, e = eval(aepart)
    angle.append(a)
    epochs.append(e)
    exec(tfpart.strip())
    performance_list.append(performance)  # type: ignore
    training_time_list.append(training_time)  # type: ignore

columns_name = ['angle', 'epochs', 'performance', 'training_time']
data = np.asarray([np.array(i) for i in (angle, epochs, performance_list, training_time_list)])

table = pd.DataFrame(data.T, columns=columns_name)
table.to_csv('performance_rotation.csv')
