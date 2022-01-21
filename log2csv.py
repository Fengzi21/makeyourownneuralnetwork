import pandas as pd
import numpy as np


perfile = "perf.log"

with open(perfile, 'r') as f:
    lines = sorted(f.readlines())

# print(len(lines))

hidden_nodes, learning_rate, epochs, performance, training_time = [], [], [], [], []

for i in range(0, len(lines), 2):
    hn, lr, e = eval(lines[i].split(":")[0])
    perf = float(lines[i].split("=")[-1][:-1])
    tt = lines[i+1].split("=")[-1].strip().split(" ")[0]

    hidden_nodes.append(hn)
    learning_rate.append(lr)
    epochs.append(e)
    performance.append(perf)
    training_time.append(tt)

columns_name = ['hidden_nodes', 'learning_rate', 'epochs', 'performance', 'training_time']
data = np.asarray([np.array(i) for i in (hidden_nodes, learning_rate, epochs, performance, training_time)])

table = pd.DataFrame(data.T, columns=columns_name)
table.to_csv('performance.csv')


perfile = "perf_rotation.log"

with open(perfile, 'r') as f:
    lines = sorted(f.readlines())

angle, epochs, performance, training_time = [], [], [], []

for i in range(0, len(lines), 2):
    a, e = eval(lines[i].split(":")[0])
    perf = float(lines[i].split("=")[-1][:-1])
    tt = lines[i+1].split("=")[-1].strip().split(" ")[0]
    angle.append(a)
    epochs.append(e)
    performance.append(perf)
    training_time.append(tt)

columns_name = ['angle', 'epochs', 'performance', 'training_time']
data = np.asarray([np.array(i) for i in (angle, epochs, performance, training_time)])

table = pd.DataFrame(data.T, columns=columns_name)
table.to_csv('performance_rotation.csv')



