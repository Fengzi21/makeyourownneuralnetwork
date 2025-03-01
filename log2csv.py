import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--logfile', default='perf.log', type=str, help='Input logfile.')
parser.add_argument('-o', '--output', default='performance.csv', type=str, help='Output csv file.')
args = parser.parse_args()

perfile = args.logfile
csvfile = args.output

with open(perfile, 'r') as f:
    lines = sorted(f.readlines())

# print(len(lines))

hidden_nodes, learning_rate, epochs, perf_list, tt_list = [], [], [], [], []

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
    perf_list.append(performance)  # type: ignore
    tt_list.append(training_time)  # type: ignore

columns_name = ['hidden_nodes', 'learning_rate', 'epochs', 'performance', 'training_time']
data = np.asarray([np.array(i) for i in (hidden_nodes, learning_rate, epochs, perf_list, tt_list)])

table = pd.DataFrame(data.T, columns=columns_name)
table.to_csv(csvfile)
