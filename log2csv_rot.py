import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('-l', '--logfile', default='perf_rotation.log', type=str, help='Input logfile.')
parser.add_argument('-o', '--output', default='performance_rotation.csv', type=str, help='Output csv file.')
args = parser.parse_args()

perfile = args.logfile
csvfile = args.output

with open(perfile, 'r') as f:
    lines = sorted(f.readlines())

angle, epochs, perf_list, tt_list = [], [], [], []

for line in lines:
    if not line.startswith('('):
        continue
    aepart, tfpart = line[:-2].split(':')
    tfpart = tfpart.replace('seconds,', ';')
    a, e = eval(aepart)
    angle.append(a)
    epochs.append(e)
    exec(tfpart.strip())
    perf_list.append(performance)  # type: ignore
    tt_list.append(training_time)  # type: ignore

columns_name = ['angle', 'epochs', 'performance', 'training_time']
data = np.asarray([np.array(i) for i in (angle, epochs, perf_list, tt_list)])

table = pd.DataFrame(data.T, columns=columns_name)
table.to_csv(csvfile)
