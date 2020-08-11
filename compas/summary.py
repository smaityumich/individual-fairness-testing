import itertools
import sys
import os
import numpy as np

def part_summary(args):
    expt, i, lr = args
    np.random.seed(1)
    seeds = np.load('./seeds.npy')
    if expt == 'reduction':
        seed = seeds[i, 0]
        os.system(f'python3 ./{expt}/summary.py {seed} {lr}')
    elif expt == 'baseline_bal':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')
    elif expt == 'baseline':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')

    else:
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/summary.py {data_seed} {expt_seed} {lr}')


if __name__ == '__main__':
    
    expts = ['baseline', 'baseline_bal', 'sensr', 'reduction'] 
    iteration = range(10)
    lrs = [2e-3, 5e-3]

    a = itertools.product(expts, iteration, lrs)
    b = [i for i in a]
    i = int(sys.argv[1])
    part_summary(b[i])





