import itertools
import sys
import os
import numpy as np

if __name__ == '__main__':
    starts = np.arange(0, 9001, 20)
    #ends = np.arange(200, 9201, 200)
    #ends[-1] = 9045
    expt = 'sensr'
    #data_index = range(ends.shape[0])
    i = 0
    lr =  5e-3
    step = 500

    i = int(sys.argv[1])
    start = starts[i]
    end = start + 20
    seeds = np.load('./seeds.npy')
    data_seed = seeds[i, 0]
    expt_seed = seeds[i, 1]
    os.system(f'python3 ./sensr/adv_ratio.py {data_seed} {expt_seed} {lr} {step} {start} {end}')