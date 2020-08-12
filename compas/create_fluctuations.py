import itertools
import sys
import os
import numpy as np





def part_fluc(args):
    expt, d, i, lr = args
    start = starts[d]
    end = ends[d]
    np.random.seed(1)
    seeds = np.load('./seeds.npy')
    if expt == 'reduction':
        #seeds = np.random.randint(100000, size = (10, ))
        seed = seeds[i, 0]
        os.system(f'python3 ./{expt}/adv_ratio.py {start} {end} {seed} {lr}')
    elif expt == 'project':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/adv_ratio.py {start} {end} {data_seed} {expt_seed} {lr}')
    elif expt == 'baseline':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/adv_ratio.py {start} {end} {data_seed} {expt_seed} {lr}')

    else:
        #seeds = np.random.randint(100000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/adv_ratio.py {start} {end} {data_seed} {expt_seed} {lr}')

if __name__ == '__main__':
    starts = np.array([0,])#np.arange(0, 901, 100)
    ends = np.array([1000,])#np.arange(100, 1001, 100)
    expts = ['sensr', 'reduction', 'baseline', 'project'] 
    data_index = range(1)#ends.shape[0])
    iteration = range(10)
    lrs = [5e-3, 4e-3, 3e-3, 2e-3, 6e-3, 7e-3]

    a = itertools.product(expts, data_index, iteration, lrs)
    b = [i for i in a]
    i = int(sys.argv[1])
    print(b[i])
    part_fluc(b[i])

