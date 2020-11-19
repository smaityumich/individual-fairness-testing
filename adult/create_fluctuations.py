import itertools
import sys
import os
import numpy as np





def part_fluc(args):
    starts = np.arange(0, 9001, 200)
    ends = np.arange(200, 9201, 200)
    ends[-1] = 9045
    print(str(args)+'\n\n')
    expt, i, lr, step, d = args
    start = starts[d]
    end = ends[d]
    np.random.seed(1)
    seeds = np.load('./seeds.npy')
    if expt == 'reduction':
        #seeds = np.random.randint(100000, size = (10, ))
        seed = seeds[i, 0]
        os.system(f'python3 ./{expt}/adv_ratio.py {seed} {lr} {step} {start} {end}')
    elif expt == 'project':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/adv_ratio.py {data_seed} {expt_seed} {lr} {step} {start} {end}')
    elif expt == 'baseline':
        #seeds = np.random.randint(10000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/adv_ratio.py {data_seed} {expt_seed} {lr} {step} {start} {end}')

    else:
        #seeds = np.random.randint(100000, size = (10, 2))
        data_seed = seeds[i, 0]
        expt_seed = seeds[i, 1]
        os.system(f'python3 ./{expt}/adv_ratio.py {data_seed} {expt_seed} {lr} {step} {start} {end}')

if __name__ == '__main__':
    starts = np.arange(0, 9001, 200)
    ends = np.arange(200, 9201, 200)
    ends[-1] = 9045
    expts = ['sensr', 'reduction', 'baseline', 'project'] 
    data_index = range(ends.shape[0])
    iteration = range(10)
    lrs = [5e-3]#[5e-4, 2e-3, 5e-3]
    steps = [500]#[10, 20, 40, 80, 160, 320, 640, 1280, 2560]

    a = itertools.product(expts, iteration, lrs, steps, data_index)
    b = [i for i in a]
    i = int(sys.argv[1])
    part_fluc(b[i])
    #print(f'Done {i}\n\n')

