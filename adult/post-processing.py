import numpy as np
import itertools
import sys



outfile = sys.argv[1]
expts = ['sensr', 'reduction', 'baseline', 'project'] 
    #data_index = range(ends.shape[0])
iteration = range(10)
lrs = [5e-4, 2e-3, 5e-3]
steps = [10, 20, 40, 80, 160, 320]



with open(outfile, 'a') as f:

    for args in itertools.product(expts, iteration, lrs, steps):
        expt, i, lr, step = args
        end = 1000
        seeds = np.load('./seeds.npy')
        out_dict = dict()
        out_dict['expt'] = expt
        out_dict['lr'] = lr
        out_dict['step'] = step
        if expt == 'reduction':
            seed = seeds[i, 0]
            filename = f'./{expt}/outcome/perturbed_ratio_seed_{seed}_lr_{lr}_step_{step}.npy'
            out_dict['seed-data'] = seed
            out_dict['seed-model'] = 0
        else:
            seed_data = seeds[i, 0]
            seed_model = seeds[i, 1]
            filename = f'./{expt}/outcome/perturbed_ratio_seed_{seed_data}_{seed_model}_lr_{lr}_step_{step}.npy'
            out_dict['seed-data'] = seed_data
            out_dict['seed-model'] = seed_model
        ratios = np.load(filename)
        ratios = ratios[~np.isnan(ratios)]
        out_dict['mean'], out_dict['std'], out_dict['sample-size'] = np.mean(ratios), np.std(ratios), np.shape(ratios)[0] 
        f.writelines(str(out_dict)+'\n')
        print('Done: '+str(args)+'\n')

    

    