import numpy as np
import itertools
import sys



outfile = sys.argv[1]
expts = ['sensr', 'reduction', 'baseline', 'project'] 
    #data_index = range(ends.shape[0])
iteration = range(10)
lrs = [5e-3]#[5e-4, 2e-3, 5e-3]
steps = [1000]#[10, 20, 40, 80, 160, 320, 640, 1280, 2560]
starts = np.arange(0, 9001, 20)
ends = np.arange(20, 9021, 20)



with open(outfile, 'a') as f:

    for args in itertools.product(expts, iteration, lrs, steps, zip(starts, ends)):
        expt, i, lr, step, (start, end) = args
        #end = 1000
        seeds = np.load('./seeds.npy')
        out_dict = dict()
        out_dict['expt'] = expt
        out_dict['lr'] = lr
        out_dict['step'] = step
        out_dict['start'] = start
        out_dict['end'] = end
        
        if expt == 'reduction':
            seed = seeds[i, 0]
            filename = f'./{expt}/outcome/perturbed_ratio_seed_{seed}_lr_{lr}_step_{step}_start_{start}_end_{end}.npy'
            out_dict['seed-data'] = seed
            out_dict['seed-model'] = 0
        else:
            seed_data = seeds[i, 0]
            seed_model = seeds[i, 1]
            filename = f'./{expt}/outcome/perturbed_ratio_seed_{seed_data}_{seed_model}_lr_{lr}_step_{step}_start_{start}_end_{end}.npy'
            out_dict['seed-data'] = seed_data
            out_dict['seed-model'] = seed_model
        try:
            summary = np.load(filename)
            ratios = summary[:, 0]
            ratios = ratios[~np.isnan(ratios)]
            summary = summary[:, 1:] 
            summary = summary[~np.isnan(summary).any(axis=1), :]
            loss_start, loss_end = summary[:, 0], summary[:, 1]
            #ratios = ratios[~np.isnan(ratios)]
            out_dict['sum-ratio'], out_dict['sum-sq-ratio'],\
               out_dict['sample-size-ratios'] = np.sum(ratios), np.sum(ratios**2), np.shape(ratios)[0] 
            out_dict['sum-start'], out_dict['sum-end'], out_dict['sum-cov'], out_dict['sample-size-0-1']\
               = np.sum(loss_start), np.sum(loss_end), np.sum(loss_start*loss_end), np.shape(loss_start)[0]
            f.writelines(str(out_dict)+'\n')
            print('Done: '+str(args)+'\n')
            print('Ratios, loss_start, loss_end: \n'+str((ratios, loss_start-loss_end))+'\n\n\n')
        except:
            print('No such file exists\n')
            continue
    

    
