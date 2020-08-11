#!/usr/bin/env python

import os


job_file = 'submit.sbat'

for index in range(1200):
    '''creates job submitter script'''
    os.system(f'touch {job_file}')

        
    with open(job_file,'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines(f"#SBATCH --job-name=running_{str(index)}.job\n")
        fh.writelines('#SBATCH --nodes=1\n')
        fh.writelines('#SBATCH --cpus-per-task=1\n')
        fh.writelines('#SBATCH --mem-per-cpu=6gb\n')
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --account=yuekai1\n")
        fh.writelines("#SBATCH --mail-type=NONE\n")
        fh.writelines("#SBATCH --mail-user=smaity@umich.edu\n")
        fh.writelines('#SBATCH --partition=standard\n')
        fh.writelines(f"python3 create_fluctuations.py {index}")

    os.system("sbatch %s" %job_file)
    os.system(f'rm {job_file}') 
