import os
#for i in range(40, 80):
    #os.system(f'python3 create_fluctuations.py {i}')
    #os.system(f'python3 cf_reduction.py {i}')

#os.system('rm all_summary.out')
for i in range(80):
    print(f'\n\n\n{i}\n\n\n')
    os.system(f'python3 summary.py {i}')