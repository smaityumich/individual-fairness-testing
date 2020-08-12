import os
for i in range(40, 80):
    os.system(f'python3 create_fluctuations.py {i}')
    #os.system(f'python3 cf_reduction.py {i}')

