# Individual Fairness Testing on Simulated data

We perform some simulation studies related to the individual fairness test. A description and plot for the simulated data is presented in plot.ipynb file. We consider several logistic classifiers with given weights and appropriate biases for comparison purpose. The fair direction was determined from expert knowledge. We also perturb the fair direction by a certain angle to study performance of the test under mis-specified fair metric. 

We summarize the functionality of each script below.

| Script | Functionality | 
| --- | --- |
| `generate_data.py` | Generates synthetic data. Look at `plot.ipynb` for data description. |
| `loss_linear.py` | Calculates lower bound for average ratio of losses over a grid of weights. |  

Command to generate synthetic data:
```bash
python3 generate_data.py
```

The following arguments in `loss_linear.py` controls the important parameters in experiments:

| Argument | Description | 
| --- | --- | 
| `arg 1`| Angle of rotation for fair direction. |
| `arg 2` | Regularizer for fair distance. | 
| `arg 3` | Initial learning rate for gradient flow attack. The learning rate at step `i` is `learning rate / (i^{2/3})`.  |
| `arg 4` | Total number of gradient flow steps to be performed. | 

Demo code for calculating lower bounds with perturbation angle 5 degree, regularizer 10, learning rate 0.1, and number of steps 100 is given below. The lower bounds are saved as `./data/test_stat_ang_5_reg_10_lr_0.1_step_100.npy`.
```bash
python3 loss_linear.py 5 10 0.1 100
```
Demo code for calculating lower bounds with perturbation angle 5 degree, and default regularizer, learning rate, number of steps is given below. The lower bounds are saved as `./data/test_stat_5.npy`.
```bash
python3 loss_linear.py 5
```

Codes for reproducing the results are given below. The plot is saves as `./plots/mean_ratios.pdf`.
```bash
python3 loss_linear.py 0
python3 loss_linear.py 5
python3 loss_linear.py 10
python3 plot.py
```
