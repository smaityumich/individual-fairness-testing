# Individual Fairness Testing on Simulated data

We perform some simulation studies related to the individual fairness test. A description and plot for the simulated data is presented in plot.ipynb file. We consider sevaral logistic classifiers with given weights and appropriate biases for comparison purpose. The fair direction was determined from expert knowledge. We also perturb the fair direction by a certain angle to study performance of the test under misspecified fair metric. 

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

For reproduce the results: 
```bash
python3 loss_linear.py 0
python3 loss_linear.py 5
python3 loss_linear.py 10
pyython3 plot.py
```
