# Individual Fairness Testing on Simulated data

We perform some simulation studies related to the individual fairness test. A description and plot for the simulated data is presented in plot.ipynb file. We consider sevaral logistic classifiers with given weights and appropriate biases for comparison purpose. The fair direction was determined from expert knowledge. We also perturb the fair direction by a certain angle to study performance of the test under misspecified fair metric. 

We summarize the functionality of each script below.

| Script | Functionality | 
| --- | --- |
| `generate_data.py` | Generates synthetic data. Look at `plot.ipynb` for data description. |
| `loss_linear.py` | Calculates lower bound for average ratio of losses over a grid of weights. |  