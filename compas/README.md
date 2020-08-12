# Individual Fairness Testing for **COMPAS** data

The proposed test is applied on four different ML models fitted on COMPAS data: baseline, project, reduction and SenSR. 
Two protected attributes are considered: gender and race. The baseline  is obtained by simple 2 layer neural net. The projection is obtained by fitting neural net on the data, where the sensitive directions 
are projected out from feature vectors. Reduction approach consider the model enforcing group fairness in classification, 
described in [A reductions approach to fair classification](https://arxiv.org/abs/1803.02453) paper by Agarwal et al. 
SenSR enforces individual fairness corresponding to a specific sensitive subspace 
(see [Training individually fair ML models with Sensitive Subspace Robustness](https://arxiv.org/abs/1907.00020) for further details).

We summarize functionality of some important scripts below.

| Script | Functionality |
| --- | --- | 
| `compas_data.py` | Preprocessing of COMPAS data |
| `adv_ratio.py` | Calculates ratio of perturbed loss over original loss |
| `metrics.py` | Calculates popular fairness metrics |
| `summary.py` | Calculates fairness metrics and lower bound, and store them in `summary.out` file |
| `utils.py` | Contains utility functions such as entropy and accuracy | 
| `create_fluctuations.py` | Creates `.npy` file for ratio of perturbed loss and original loss. This is useful for submitting batch job |

A demo for single run for SenSR and reduction is presented in `demo.ipynb`.
