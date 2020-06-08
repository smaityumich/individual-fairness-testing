# Statistical Inference for Individual Fairness

This is the supplimentary code acompanying NeurIPS 2020 submission "Statistical inference for individual fairness"

## Overview

The work proposes an inferential procedure to test for violation of individual fairness for a ML model. The main idea is to perform an adversarial attack on each of the data-points aiming to increase the loss, while restricting the movement in sensitve subspace. The average loss ratio $`E[\ell(f(X_{\text{attacked}}), Y)/\ell(f(X), y)]`$
