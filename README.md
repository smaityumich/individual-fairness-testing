# Statistical Inference for Individual Fairness

This is the supplimentary code acompanying NeurIPS 2020 submission "Statistical inference for individual fairness"

## Overview

The work proposes an inferential procedure to test for violation of individual fairness for a ML model. The main idea is to perform an adversarial attack on each of the data-points aiming to increase the loss, while restricting the movement in sensitve subspace. The average of loss ratio between attacked points and original points tend to be large if the ML model is unfair. This property is used in the proposed test. We use finite time gradient flow for adversarial attack, where the continuous time gradient flow is approximated by Euler's method. 


We perform simulated experiment to study the properties of proposed test and apply it to *Adult* data. More descriptions are provided in corresponding folders. 