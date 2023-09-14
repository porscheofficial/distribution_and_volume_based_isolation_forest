**This branch contains code of the project that didn't make it into the final project. In particular it contains a branch and bound implementation and a mixed integer nonlinear programming approach to implementing an exact HAS_RARE_PATTERN subroutine that provides an alternative to the isolation forest random forest procedure. This turned out to be performing poorly, we leave it here for the interested reader to build on.**

**Moreover, the requirements from the requirements.txt might not suffice to make the code on this branch work**

# Reproduction instructions

This directory contains the code required to reproduce the results presented in the submission "Distribution and volume based scoring for Isolation Forests".

## Setup

The benchmarks are run using the [_**ADBench**: Anomaly Detection Benchmark_](https://arxiv.org/abs/2206.09426). Hence users are required to set up that benchmark. However, since that benchmark is quite large and we don't require all the packages required in that repository, we ask users to follow the following custom setup instructions

1. Clone the [Github repository](https://github.com/Minqi824/ADBench)
2. Copy this folder into the root directory of the cloned ADBench repo
3. Create a virtual environment (we used Python version 3.9.6) and install the requirements from the `requirements.txt` contained in this folder.

You can then execute the two notebooks required for generating results and plots from the paper as usual.
