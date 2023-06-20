# `pac_dist_iforest`

The `pac_dist_iforest` is a an anomaly detection Python package. In the context of this project, 
two algorithms were developed. They are extensions of Isolation Forest. One uses a distribution based scoring function based on a probably approximately 
correct concept from computational learning theory instead of the depth as a measure for anomaly detection, and both use the Renyi divergences 
to aggregate the calculated scores for the anomaly detection.

# Installation: 

1. First create a new virtual environment for the installation process.
2. You need to install the packages provided in the requirement file with pip.
3. To verify the installation was succesfull, you can run the python file (examples.py)

Required python packages:
    - cfgv : 3.3.1
    - distlib : 0.3.6
    - filelock : 3.12.2
    - identify : 2.5.24
    - joblib : 1.2.0
    - nodeenv : 1.8.0
    - numpy : 1.25.0
    - platformdirs : 3.6.0
    - pre-commit : 3.3.3
    - PyYAML : 6.0
    - scikit-learn : 1.2.2
    - scipy : 1.10.1
    - threadpoolctl : 3.1.0
    - virtualenv : 20.23.1

# Linting:
- Black, mypy, pylint and pydocstyle were used to format the code. All the documented exceptions can be found in .pre-commit-config.yaml file.
- to check for linting issues run: pre-commit run --all-file 