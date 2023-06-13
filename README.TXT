# `pac_dist_iforest`

The `pac_dist_iforest` is a an anomaly detection Python package. In the context of this project, 
two algorithms were developed. They are extensions of Isolation Forest. One uses a distribution based scoring function based on a probably approximately 
correct concept from computational learning theory instead of the depth as a measure for anomaly detection, and both use the Renyi divergences 
to aggregate the calculated scores.

# Installation 

1. First create a new virtual environment for the installation process 
2. You need to install the packages provided in the requirement files (you can use either pip or conda).
3. To verify the installation was succesfull, you can run teh python file (script-test-installation.py) and have a look at the output plot. 
It should be similar to this one: ![succesfull installation](renyi_isolation_forest/images/test-out.png)