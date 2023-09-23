# Cyclicity Analysis of Time-Series
This repository contains a working implementation of Cyclicity Analysis, which is a pattern recognition technique for analyzing the leader follower dynamics of multiple time-series.

Full documentation and an example Jupyter notebook are available in the [GitHub repository](https://github.com/vskaush2/cyclicityanalysis).

## Requirements
Download [Python >=3.10](https://www.python.org/downloads/)

## Installation

```bash
pip3 install cyclicityanalysis
```

## Usage

```python
from cyclicityanalysis.orientedarea import *
from cyclicityanalysis.coom import *

df = pd.DataFrame([[0, 1], [1, 0], [0, 0]], columns=['0', '1'])


oa = OrientedArea(df)
# Returns the lead lag matrix of df as a dataframe
lead_lag_df = oa.compute_lead_lag_df()

coom = COOM(lead_lag_df)

# Returns leading eigenvector of the lead lag matrix as an array, the leading eigenvector component phases as an array,
# and sequential order of the lead lag matrix according to COOM as a dictionary 
leading_eigenvector, leading_eigenvector_component_phases, sequential_order_dict = coom.compute_sequential_order(0)
lead_lag_df , leading_eigenvector, leading_eigenvector_component_phases, sequential_order_dict
 ```

## References 

* Cyclicity in Multivariate Time-series and Applications to Functional MRI data : [paper](https://ieeexplore.ieee.org/document/7798498)
* Dissociating Tinnitus Patients from Healthy Controls using Resting-state Cyclicity Analysis and Clustering : [paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6326732/)
* Slow Cortical Waves through Cyclicity Analysis : [paper](https://www.biorxiv.org/content/10.1101/2021.05.16.444387v2.full)
* Comparing Cyclicity Analysis With Pre-established Functional Connectivity Methods to Identify Individuals and Subject Groups Using Resting State fMRI : [paper](https://www.frontiersin.org/article/10.3389/fncom.2019.00094/full)


