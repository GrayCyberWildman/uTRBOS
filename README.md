# uTRBOS
Source code for the algorithms implemented in the uTRBOS project, including peak analysis, TRBO, and control group algorithms.

## Overview 
This section details the functionality of each script.
- **peaks.py** conducts real-time analysis of photoluminescence spectra and visualizes the results.
- **boost_model_pretrain.py** prepares a customized surrogate model for AdaBoost.R2 or TRBO, utilizing source domain datasets. The pretrained model is stored in the **bayes** subdirectory.
- **boost_model.py** defines the customized surrogate model used by **boost_model_pretrain.py**.
- **bayes.py** and **cmaes.py** execute iterative optimization of synthetic conditions, utilizing models and parameters stored in the **bayes** or **cmaes** subdirectory.

Note that all scripts, except for **boost_model_pretrain.py** and **boost_model.py**, are designed to interface directly with LabVIEW connectivity nodes in VI scripts through function names.

## Usage
This section describes the usage of each script.
### peaks.py
To perform analysis, invoke the `analyze()` function with a vectorized photoluminescent spectrum. The function will print and plot the height, center, and FWHM of all detected peaks. 
### boost_model_pretrain.py & boost_model.py
To pretrain a customized surrogate model with a source domain dataset, run the following command for argument details:
```python
python boost_model_pretrain.py -h
```
### bayes.py & cmaes.py
To execute iterative optimization with AdaBoost.R2 or TRBO, run **boost_model_pretrain.py** to generate a surrogate model first, and then set parameters in **opt_para.json** and **target_para.csv** located in either **bayes** or **cmaes** subdirectory. To execute iterative optimization with CMA-ES or standard BO, just set parameters.  
Subsequently, iteratively invoke the `make_decision()` function with an evaluated synthetic condition and the corresponding experimental result. The input will be recorded to conduct an optimization step, and a new condition will be recommended once the step is completed.

## Dependencies
- **All scripts**
  - Python >=3.8
  - numpy
  - scipy
  - pandas
- **peaks.py**
  - matplotlib
- **boost_model_pretrain.py & boost_model.py**
  - scikit-learn
- **cmaes.py**
  - cma >=3.3.0  
https://github.com/CMA-ES/pycma
- **bayes.py**
  - bayes_opt >=1.4.0, <2.0.0  
https://github.com/bayesian-optimization/BayesianOptimization
 