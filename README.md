# uTRBOS
Source code for the algorithms implemented in the uTRBOS project, including peak analysis, AdaBoost.R2 and TRBO.

## Overview 
This section details the functionality of each script.
- **peaks.py** conducts real-time analysis of photoluminescence spectra and visualizes the results.
- **boost_model_pretrain.py** prepares a customized surrogate model for Bayesian optimization, utilizing source domain datasets. The pretrained model is stored in the **bayes** subdirectory.
- **boost_model.py**: Defines the surrogate model (AdaBoost.R2 or TRBO) used by **boost_model_pretrain.py**.
- **bayes.py** and **cmaes.py** execute iterative optimization of synthetic conditions, utilizing models and parameters stored in the **bayes** or **cmaes** subdirectory.

Note that all scripts, except for **boost_model_pretrain.py** and **boost_model.py**, are designed to interface directly with LabVIEW connectivity nodes in VI scripts through function names.

## Usage
This section describes the usage of each script.
### peaks.py
To perform analysis, invoke the `analyze()` function with a vectorized photoluminescent spectrum. The function will print and plot the height, center, and FWHM of all detected peaks. 
### boost_model_pretrain.py & boost_model.py
To pretrain a surrogate model with a source domain dataset, run the following command for argument details:
```python
python boost_model_pretrain.py -h
```
### bayes.py & cmaes.py
To execute iterative optimization, run **boost_model_pretrain.py** to generate a surrogate model first, and then set parameters in **opt_para.json** and **target_para.csv** located in either **bayes** or **cmaes** subdirectory.  
Subsequently, iteratively invoke the `make_decision()` function with an evaluated synthetic condition and the corresponding experimental result. The condition will be recorded, and a new condition will be recommended for the next optimization step.

## Dependencies
- **All scripts**  
  - numpy
  - scipy
  - pandas
- **peaks.py**  
  - matplotlib
- **boost_model_pretrain.py & boost_model.py**  
  - scikit-learn
- **cma-es.py**  
  - cma >=3.3.0  
https://github.com/CMA-ES/pycma
- **boost_model_pretrain.py & boost_model.py**  
  - bayes_opt >=1.4.0, <2.0.0  
https://github.com/bayesian-optimization/BayesianOptimization
 