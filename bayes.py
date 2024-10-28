#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'A bayesian optimization framework for the synthesis of quantum dots'

__author__ = 'Gray Wildman'

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization, UtilityFunction

def calculate_taget(height, center, hlf, area):
    if height[0] < 0 or center[0] < 0 or hlf[0] < 0:
        return -99999
    else:
        abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
        para = pd.read_csv(abs_dict / 'bayes/target_para.csv', index_col=0)
        sp, scaling, weight = np.array(para['sp']), np.array(para['scaling']), np.array(para['weight'])
        score = np.sum(np.abs(np.array((height, center, hlf)).T - sp) / scaling * weight, axis=-1)
        return np.average(score, weights=area)

def make_decision(q_harvard, q_longer_1, q_longer_2, q_xingda, temp, height, center, hlf, area):
    abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
    with open(abs_dict / 'bayes/opt_para.json', 'r') as f:
        para = json.load(f)
    utility = UtilityFunction(**para['utility'])
    if center is None or center[0] == 0:
        #init_optimizer
        optimizer = BayesianOptimization(
            f=None,
            pbounds=para['bounds'],
            verbose=2,
            random_state=para['random_state']
        )
        #init_gp_model
        if para['init_gp_model_path'] is not None:
            with open(abs_dict / para['init_gp_model_path'], 'rb') as f:
                optimizer._gp = pickle.load(f)
        #init_data
        current_data = {'q_harvard': 0, 'q_longer_1': 0,
                        'q_longer_2': 0, 'q_xingda': 0, 'temp': temp}
        #init_acquisition
        next_point = optimizer.suggest(utility)
        with open(abs_dict / 'debug/bayes_debug.txt', 'w') as f: # output file for debugging in LabVIEW
            print(f'next_point: {next_point}', file=f)
    else:
        #load_optimizer
        with open(abs_dict / 'bayes/optimizer.pkl', 'rb') as f:
            optimizer = pickle.load(f)
        #update_data
        current_data = {'q_harvard': q_harvard, 'q_longer_1': q_longer_1,
                        'q_longer_2': q_longer_2, 'q_xingda': q_xingda, 'temp': temp}
        prev_point = {var: current_data[var] for var in para['bounds']}
        #update_optimizer
        target = calculate_taget(height, center, hlf, area)
        optimizer.register(params=prev_point, target=target)
        #acquisition
        next_point = optimizer.suggest(utility)
        with open(abs_dict / 'debug/bayes_debug.txt', 'a') as f: # output file for debugging in LabVIEW
            print(f'target: {target}', file=f)
            print(f'next_point: {next_point}', file=f)
    with open(abs_dict / 'bayes/optimizer.pkl', 'wb') as f:
        pickle.dump(optimizer, f)
    for var in current_data:
        if var in para['bounds']:
            current_data[var] = next_point[var]
    return [current_data['q_harvard'], current_data['q_longer_1'], current_data['q_longer_2'],
            current_data['q_xingda'], current_data['temp']] # [] for LabVIEW compatibility

if __name__ == '__main__':
    result = make_decision(None, None, None, None, None, None, None, None, None)
    q_harvard, q_longer_1, q_longer_2, q_xingda, temp = 100, 5000, 0, 1000, 25
    height, center, hlf, area = [2000, ], [550, ], [20, ], [30120, ]
    result = make_decision(q_harvard, q_longer_1, q_longer_2, q_xingda, temp, height, center, hlf, area)
    print(f'Result: {result}')