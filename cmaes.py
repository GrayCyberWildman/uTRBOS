#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'A cma-es optimization framework for the synthesis of quantum dots'

__author__ = 'Gray Wildman'

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from cma.evolution_strategy import CMAEvolutionStrategy

def calculate_taget(height, center, hlf, area):
    if height[0] < 0 or center[0] < 0 or hlf[0] < 0:
        return -99999
    else:
        abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
        para = pd.read_csv(abs_dict / 'cmaes/target_para.csv', index_col=0)
        sp, scaling, weight = np.array(para['sp']), np.array(para['scaling']), np.array(para['weight'])
        score = np.sum(np.abs(np.array((height, center, hlf)).T - sp) / scaling * weight, axis=-1)
        return np.average(score, weights=area)

def cal_scale(input_array, bound_dict, reverse=False):
    lower_bound, upper_bound = list(zip(bound_dict['q_harvard'], bound_dict['q_xingda'], bound_dict['temp']))
    lb_array = np.array(lower_bound)
    ub_array = np.array(upper_bound)
    if not reverse:
        output_array = (input_array - lb_array) / (ub_array - lb_array)
    else:
        output_array = input_array * (ub_array - lb_array) + lb_array
    return output_array

def make_decision(q_harvard, q_longer_1, q_longer_2, q_xingda, temp, height, center, hlf, area):
    abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
    with open(abs_dict / 'cmaes/opt_para.json', 'r') as f:
        para = json.load(f)
    if center is None or center[0] == 0:
        #init_optimizer
        optimizer = CMAEvolutionStrategy(
            [0.50, 0.50, 0.50],
            0.15,
            {
                'bounds': [[0, 0, 0], [1, 1, 1]],
                'seed': para['random_state'],
                'popsize': para['batch_size'],
                'verb_filenameprefix': str(abs_dict / para['log_path']) + '\\'
            }
        )
        #init_test_queue
        test_queue = {'to_test': optimizer.ask(),
                        'tested':[], 'target':[]}
        #init_data
        q_longer_1 = 0
        q_longer_2 = 0
        with open(abs_dict / 'cmaes/optimizer.pkl', 'wb') as f:
            f.write(optimizer.pickle_dumps())
    else:
        #load_test_queue
        with open(abs_dict / 'cmaes/test_queue.pkl', 'rb') as f:
            test_queue = pickle.load(f)
        #update_test_queue
        prev_cal_point = cal_scale(np.array([q_harvard, q_xingda, temp]), para['bounds'])
        target = calculate_taget(height, center, hlf, area)
        test_queue['tested'].append(prev_cal_point)
        test_queue['target'].append(target)
        with open(abs_dict / 'debug/cmaes_debug.txt', 'a') as f: # output file for debugging in LabVIEW
            print(f'target: {target}', file=f)
        if len(test_queue['to_test']) == 0:
            #call_optimizer
            with open(abs_dict / 'cmaes/optimizer.pkl', 'rb') as f:
                optimizer = pickle.load(f)
            assert len(test_queue['tested']) == optimizer.popsize, 'Numeric Error!'
            assert len(test_queue['target']) == optimizer.popsize, 'Numeric Error!'
            optimizer.tell(test_queue['tested'], test_queue['target'])
            optimizer.logger.add()
            test_queue['to_test'] = optimizer.ask()
            test_queue['tested'] = []
            test_queue['target'] = []
            with open(abs_dict / 'cmaes/optimizer.pkl', 'wb') as f:
                f.write(optimizer.pickle_dumps())
    print(f'\ntest_queue: {test_queue}\n')
    next_cal_point = test_queue['to_test'].pop(0)
    next_exp_point = cal_scale(next_cal_point, para['bounds'], reverse=True)
    with open(abs_dict / 'debug/cmaes_debug.txt', 'a') as f: # output file for debugging in LabVIEW
        print(f'next_point: {next_exp_point}', file=f)
    with open(abs_dict / 'cmaes/test_queue.pkl', 'wb') as f:
        pickle.dump(test_queue, f)
    return [next_exp_point[0], q_longer_1, q_longer_2,
            next_exp_point[1], next_exp_point[2]] # [] for LabVIEW compatibility

if __name__ == '__main__':
    result = make_decision(None, None, None, None, None, None, None, None, None)
    for i in range(10):
        print(f'------------Exp {i + 1}------------')
        print(f'Cond: {result}')
        q_harvard, q_longer_1, q_longer_2, q_xingda, temp = result
        if i == 4:
            height, center, hlf = [-1,], [-1,], [-1,]
        elif i == 8:
            height, center, hlf = [-2, -2, -2, -2], [440, 460, 480, 500], [-2, -2, -2, -2]
        else:
            height = [1000, 1000] + 2000 * np.random.rand(2)
            center = [450, 490] + 30 * np.random.rand(2)
            hlf = [15, 15] + 10 * np.random.rand(2)
        area = [0.753 * he * hl for he, hl in zip(height, hlf)]
        print(f'Height: {height}\nCenter: {center}\nHlf: {hlf}\nArea: {area}')
        result = make_decision(q_harvard, q_longer_1, q_longer_2, q_xingda, temp, height, center, hlf, area)