#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Pretrain an adaptive transfer learning-based surrogate model for the synthesis of quantum dots'

__author__ = 'Gray Wildman'

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from boost_model import AdaBoostModel, TrAdaBoostModel

def calculate_taget(sample, para):
    if sample['n_peaks'] == 0 or sample['n_peaks'] >= 4:
        return -99999
    else:
        zipped = [(sample[f'height{x + 1}'], sample[f'center{x + 1}'], sample[f'hlf{x + 1}']) for x in range(sample['n_peaks'])]
        height, center, hlf = zip(*zipped)
        area = [1.0645 * x * y for x, y in zip(height, hlf)]
        sp, scaling, weight = np.array(para['sp']), np.array(para['scaling']), np.array(para['weight'])
        score = np.sum(np.abs(np.array((height, center, hlf)).T - sp) / scaling * weight, axis=-1)
        return np.average(score, weights=area)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', '--name', type=str, default='test', help='the name of the model')
parser.add_argument('-tp', '--target_path', type=str, default='bayes/target_para.csv',
                    help='the path of the csv file for target parameters')
parser.add_argument('-md', '--max_depth', type=int, default=4,
                    help='the maximum depth of decesion tree regressors as base learners')
parser.add_argument('-nl', '--n_learners', type=int, default=25, help='the number of base learners')
parser.add_argument('-tr', '--transfer', action='store_true', help='use TRBO instead of AdaBoost.R2')

if __name__ == '__main__':
    args = parser.parse_args()
    print('Preparing dataset...')
    data = pd.read_excel('source_domain_dataset.xlsx', sheet_name='dataset')
    para = pd.read_csv(args.target_path, index_col=0)
    data['target'] = data.apply(calculate_taget, axis=1, args=(para, ))
    print(data)
    X, y = data[['q_harvard', 'q_xingda', 'temp']], data['target']
    print('Building and fitting source regressor...')
    base_learner = DecisionTreeRegressor(max_depth=args.max_depth)
    reg_source = AdaBoostRegressor(estimator=base_learner, n_estimators=args.n_learners)
    reg_source.fit(X, y)
    print(f'The initial training R2 is: {reg_source.score(X, y)}')
    assert len(reg_source.estimators_) == len(reg_source.estimator_weights_), f'Early stop happened at estimator {len(reg_source.estimator_weights_)}!'
    print('Building surrogate model...')
    model = TrAdaBoostModel(X, y, reg_source) if args.transfer else AdaBoostModel(X, y, reg_source)
    with open(f'bayes/model_{args.name}.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'The surrogate model has been saved to: ./bayes/model_{args.name}.pkl')