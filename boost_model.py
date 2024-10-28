#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Define an adaptive transfer learning-based surrogate model for bayesian optimization'

__author__ = 'Gray Wildman'

from abc import abstractmethod
from pathlib import Path
import numpy as np

class BoostModel(object):

    def __init__(self, X_source, y_source, reg_source, default_temp):
        self._X_source, self._y_source = X_source, y_source
        self._n_source = len(y_source)
        self._sample_weight_source = np.ones(self._n_source) / self._n_source
        self._sample_weight_target = np.array([])
        self._input_dim = reg_source.n_features_in_
        self._default_temp = default_temp
        self._estimators = reg_source.estimators_
        self._estimator_weights = reg_source.estimator_weights_
        self._learning_rate = reg_source.learning_rate
        self._random_state = reg_source.random_state

    def transform_input(self, raw_X):
        if np.ndim(raw_X) == 1:
            raw_X = raw_X[np.newaxis, :]
        # 'raw_X.shape[-1] == 2' means [[q_harvard, q_xingda]]
        # 'raw_X.shape[-1] == 3' means [[q_harvard, q_xingda, temp]]
        # 'self._input_dim == 3' means [[q_harvard, q_xingda, temp]]
        if raw_X.shape[-1] == self._input_dim:
            return raw_X
        elif raw_X.shape[-1] == 2 and self._input_dim == 3:
            return np.hstack((raw_X, np.full((raw_X.shape[0], 1), self._default_temp)))
        else:
            raise ValueError(f'Cannot trasform length {len(raw_X)} to {self._input_dim}!')

    def fit(self, raw_X, y):
        X = self.transform_input(raw_X)
        num_new = len(y) - len(self._sample_weight_target)
        self._sample_weight_target = np.append(self._sample_weight_target, self._align_weight(num_new))
        _X = np.vstack((self._X_source, X))
        _y = np.append(self._y_source, y)
        sample_weight = np.append(self._sample_weight_source, self._sample_weight_target)
        sample_weight /= np.sum(sample_weight)
        rng = np.random.default_rng(self._random_state)
        for i, est in enumerate(self._estimators):
            bootstrap_idx = rng.choice(np.arange(_X.shape[0]), size=_X.shape[0], replace=True, p=sample_weight)
            est.fit(_X[bootstrap_idx], _y[bootstrap_idx])
            self._estimator_weights[i], sample_weight = self._update_weight(_X, _y, sample_weight, est)
            sample_weight_sum = np.sum(sample_weight)
            assert np.isfinite(sample_weight_sum) and sample_weight_sum > 0, 'The sum of sample weights must be positive and finite!'
            sample_weight /= sample_weight_sum
        self._sample_weight_source = sample_weight[:self._n_source]
        self._sample_weight_target = sample_weight[self._n_source:]
        return self

    @abstractmethod
    def _align_weight(self, num_new):
        pass

    @abstractmethod
    def _update_weight(self, _X, _y, sample_weight, est):
        pass

    def predict(self, raw_X, return_std=False):
        X = self.transform_input(raw_X)
        predictions = np.array([est.predict(X) for est in self._estimators]).T
        # Compute weighted median
        sorted_idx = np.argsort(predictions, axis=1)
        weight_cdf = np.cumsum(self._estimator_weights[sorted_idx], axis=1)
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        median_idx = median_or_above.argmax(axis=1)
        median_estimators = sorted_idx[np.arange(X.shape[0]), median_idx]
        y = predictions[np.arange(X.shape[0]), median_estimators]
        # Compute weighted standard deviation
        y_avg = np.average(predictions, axis=1, weights=self._estimator_weights)
        y_var = np.average((predictions - y_avg.reshape(-1, 1)) ** 2, axis=1, weights=self._estimator_weights)
        return (y, np.sqrt(y_var)) if return_std else y

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self._regressor, key, value)
        return self
    
    def debug_print(self, text):
        abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
        with open(abs_dict / 'debug/bayes_debug.txt', 'w') as f: # output file for debugging in LabVIEW
            print(text, file=f)
        return

class AdaBoostModel(BoostModel):

    def __init__(self, X_source, y_source, reg_source, default_temp=27):
        super().__init__(X_source, y_source, reg_source, default_temp)

    def _align_weight(self, num_new):
        sample_weight = np.append(self._sample_weight_source, self._sample_weight_target)
        return np.average(sample_weight) * np.ones(num_new)

    def _update_weight(self, _X, _y, sample_weight, est):
        error_vect = np.abs(est.predict(_X) - _y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
        error_max = masked_error_vector.max()
        assert error_max > 0, f'Max error must be positive! Current max error: {error_max:.4f}'
        masked_error_vector = (masked_error_vector / error_max) ** 2
        estimator_error = (masked_sample_weight * masked_error_vector).sum()
        assert estimator_error > 0, f'The learner is too perfect! Current error: {estimator_error:.4f}'
        assert estimator_error < 0.5, f'The learner is too weak! Current error: {estimator_error:.4f}'
        beta = estimator_error / (1.0 - estimator_error)
        estimator_weight = self._learning_rate * np.log(1.0 / beta)
        sample_weight[sample_mask] *= self._learning_rate * np.power(beta, (1.0 - masked_error_vector))
        return estimator_weight, sample_weight

class TrAdaBoostModel(BoostModel):

    def __init__(self, X_source, y_source, reg_source, target_init_weight_ratio=0.3, target_error_weight=0.7, default_temp=27):
        super().__init__(X_source, y_source, reg_source, default_temp)
        self._target_init_weight_ratio = target_init_weight_ratio
        self._target_error_weight = target_error_weight

    def _align_weight(self, num_new):
        if len(self._sample_weight_target):
            return np.average(self._sample_weight_target) * np.ones(num_new)
        else:
            return np.sum(self._sample_weight_source) * self._target_init_weight_ratio

    def _update_weight(self, _X, _y, sample_weight, est):
        error_vect = np.abs(est.predict(_X) - _y)
        sample_mask = sample_weight > 0
        masked_sample_weight = sample_weight[sample_mask]
        masked_error_vector = error_vect[sample_mask]
        error_max = masked_error_vector[self._n_source:].max()
        assert error_max > 0, f'Max error must be positive! Current max error: {error_max:.4f}'
        masked_error_vector = (masked_error_vector / error_max) ** 2
        estimator_error_vector = masked_sample_weight * masked_error_vector
        masked_sample_weight_source = masked_sample_weight[:self._n_source] / masked_sample_weight[:self._n_source].sum()
        masked_sample_weight_target = masked_sample_weight[self._n_source:] / masked_sample_weight[self._n_source:].sum()
        estimator_error_source = sum(estimator_error_vector[:self._n_source] * masked_sample_weight_source)
        estimator_error_target = sum(estimator_error_vector[self._n_source:] * masked_sample_weight_target)
        estimator_error = estimator_error_source * (1.0 - self._target_error_weight) + estimator_error_target * self._target_error_weight
        assert estimator_error > 0, f'The learner is too perfect! Current error: {estimator_error:.4f}' 
        assert estimator_error < 0.5, f'The learner is too weak! Current error: {estimator_error:.4f}'
        beta = estimator_error / (1.0 - estimator_error)
        beta[:self._n_source] = 1 / (1 + (2 * np.log(self._n_source / len(self._estimators))) ** 0.5 )
        estimator_weight = self._learning_rate * np.log(1.0 / beta)
        sample_weight_factor = np.power(beta, (1.0 - masked_error_vector))
        sample_weight_factor[:self._n_source] = 1.0 / sample_weight_factor[:self._n_source]
        sample_weight[sample_mask] *= self._learning_rate * sample_weight_factor
        return estimator_weight, sample_weight