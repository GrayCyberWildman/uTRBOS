#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'Estimate the peaks together with their heights, widths at half maxima and areas of a PL spectrum curve'

__author__ = 'Gray Wildman'

from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, peak_widths
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gauss(x, a, b, c):
    return a * np.exp(-(x - b) ** 2 / (2 * c ** 2))

def gauss_fit_with_center(wavelength, seq, center):
    num = len(center)
    p0, bounds = [1000, 10] * num, ([0, 0] * num, [np.inf, 15] * num)
    func = lambda x, *p: sum([gauss(x, p[2 * i], center[i], p[2 * i + 1]) for i in range(num)])
    popt, _ = curve_fit(func, wavelength, seq, p0=p0, bounds=bounds)
    return popt[0::2], [2.355 * x for x in popt[1::2]]

def visualize_fit(title, wavelength, seq, height, center, hlf):
    xx = np.linspace(350, 750, 1001)
    plt.switch_backend('agg') # non-GUI backend for LabVIEW compatibility
    plt.plot(wavelength, seq, label='original (-base)')
    fit = []
    for i, (he, ce, hl) in enumerate(zip(height, center, hlf)):
        fit.append(gauss(xx, he, ce, hl / 2.355))
        plt.plot(xx, fit[-1], linestyle='--', label='gauss' + str(i + 1))
    plt.plot(xx, sum(fit), label='fit')
    plt.legend()
    plt.xlabel('wavelength/nm')
    plt.title(title)
    abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
    plt.savefig(abs_dict / 'fit_temp.png', bbox_inches='tight')
    return fit

def analyze(interval, fit_interval, seq, pm=50):
    assert int(interval[2]) == len(seq)
    wavelength, step = np.linspace(interval[0], interval[1], num=int(interval[2]), retstep=True)
    start_i, end_i = ((fit_interval - interval[0]) / step).astype(int)
    fit_wavelength, fit_seq = wavelength[start_i: end_i], seq[start_i: end_i]
    fit_seq = fit_seq - min(fit_seq)
    pks, _ = find_peaks(fit_seq, prominence=pm)
    center = [fit_wavelength[x] for x in pks]
    if len(pks) == 0:
        center, height, hlf = [-1,], [-1,], [-1,]
    elif len(pks) >= 4:
        height, hlf = [-2 for x in pks], [-2 for x in pks]
    elif len(pks) == 1:
        height, hlf = [fit_seq[x] for x in pks], peak_widths(fit_seq, pks)[0]
        visualize_fit('Peak Analysis', fit_wavelength, fit_seq, height, center, hlf)
    elif len(pks) in [2, 3]:
        height, hlf = gauss_fit_with_center(fit_wavelength, fit_seq, center)
        visualize_fit('Peak Analysis', fit_wavelength, fit_seq, height, center, hlf)
    area = [1.0645 * x * y for x, y in zip(height, hlf)]
    abs_dict = Path(__file__).parent.resolve() # absolute path for LabVIEW compatibility
    with open(abs_dict / 'debug/peaks_debug.txt', 'w') as f: # output file for debugging in LabVIEW
        print(f'fit_wavelength:\n{fit_wavelength}', file=f)
        print(f'fit_seq:\n{fit_seq}', file=f)
        print(f'height:\n{height}', file=f)
        print(f'center:\n{center}', file=f)
        print(f'hlf:\n{hlf}', file=f)
        print(f'area:\n{area}', file=f)
    return np.array([height, center, hlf, area]) # array for LabVIEW compatibility

if __name__ == '__main__':
    interval, fit_interval = [300, 812, 1024], np.array([400, 700])
    wavelength = np.linspace(interval[0], interval[1], num=int(interval[2]))
    seq = gauss(wavelength, 1000, 440, 12) + gauss(wavelength, 2400, 480, 12)
    height, center, hlf, area = analyze(interval, fit_interval, seq)
    for i, (he, ce, hl, ar) in enumerate(zip(height, center, hlf, area)):
        print(f'The No.{i + 1} peak has been found at wavelength {ce:.2f}nm.',
            f'Its height = {he:.2f}, width = {hl:.2f}nm and area = {ar:.2f}.')
