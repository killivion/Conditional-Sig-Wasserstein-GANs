import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

from hyperparameters import SIGCWGAN_CONFIGS
from lib.algos.base import BaseConfig
from lib.algos.base import is_multivariate
from lib.algos.sigcwgan import calibrate_sigw1_metric, sample_sig_fake
from lib.algos.sigcwgan import sigcwgan_loss
from lib.arfnn import SimpleGenerator
from lib.plot import plot_summary, compare_cross_corr
from lib.test_metrics import test_metrics
from lib.utils import load_pickle, to_numpy


def generate_from_generator(experiment_dir, dataset, use_cuda=True):
    torch.random.manual_seed(0)
    device = 'cuda' if use_cuda else 'cpu'

    #sig_config = get_algo_config(dataset, experiment_dir)
    base_config = BaseConfig(device=device)
    p, q = base_config.p, base_config.q
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = load_pickle(os.path.join(os.path.dirname(experiment_dir), 'x_real_test.torch')).to(device)
    print(x_real)
    x_past = x_real[:, :p]
    #x_future = x_real[:, p:p + q]
    dim = x_real.shape[-1]
    # ----------------------------------------------
    # Load generator weights and hyperparameters
    # ----------------------------------------------
    G_weights = load_pickle(os.path.join(experiment_dir, 'G_weights.torch'))
    G = SimpleGenerator(dim * p, dim, 3 * (50,), dim).to(device)
    G.load_state_dict(G_weights)
    # ----------------------------------------------
    # generate fake paths
    # ----------------------------------------------
    with torch.no_grad():
        _x_past = x_past.clone()
        x_fake_future = G.sample(q, _x_past)
    print(f'x_fake_future: {x_fake_future}')

    return x_fake_future


def generate_data(args):
    #algo_path = os.path.join(args.base_dir, args.dataset, experiment_dir, seed_dir, args.algo)
    spec = 'mu=[0.06]_sigma=[[0.4472136]]_window_size=1000'
    algo_path = f'./numerical_results/{args.dataset}/{spec}/seed=42/{args.algo}'
    print(algo_path)
    generate_from_generator(experiment_dir=algo_path, dataset=args.dataset, use_cuda=True)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Turn cuda off / on during evalution.')
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-datasets', default=['correlated_Blackscholes'], nargs="+")  # , 'STOCKS', 'ECG', 'VAR',
    parser.add_argument('-algos', default=['SigCWGAN'], nargs="+") #, 'GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN',

    parser.add_argument('-dataset', default='correlated_Blackscholes', type=str)
    parser.add_argument('-algo', default='SigCWGAN', type=str)

    args = parser.parse_args()
    generate_data(args)
