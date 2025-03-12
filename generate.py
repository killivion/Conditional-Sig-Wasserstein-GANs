import os

import torch

#from hyperparameters import SIGCWGAN_CONFIGS
from lib.algos.base import BaseConfig
from lib.arfnn import SimpleGenerator
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

    parser.add_argument('-dataset', default='correlated_Blackscholes', type=str)
    parser.add_argument('-algo', default='SigCWGAN', type=str)

    args = parser.parse_args()
    generate_data(args)

""" Concept:
1) One-time train the GAN
2) Evaluate for good performance

Sample data:
1) Load G in td3.train, so it's only loaded once
2) get paths of length windowsize q? (based on any past paths p - possibly shorten them or sample or smth. / perhaps load them in also and sample)
with that make sure that the drift and volatility are equal in GAN training and output specified for TD3
3) transform so they are of the right form for TD3
4) Train TD3 [construct if GAN_sampled -> different sample mechanism]
5) Eval TD3 ... 

"""