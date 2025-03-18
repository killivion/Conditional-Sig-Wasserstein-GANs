import os

import torch
import pandas as pd
import numpy as np

#from hyperparameters import SIGCWGAN_CONFIGS
from lib.algos.base import BaseConfig
from lib.arfnn import SimpleGenerator
from lib.utils import load_pickle, to_numpy


def generate_from_generator(spec, experiment_dir, dataset, use_cuda=True):
    torch.random.manual_seed(0)
    device = 'cuda' if use_cuda else 'cpu'

    #sig_config = get_algo_config(dataset, experiment_dir)
    base_config = BaseConfig(device=device)
    p, q = args.sig_p, args.sig_q
    # ----------------------------------------------
    # Load and prepare real path.
    # ----------------------------------------------
    x_real = load_pickle(os.path.join(os.path.dirname(experiment_dir), 'x_real_test.torch')).to(device)
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

    return x_fake_future


def generate_data(spec, args):
    #algo_path = os.path.join(args.base_dir, args.dataset, experiment_dir, seed_dir, args.algo)
    algo_path = f'numerical_results/{args.dataset}/{spec}/seed=42/{args.algo}'

    x_fake_future = generate_from_generator(spec=spec, experiment_dir=algo_path, dataset=args.dataset, use_cuda=True)

    stats = torch.load(f'./numerical_results/{args.dataset}/{spec}/seed=42/meanstd.pt', weights_only=True)
    real_mean, reaL_std = stats['mean'], stats['std']

    # Reverse the scaling transformation
    from lib.data import Pipeline, StandardScalerTS
    pipeline = Pipeline(steps=[('standard_scale', StandardScalerTS(axis=(0, 1)))])
    logrtn_recovered = pipeline.inverse_transform(x=x_fake_future, real_mean=real_mean, real_std=reaL_std)
    logrtn_recovered = logrtn_recovered.detach().cpu().numpy() if isinstance(logrtn_recovered, torch.Tensor) else logrtn_recovered
    logrtn_recovered = logrtn_recovered.squeeze(-1)
    log_prices_reconstructed = np.cumsum(logrtn_recovered, axis=1)
    price_paths_reconstructed = np.exp(log_prices_reconstructed)
    price_paths_reconstructed = np.insert(price_paths_reconstructed, 0, 1)

    return pd.DataFrame(price_paths_reconstructed)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Turn cuda off / on during evalution.')
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')

    parser.add_argument('-dataset', default='correlated_Blackscholes', type=str)
    parser.add_argument('-algo', default='SigCWGAN', type=str)
    parser.add_argument('-spec', default='mu=[0.06]_sigma=[[0.4472136]]_window_size=1000', type=str)
    parser.add_argument('-sig_p', default=3, type=int)
    parser.add_argument('-sig_q', default=3, type=int)

    args = parser.parse_args()

    results = []
    from tqdm import tqdm
    for _ in tqdm(range(10000), desc="YFinance", leave=False):
        df = generate_data(args.spec, args)
        # Extract the last row (i.e. the final values) from the DataFrame
        last_row = df.iloc[-1]
        results.append(last_row)

    # Create a DataFrame from the collected rows
    results_df = pd.DataFrame(results)

    # Compute mean and standard deviation for each column
    mean_values = results_df.mean()
    std_values = results_df.std()

    print("Mean values over 10,000 samples:")
    print(mean_values)
    print("\nStandard deviations over 10,000 samples:")
    print(std_values)

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