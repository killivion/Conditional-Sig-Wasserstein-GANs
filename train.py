import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from hyperparameters import SIGCWGAN_CONFIGS
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import download_man_ahl_dataset, download_mit_ecg_dataset
from lib.data import get_data
from lib.plot import savefig, create_summary
from lib.utils import pickle_it
from TD3.data_generator import generate_random_params, heston_params


def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    return SIGCWGAN_CONFIGS[key]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id == 'SigCWGAN':
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def run(args, algo_id, base_config, base_dir, dataset, spec, data_params={}):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    # Set seed for exact reproducibility of the experiments
    set_seed(base_config.seed)
    # initialise dataset and algo
    x_real = get_data(dataset, args.p, args.q, isSigLib=True, spec=spec, **data_params)
    x_real = x_real.to(base_config.device)
    ind_train = int(x_real.shape[0] * 0.8)
    x_real_train, x_real_test = x_real[:ind_train], x_real[ind_train:] #train_test_split(x_real, train_size = 0.8)

    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    # Train the algorithm
    algo.fit()
    # create summary
    create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_real_test)
    savefig('summary.png', experiment_directory)
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, 8000, x_real_test, one=True)
    savefig('summary_long.png', experiment_directory)
    plt.plot(x_fake.cpu().numpy()[0, :2000])
    savefig('long_path.png', experiment_directory)
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_real, pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(x_real_test, pt.join(pt.dirname(experiment_directory), 'x_real_test.torch'))
    pickle_it(x_real_train, pt.join(pt.dirname(experiment_directory), 'x_real_train.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)


def get_dataset_configuration(dataset, window_size, num_paths, num_bm, grid_points, q):
    if dataset == 'ECG':
        generator = [('id=100', dict(filenames=['100']))]
    elif dataset == 'STOCKS':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('SPX',), ('SPX', 'DJI')])
    elif dataset == 'VAR':
        par1 = itertools.product([1], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)])
        par2 = itertools.product([2], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        par3 = itertools.product([3], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        combinations = itertools.chain(par1, par2, par3)
        generator = (
            ('dim={}_phi={}_sigma={}_window_size={}'.format(dim, phi, sigma, window_size), dict(dim=dim, phi=phi, sigma=sigma, window_size=window_size, num_paths=num_paths))
            for dim, (phi, sigma) in combinations
        )
    elif dataset == 'ARCH':
        generator = (('lag={}_window_size={}'.format(lag, window_size), dict(lag=lag, window_size=window_size, num_paths=num_paths)) for lag in [3])
    elif dataset == 'SINE':
        generator = [('a', dict())]
    elif dataset == 'Blackscholes':
        generator = (('mu={}_sigma={}_window_size={}'.format(mu, sigma, window_size), dict(data_params=dict(mu=mu, sigma=sigma, window_size=window_size, num_paths=num_paths, grid_points=grid_points)))
                     for mu, sigma in [(0.06, 0.2)]
        )
    elif dataset == 'correlated_Blackscholes':
        param_mu, param_vola_matrix = generate_random_params(num_paths, num_bm)
        generator = (('mu={}_sigma={}_q={}'.format(mu, vola_matrix, q), dict(
            data_params=dict(mu=mu, vola_matrix=vola_matrix, window_size=window_size, num_paths=num_paths, num_bm=num_bm,
                             grid_points=grid_points)))
                     for mu, vola_matrix in [(param_mu, param_vola_matrix)]
                     )
    elif dataset == 'Heston':
        lambda_0, v0, kappa, theta, xi, rho = heston_params()
        generator = (('mu={}_sigma={}_q={}'.format(lambda_0, v0, q), dict(data_params=dict(lambda_0=lambda_0, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, window_size=window_size, num_paths=num_paths, grid_points=grid_points)))
                     for lambda_0, v0, kappa, theta, xi, rho in [(lambda_0, v0, kappa, theta, xi, rho)]
        )
    elif dataset == 'VarianceGamma':
        generator = (('mu={}_sigma={}_q={}'.format(mu, sigma, q), dict(data_params=dict(mu=mu, sigma=sigma, nu=nu, window_size=window_size, num_paths=num_paths, grid_points=grid_points)))
                     for mu, sigma, nu in [(0.05, 0.2, 0.02)]
        )
    elif dataset == 'Kou_Jump_Diffusion':
        generator = (('mu={}_sigma={}_window_size={}'.format(mu, sigma, window_size), dict(data_params=dict(mu=mu, sigma=sigma, kou_lambda=kou_lambda, p=p, eta1=eta1, eta2=eta2, window_size=window_size, num_paths=num_paths, grid_points=grid_points)))
                     for mu, sigma, kou_lambda, p, eta1, eta2 in [(0.05, 0.2, 2, 0.3, 25, 50)]
        )
    elif dataset == 'Levy_Ito':
        generator = (('mu={}_sigma={}_window_size={}'.format(mu, sigma, window_size), dict(data_params=dict(mu=mu, sigma=sigma, lambda_large=lambda_large, lambda_small=lambda_small, jump_mean_large=jump_mean_large, jump_std_large=jump_std_large, jump_mean_small=jump_mean_small, jump_std_small=jump_std_small, window_size=window_size, num_paths=num_paths, grid_points=grid_points)))
                     for mu, sigma, lambda_large, lambda_small, jump_mean_large, jump_std_large, jump_mean_small, jump_std_small in [(0.05, 0.2, 2, 300, 0.03, 0.05, 0.0005, 0.0005)]
        )
    elif dataset == 'YFinance':
        generator = ((
                'ticker={}_start={}_end={}_q={}'.format(ticker, start, end, q),
                dict(data_params=dict(ticker=ticker, start=start, end=end)))
            for ticker, start, end in [
            ("^GSPC", "2000-01-01", "2025-01-01"),
            ])
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator


def main(args):
    """
    if not pt.exists('./data'):
        os.mkdir('./data')
    if not pt.exists('./data/oxfordmanrealizedvolatilityindices.csv'):
        print('Downloading Oxford MAN AHL realised library...')
        download_man_ahl_dataset()
    if not pt.exists('./data/mitdb'):
        print('Downloading MIT-ECG database...')
        download_mit_ecg_dataset()
    """

    print('Start of training. CUDA: %s' % args.use_cuda)
    for dataset in args.datasets:
        for algo_id in args.algos:
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                base_config = BaseConfig(
                    device='cuda:{}'.format(args.device) if args.use_cuda and torch.cuda.is_available() else 'cpu',
                    seed=seed,
                    batch_size=args.batch_size,
                    hidden_dims=args.hidden_dims,
                    p=args.p,
                    q=args.q,
                    total_steps=args.total_steps,
                    mc_samples=1000
                )
                set_seed(seed)
                generator = get_dataset_configuration(dataset, window_size=args.window_size, num_paths=args.num_paths, num_bm=args.num_bm, grid_points=args.grid_points, q=args.q)
                for spec, data_params in generator:
                    run(args=args,
                        algo_id=algo_id,
                        base_config=base_config,
                        data_params=data_params,
                        dataset=dataset,
                        base_dir=args.base_dir,
                        spec=spec,
                    )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda', action='store_true')
    parser.add_argument('-device', default=0, type=int)
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    #parser.add_argument('-datasets', default=['ARCH', 'STOCKS', 'ECG', 'VAR', ], nargs="+")
    parser.add_argument('-datasets', default=['correlated_Blackscholes'], nargs="+")  # ['Stocks', 'ARCH', 'VAR'] 'Blackscholes', 'Heston', 'VarianceGamma', 'Kou_Jump_Diffusion', 'Levy_Ito', 'YFinance'
    parser.add_argument('-algos', default=['SigCWGAN'], nargs="+")  # 'SigCWGAN', 'GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN', 'CWGAN',


    # Algo hyperparameters
    parser.add_argument('-batch_size', default=200, type=int)
    parser.add_argument('-p', default=3, type=int)
    parser.add_argument('-q', default=3, type=int)
    parser.add_argument('-hidden_dims', default=3 * (50,), type=tuple)
    parser.add_argument('-total_steps', default=1000, type=int)
    parser.add_argument('-window_size', default=1000, type=int)
    parser.add_argument('-grid_points', default=252, type=int)
    parser.add_argument('-num_paths', default=1, type=int)  # atm unnecessary because only one path is allowed: If this is increased the paths will be merged into one path with [(windowsize - 1) * num_paths] values
    parser.add_argument('-num_bm', default=1, type=int)

    args = parser.parse_args()
    main(args)
