from stable_baselines3 import TD3
import numpy as np
import yfinance as yf
import os
import torch
import eval_actor


def main(args):
    from train import get_dataset_configuration
    if args.dataset == 'correlated_Blackscholes':
        mu, sigma_cov = generate_random_params(args.num_paths)
        spec = 'args.num_paths={}_window_size={}'.format(args.num_paths, args.window_size)
        data_params = dict(data_params=dict(mu=mu, sigma_cov=sigma_cov, window_size=args.window_size, num_paths=args.num_paths, grid_points=args.window_size))
    else:
        generator = get_dataset_configuration(args.dataset, window_size=args.window_size, num_paths=args.num_paths, grid_points=args.window_size)
        for s, d in generator:
            spec, data_params = s, d  # odd way to do it, works in 1-d

    returns = pull_data(data_params, args.dataset, args.risk_free_rate)
    run(args, spec, data_params, returns)


def generate_random_params(num_paths):
    low_vol = 0.1 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (
        1.8))  # Adjustment of up and lower bound depending on num_paths size (number of correlations)
    up_vol = 0.25 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8))  # amounts to slightly more than 20% vol

    mu = np.random.uniform(0.03, 0.13, size=num_paths)
    volatilities = np.random.uniform(low_vol, up_vol, size=num_paths)
    correlation = np.random.uniform(-1, 1, size=(num_paths, num_paths))
    np.fill_diagonal(correlation, 1)
    correlation = (correlation + correlation.T) / 2
    eigvals, eigvecs = np.linalg.eigh(correlation)
    eigvals[eigvals < 0] = 1e-5
    correlation = eigvecs @ np.diag(eigvals) @ eigvecs.T

    sigma_cov = correlation * np.outer(volatilities, volatilities)
    return mu, sigma_cov

def pull_data(data_params, dataset, risk_free_rate):
    from lib.data import get_data
    if dataset == 'YFinance':
        ticker = data_params['data_params']['ticker']
        data = yf.download(ticker, start="2020-01-01", end="2024-01-01")['Adj Close']
    else:
        data = get_data(dataset, p=1, q=0, isSigLib=False, **data_params).T

    returns = data.pct_change().dropna().values + 1  # Compute daily change ratio [not daily returns]
    daily_risk_free_rate = (1 + risk_free_rate) ** (1 / 252)
    risk_free_column = np.full((returns.shape[0], 1), daily_risk_free_rate)
    return np.hstack((risk_free_column, returns))


def run(args, spec, data_params, returns):
    print('Executing TD3 on %s, %s' % (args.dataset, spec))
    from portfolio_env import PortfolioEnv
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns), filename='log')
    vec_env = DummyVecEnv([lambda: env])

    # Add action noise (exploration)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_env, action_noise=action_noise, verbose=1, tensorboard_log="./logs")

    hardware = 'LRZ' if torch.cuda.is_available() else 'cpu'
    model_save_path = f"./agent/{hardware}_td3_agent_{args.actor_dataset}"
    if os.path.exists(model_save_path):
        model = TD3.load(model_save_path)
    if args.mode == 'train':
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True, tb_log_name="TD3")
        model.save(model_save_path)
        import track_learning
        track_learning.monitor_plot()
        # tensorboard --logdir ./logs
    if args.mode in ['test', 'train']:
        eval_actor.test_actor(model, env)
    elif args.mode == 'eval':
        eval_actor.evaluate_actor(args, data_params, model, env)
    elif args.mode == 'compare':
        eval_actor.compare_actor(args, data_params, model, env)
        # trained_rewards, random_rewards, trained_portfolio_values, random_portfolio_values


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-utility_function', default="power", type=str)
    parser.add_argument('-p', default=0.8, type=int)
    parser.add_argument('-dataset', default='correlated_Blackscholes', type=str)  # 'Blackscholes', 'Heston', 'VarianceGamma', 'Kou_Jump_Diffusion', 'Levy_Ito', 'YFinance', 'correlated_Blackscholes'
    parser.add_argument('-actor_dataset', default='correlated_Blackscholes', type=str)  # An Actor ID to determine which actor will be loaded (if it exists), then trained or tested/evaluated on
    parser.add_argument('-risk_free_rate', default=0.025, type=int)
    parser.add_argument('-window_size', default=100, type=int)
    parser.add_argument('-num_paths', default=30, type=int)
    parser.add_argument('-total_timesteps', default=5000, type=int)
    parser.add_argument('-num_episodes', default=1000, type=int)
    parser.add_argument('-mode', default='compare', type=str)  # 'train' 'test' 'eval' 'compare'

    args = parser.parse_args()
    main(args)
