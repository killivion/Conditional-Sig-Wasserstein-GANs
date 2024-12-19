import os
import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
import yfinance as yf
import numpy as np


class ActionLoggingCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose: int = 0):
        super(ActionLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)

    def _on_step(self) -> bool:
        # Log action at the end of each episode
        if self.locals["dones"].any():  # Check if episode is done
            actions = self.locals["actions"]  # Extract the last action(s)
            episode_num = self.n_calls

            # Log the action to TensorBoard
            with self.summary_writer.as_default():
                for i, action in enumerate(actions):
                    tf.summary.scalar(f"Action_{i}", sum(action), step=episode_num)

        return True


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


def generate_random_params(num_paths):
    if num_paths!= 1:
        low_vol = 0.1 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8)) if num_paths != 1 else 0.2 # Adjustment of up and lower bound depending on num_paths size (number of correlations)
        up_vol = 0.25 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8)) if num_paths != 1 else 0.2 # amounts to slightly more than 20% vol
        low_mu, up_mu = 0.03, 0.13
    else:
        low_vol, up_vol, low_mu, up_mu = 0.2, 0.2, 0.15, 0.15
    mu = np.random.uniform(low_mu, up_mu, size=num_paths)
    volatilities = np.random.uniform(low_vol, up_vol, size=num_paths)
    correlation = np.random.uniform(-1, 1, size=(num_paths, num_paths))
    np.fill_diagonal(correlation, 1)
    correlation = (correlation + correlation.T) / 2
    eigvals, eigvecs = np.linalg.eigh(correlation)
    eigvals[eigvals < 0] = 1e-5
    correlation = eigvecs @ np.diag(eigvals) @ eigvecs.T

    sigma_cov = correlation * np.outer(volatilities, volatilities)
    return mu, sigma_cov


def analytical_solutions(args, data_params):
    cholesky = np.linalg.cholesky(data_params['data_params']['sigma_cov'])
    risky_lambda = data_params['data_params']['mu'] - args.risk_free_rate
    analytical_risky_action = 1 / args.p * risky_lambda @ ((cholesky @ cholesky.T) ** (-1))
    analytical_utility = np.exp(
        (1 - args.p) * (args.risk_free_rate + 1 / 2 * analytical_risky_action @ risky_lambda))

    return analytical_risky_action, analytical_utility


def find_largest_td3_folder():
    largest_number = 0
    for folder_name in os.listdir("./logs"):
        largest_number = max(largest_number, int(folder_name.split('_')[1]))
    return f"./logs/TD3_actions_{largest_number+1}"  # f"./logs/TD3_{largest_number}"

