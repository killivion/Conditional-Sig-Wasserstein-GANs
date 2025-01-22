import os
import tensorflow as tf
from stable_baselines3.common.callbacks import BaseCallback
import yfinance as yf
import numpy as np
import shutil


class ActionLoggingCallback(BaseCallback):
    def __init__(self, log_dir: str, verbose: int = 0):
        super(ActionLoggingCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.summary_writer = tf.summary.create_file_writer(log_dir)
        self.episode_rewards = []  # To track rewards for the current episode

    def _on_step(self) -> bool:
        # Accumulate rewards for the current episode
        rewards = self.locals["rewards"]
        self.episode_rewards.extend(rewards)

        # Check if the episode is done
        if self.locals["dones"].any():
            actions = self.locals["actions"]  # Extract the last action(s)
            episode_num = self.n_calls

            # Calculate the total reward for the episode
            total_reward = sum(self.episode_rewards)
            self.episode_rewards = []  # Reset for the next episode

            # Extract actor and critic losses if available
            actor_loss = self.locals.get("actor_loss", None)
            critic_loss = self.locals.get("critic_loss", None)

            # Log data to TensorBoard
            with self.summary_writer.as_default():
                # Log actions
                for i, action in enumerate(actions):
                    tf.summary.scalar(f"action/Action", sum(action), step=episode_num)

                # Log episode reward
                tf.summary.scalar("action/Episode_Reward", total_reward/2 - 0.1, step=episode_num)

                # Log actor loss (if available)
                if actor_loss is not None:
                    tf.summary.scalar("action/Actor_Loss", actor_loss, step=episode_num)

                # Log critic loss (if available)
                if critic_loss is not None:
                    tf.summary.scalar("action/Critic_Loss", critic_loss, step=episode_num)

        return True


def pull_data(args, data_params):
    from lib.data import get_data
    if args.dataset == 'YFinance':
        ticker = data_params['data_params']['ticker']
        data = yf.download(ticker, start="2020-01-01", end="2024-01-01")['Adj Close']
    else:
        data = get_data(args.dataset, p=1, q=0, isSigLib=False, **data_params).T
    returns = data.pct_change().dropna().values + 1  # Compute dt change ratio [not dt returns]
    daily_risk_free_rate = (1 + args.risk_free_rate) ** (1 / args.grid_points)
    risk_free_column = np.full((returns.shape[0], 1), daily_risk_free_rate)
    return np.hstack((risk_free_column, returns))


def generate_random_params(num_paths):
    if num_paths != 1:
        low_vol = 0.1 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8)) if num_paths != 1 else 0.2 # Adjustment of up and lower bound depending on num_paths size (number of correlations)
        up_vol = 0.25 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8)) if num_paths != 1 else 0.2 # amounts to slightly more than 20% vol
        low_mu, up_mu = 0.03, 0.13
    else:
        low_vol, up_vol, low_mu, up_mu = 0.2, 0.2, 0.06, 0.06 # 0.2, 0.2, 0.15, 0.15
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
    analytical_utility = np.exp((1 - args.p) * (args.risk_free_rate + 1 / 2 * analytical_risky_action @ risky_lambda))

    return analytical_risky_action, analytical_utility


def find_largest_td3_folder(args):
    largest_number = 0
    for folder_name in os.listdir("./logs"):
        if folder_name[0].isdigit() and "_" in folder_name:
            try:
                largest_number = max(largest_number, int(folder_name.split('_')[0]))
            except ValueError:
                pass
        if folder_name.startswith("TD3") and "_" in folder_name:
            try:
                largest_number = max(largest_number, int(folder_name.split('_')[1]))
            except ValueError:
                pass
    return f"./logs/TD3_{largest_number+1}_actions", largest_number+1  # f"./logs/TD3_{largest_number}"


def fuse_folders(number, args):
    folder_actions = f"./logs/TD3_{number}_actions"
    folder_tensorboard = f"./logs/TD3_{number}_1"
    new_folder = f"./logs/{number}_ID_{args.model_ID}_window_{args.window_size}_batchsize_{args.batch_size}_trainfreq_{args.train_freq}"
    os.makedirs(new_folder)

    for folder in [folder_actions, folder_tensorboard]:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            new_item_path = os.path.join(new_folder, item)
            # Move each item (file or folder)
            if os.path.isdir(item_path):
                shutil.move(item_path, new_item_path)
            else:
                shutil.move(item_path, new_item_path)

    shutil.rmtree(folder_actions)
    shutil.rmtree(folder_tensorboard)

