import os
import tensorflow as tf
import yfinance as yf
import numpy as np
import shutil
from scipy.stats import norm
import torch

try:
    from stable_baselines3.td3.policies import TD3Policy
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    pass

#import generate

from sklearn.model_selection import train_test_split

def generate_random_params(num_paths, num_bm):
    if num_paths == 2 and num_bm == 2:
        total_vola = np.array([[0.07, 0.12]])
        weights = np.array([[1, 0], [0, 1]])  # rows sum to one
        mu = np.array([0.06, 0.08])
    elif num_paths == 1 and num_bm == 1:  # 1 path, 1 brownian motion
        total_vola = np.array([[0.04]])
        weights = np.array([[1]])
        mu = np.array([0.06])
    elif num_paths == 2 and num_bm == 3:
        total_vola = np.array([[0.06, 0.14]])
        weights = np.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])  # rows sum to one
        mu = np.array([0.08, 0.10])

    else:  # Adjustment of up and lower bound depending on num_paths size (number of correlations), amounts to slightly more than 20% vol
        low_vol = 0.1 * 3 * (np.log(1000)) ** (0.8) / (np.log(num_paths) ** (1.8))
        up_vol = 2.5 * low_vol
        low_mu, up_mu = 0.03, 0.13

        mu = np.random.uniform(low_mu, up_mu, size=num_paths)
        total_vola = np.random.uniform(low_vol, up_vol, size=num_paths)
        weights = np.random.rand(num_paths, num_bm)
        weights = weights / weights.sum(axis=1, keepdims=True)

        """
        correlation = np.random.uniform(-1, 1, size=(num_paths, num_bm))
        np.fill_diagonal(correlation, 1)
        correlation = (correlation + correlation.T) / 2
        eigvals, eigvecs = np.linalg.eigh(correlation)
        eigvals[eigvals < 0] = 1e-5
        correlation = eigvecs @ np.diag(eigvals) @ eigvecs.T  # correlation matrix with p_ij entries
        """

    vola_matrix = np.sqrt(weights * total_vola.T)  # [sigma] = vola_matrix

    return mu, vola_matrix  # mu is drift, vola_matrix


class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, allow_lending=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_lending = allow_lending
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.low = torch.tensor(self.action_space.low, device=device)
        self.high = torch.tensor(self.action_space.high, device=device)  # actions.device

    def softmax(self, x, axis=-1):
        return torch.softmax(x, dim=axis)

    def _predict(self, observation, deterministic=False):
        raw_actions = self.actor(observation)
        print(f'Before Softmax: {raw_actions}')
        if self.allow_lending:
            mean_actions = torch.mean(raw_actions, dim=-1, keepdim=True)
            n = raw_actions.shape[-1]
            actions = raw_actions - mean_actions + (1.0 / n)
        else:
            actions = self.softmax(raw_actions, axis=-1)
        invers_scaled_action = (actions - self.low)/(0.5 * (self.high - self.low)) - 1  # in td3 the action is in the space [-1,1], we inversely scale it to [-1,1]
        return invers_scaled_action


def action_normalizer(action):
    # return action
    if action.shape[0] == 1:
        return np.insert(action, 0, 1-action)
    else:
        return action / sum(action) if not sum(action) == 0 else np.zeros(action.shape[0]) + 1/action.shape[0]  # np.insert(action[1:], 0, 1)


def analytical_solutions(args, data_params):
    if args.dataset == 'correlated_Blackscholes':
        big_sigma = data_params['data_params']['vola_matrix'] @ data_params['data_params']['vola_matrix'].T
        risky_lambda = data_params['data_params']['mu'] - args.risk_free_rate
        analytical_risky_action = 1 / args.p * risky_lambda.T @ (np.linalg.inv(big_sigma))
        analytical_utility = expected_utility(analytical_risky_action, args, data_params)
        #other logic - gives same result: analytical_utility = np.exp((1 - args.p) * (args.risk_free_rate + 1/2 * analytical_risky_action.T @ risky_lambda))
    elif args.dataset == 'Heston':  #needs to be added
        """
        current_timestep = 1
        T = args.window_size/args.grid_size
        t = current_timestep/args.grid_size
        lambda_0, v0, kappa, theta, xi, rho = data_params['data_params']['lambda_0', 'v0', 'kappa', 'theta', 'xi', 'rho']
        kappa_tilde = kappa - args.p/(args.p) * rho * lambda_0 * sigma
        c = (args.p)/(args.p + rho**2 * (1-args.p))
        beta_tilde = -0.5 * 1/c * args.p/(1-args.p) * lambda_0**2
        a_tilde = np.sqrt(kappa_tilde**2 + 2.0 * beta_tilde * sigma**2)
        B_fT = 2*beta_tilde * (np.exp(a_tilde * (T - t)) - 1) / (np.exp(a_tilde * (T - t)) * (kappa_tilde + a_tilde) - kappa_tilde + a_tilde)

        analytical_risky_action = lambda_0/(args.p) - 1/(args.p) * c * rho * sigma * B_fT
        
        A_T = - kappa * theta * (T - t) * (kappa + a) / sigma**2
        f_tz = np.exp((gamma / c)*r*(T-t) - A_val - B_fT * z)
        
        #V(t,x,z)
        analytical_utility = 
        """
        analytical_risky_action = np.array([data_params['data_params']['lambda_0']/(args.p)])
        analytical_utility = expected_utility(analytical_risky_action, args, data_params)
    else:
        big_sigma = np.array([[0.2 * 0.2]])
        risky_lambda = np.array([0.06 - args.risk_free_rate])
        analytical_risky_action = 1 / args.p * risky_lambda.T @ (np.linalg.inv(big_sigma))
        analytical_utility = expected_utility(analytical_risky_action, args, data_params)


    return analytical_risky_action, analytical_utility


def expected_utility(action, args, data_params):
    if args.dataset == 'correlated_Blackscholes':
        risky_lambda = data_params['data_params']['mu'] - args.risk_free_rate
        big_sigma = data_params['data_params']['vola_matrix'] @ data_params['data_params']['vola_matrix'].T
        expected_utility = np.exp((1 - args.p) * (args.risk_free_rate + action.T @ risky_lambda - args.p / 2 * (action.T @ big_sigma @ action)))
    elif args.dataset == 'Heston':  #needs to be added
        big_sigma = np.array([[0.2 * 0.2]])
        risky_lambda = np.array([0.08 - args.risk_free_rate])
        expected_utility = np.exp((1 - args.p) * (args.risk_free_rate + action.T @ risky_lambda - args.p / 2 * (action.T @ big_sigma @ action)))
    else:
        big_sigma = np.array([[0.2 * 0.2]])
        risky_lambda = np.array([0.08 - args.risk_free_rate])
        expected_utility = np.exp((1 - args.p) * (args.risk_free_rate + action.T @ risky_lambda - args.p / 2 * (action.T @ big_sigma @ action)))

    return expected_utility


def analytical_entry_wealth_offset(action, args, data_params):
    analytical_risky_action, _ = analytical_solutions(args, data_params)
    policy_expected_utility = expected_utility(action[1:], args, data_params)
    analy_expected_utility = expected_utility(analytical_risky_action, args, data_params)
    entry_wealth_offset = (analy_expected_utility/policy_expected_utility) ** (1/(1-args.p))
    #alternative logic - gives same result: entry_wealth_offset = np.exp((analytical_risky_action - action[1:]) @ risky_lambda - args.p/2 * (analytical_risky_action.T @ big_sigma @ analytical_risky_action - action[1:].T @ big_sigma @ action[1:]))  # (1-p) and (1/(1-p)) cancel, r-r cancels

    return entry_wealth_offset


def find_confidence_intervals(analytical_risky_action, data_params, args):  # One could upgrade to joint confidence regions: with elliptical regions using the Mahalanobis distance
    confidence = 0.95
    z_c = norm.ppf((1 + confidence) / 2)
    dt = (1/args.grid_points)
    if args.dataset == 'correlated_Blackscholes':
        big_sigma = np.diag(data_params['data_params']['vola_matrix'] @ data_params['data_params']['vola_matrix'].T)[0]
        mu_adj = (data_params['data_params']['mu'][0] - big_sigma) * dt
    elif args.dataset == 'Heston':  # needs adjustment
        big_sigma = np.array([data_params['data_params']['v0']]) @ np.array([data_params['data_params']['v0']]).T
        mu_adj = (data_params['data_params']['lambda_0'] - big_sigma) * dt
    else: # needs adjustment
        mu_adj = np.array([0.05])
        big_sigma = np.array([0.2])


    interval = mu_adj + np.array([-1, 1]) * z_c * np.sqrt(big_sigma) * np.sqrt(dt)  # exp((1-p)[(μ-σ^2/2)T (+/-) 1.96σ*sqrt(T)])
    cf_low, cf_high = np.exp((1 - args.p) * interval)  # Extreme case x0=1 is invested in asset 1
    cf_low2, cf_high2 = sum(analytical_risky_action) * np.exp((1-args.p)*interval) + (1 - sum(analytical_risky_action)) * np.exp((1 - args.p) * args.risk_free_rate * dt)  # simplified that all risky action is in asset 1

    return cf_low - cf_low2, cf_high - cf_high2


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
    return f"./logs/TD3_{largest_number+1}_actions_{args.statement}", largest_number+1  # f"./logs/TD3_{largest_number}"


def fuse_folders(number, args):
    folder_actions = f"./logs/TD3_{number}_actions_{args.statement}"
    folder_tensorboard = f"./logs/TD3_{number}_1"
    final_folder = f"./logs/{number}_ID{args.model_ID}_assets{args.num_paths}_window{args.window_size}_batchsize{args.batch_size}_learningrate{args.learning_rate}_{args.statement}"
    os.makedirs(final_folder)

    for folder in [folder_actions, folder_tensorboard]:
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            new_item_path = os.path.join(final_folder, item)
            # Move each item (file or folder)
            if os.path.isdir(item_path):
                shutil.move(item_path, new_item_path)
            else:
                shutil.move(item_path, new_item_path)

    shutil.rmtree(folder_actions)
    shutil.rmtree(folder_tensorboard)


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
                    tf.summary.scalar(f"action/Action", sum(action_normalizer(action)[1:]), step=episode_num)
                    normalized_actions = action_normalizer(action)
                    for j in range(len(action)-1):
                        tf.summary.scalar(f"action/Action_{j+1}", normalized_actions[j+1], step=episode_num)

                # Log episode reward
                tf.summary.scalar("action/Episode_Reward", total_reward, step=episode_num)

                # Log actor loss (if available)
                if actor_loss is not None:
                    tf.summary.scalar("action/Actor_Loss", actor_loss, step=episode_num)

                # Log critic loss (if available)
                if critic_loss is not None:
                    tf.summary.scalar("action/Critic_Loss", critic_loss, step=episode_num)

        return True


def get_dataset_configuration(dataset, window_size, num_paths, grid_points):
    if dataset == 'Blackscholes':
        generator = (('mu={}_sigma={}_window_size={}'.format(mu, sigma, window_size), dict(data_params=dict(mu=mu, sigma=sigma, window_size=window_size, num_paths=num_paths, grid_points=grid_points)))
                     for mu, sigma in [(0.07, 0.2)]
        )
    elif dataset == 'YFinance':
        generator = ((
            'ticker={}_start={}_end={}'.format(ticker, start, end),
            dict(params=dict(ticker=ticker, start=start, end=end)))
            for ticker, start, end in [
            ("^GSPC", "2000-01-01", "2025-01-01"),
        ])
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator
