import gymnasium as gym
import numpy as np
from td3_train import pull_data

class PortfolioEnv(gym.Env):
    def __init__(self, args, data_params, stock_data):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.num_stocks = stock_data.shape[1]
        self.args = args
        self.data_params = data_params

        self.normalized_stock_data = (self.stock_data - 1) / np.std(self.stock_data, axis=1, keepdims=True)
        self._normalize_parameter()

        # Define action and observation space
        feature_size = len(self.mu) + len(self.sigma_cov.flatten())  #self.num_stocks + len(self.mu) + len(self.sigma_cov.flatten())
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32) if self.num_stocks > 2 else gym.spaces.Box(low=-8, high=8, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feature_size,),dtype=np.float32,)

        # Initialize state
        self.current_step, self.portfolio_value = 0, 1
        self.first_episode, self.episode_cycle = True, 0
        self.max_intermediary_reward, self.max_terminal_reward = 0, 0

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1

        action = action/sum(action) if self.num_stocks > 2 else [1 - action[0], action[0]]  # action -= np.mean(action)
        portfolio_return = np.dot(action, self.stock_data[self.current_step])  #+1 # adjusted by 1 to compensate that sum(action)=0, hence portfolio return 1 is baseline
        self.portfolio_value *= portfolio_return
        reward = self._calc_reward(done, portfolio_return)

        # obs = self.stock_data[self.current_step]
        obs = self._get_feature_map()
        truncated = False
        info = self.portfolio_value if not done and self.args.mode in ['eval', 'compare'] else {}
        return obs, reward, done, truncated, info

    def reset(self, seed=None, test=False, random_actor=False, **kwargs):
        super().reset(seed=seed)
        self.episode_cycle += 1
        self.current_step = 0
        self.portfolio_value = 1
        if seed is not None:
            np.random.seed(seed)
        if test:
            self.args.mode = 'test'
        if self.first_episode or random_actor:
            self.first_episode = False
        else:  # if it's not the first episode, new data is pulled
            np.random.seed(seed)
            if self.args.mode in ['eval', 'compare'] or self.episode_cycle == self.args.episode_reset:  # pulls new rdm parameters
                self.episode_cycle = 0
                from td3_train import generate_random_params
                mu, sigma_cov = generate_random_params(self.num_stocks-1)
                self.data_params = dict(data_params=dict(mu=mu, sigma_cov=sigma_cov, window_size=self.args.window_size, num_paths=self.args.num_paths,grid_points=self.args.window_size))
                self._normalize_parameter()
            self.stock_data = pull_data(self.data_params, self.args.dataset, self.args.risk_free_rate)
            self.normalized_stock_data = (self.stock_data - 1) / np.std(self.stock_data, axis=1, keepdims=True)
        # obs = np.array(self.stock_data[self.current_step], dtype=np.float32)
        obs = self._get_feature_map()
        info = {}
        return obs, info

    def _get_feature_map(self):
        normalized_returns = self.normalized_stock_data[self.current_step]
        feature_map = np.concatenate([self.mu, self.sigma_cov.flatten()])  #np.concatenate([normalized_returns, self.mu, self.sigma_cov.flatten()])
        return feature_map

    def _calc_reward(self, done, portfolio_return):
        if done:  # Terminal utility -> Central Reward-fct.
            reward = (self.portfolio_value ** (1 - self.args.p)) if not portfolio_return <= 0 else -1000 * abs(portfolio_return)  # / (1 - self.args.p) leave out the constant divisor since it only scales the expectation
            normalized_reward = 2 * (1.2 * reward - 0.9) if self.args.mode not in ['compare', 'eval'] else reward
            if self.args.mode == 'test':
                print('Terminal Utility is: %s' % ((self.portfolio_value) ** (1 - self.args.p)))

            #self.max_terminal_reward = max(self.max_terminal_reward, reward
            #normalized_reward = 0.5 * (1 + (reward - 1) / (self.max_terminal_reward - 1)) if self.args.mode not in ['compare', 'eval'] else reward
            """
        elif self.args.mode not in ['compare', 'eval']:  # intermediate reward function
            if portfolio_return <= 0:
                reward = -1000 * abs(portfolio_return)  # punishment for negative performance
            elif self.args.utility_function == "power":
                reward = (portfolio_return ** (1 - self.args.p))
            elif self.args.utility_function == "log":
                reward = np.log(portfolio_return)
            # self.max_intermediary_reward = max(self.max_intermediary_reward, reward)
            normalized_reward = 2 * (1.2 * reward - 0.9)  #0.5 * (1 + (reward - 1) / (self.max_terminal_reward - 1))
            normalized_reward /= (4 * self.args.window_size)  # terminal reward weighted more important
            """
        else:
            normalized_reward = 0
        return normalized_reward

    def _normalize_parameter(self):
        self.mu = np.insert(self.data_params['data_params']['mu'], 0, self.args.risk_free_rate)
        self.sigma_cov = np.zeros((self.num_stocks, self.num_stocks))
        self.sigma_cov[1:, 1:] = np.sign(self.data_params['data_params']['sigma_cov']) * np.sqrt(np.abs(self.data_params['data_params']['sigma_cov']))

        # Normalization:
        if self.num_stocks != 2:
            self.mu = (self.mu - np.mean(self.mu)) / np.std(self.mu)
            self.sigma_cov = self.sigma_cov / np.max(self.sigma_cov)
