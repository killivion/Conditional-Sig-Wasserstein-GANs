import gymnasium as gym
import numpy as np
from td3_train import pull_data

class PortfolioEnv(gym.Env):
    def __init__(self, args, data_params, stock_data):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.num_stocks = stock_data.shape[1]
        self.portfolio_value = 0
        self.first_episode = True
        self.args = args
        self.data_params = data_params

        self.mu = np.insert(data_params['data_params']['mu'], 0, args.risk_free_rate)
        self.sigma_cov = np.zeros((self.num_stocks, self.num_stocks))
        self.sigma_cov[1:, 1:] = data_params['data_params']['sigma_cov']

        # Normalization:
        self.mu = (self.mu - np.mean(self.mu)) / np.std(self.mu)
        self.sigma_cov = self.sigma_cov / np.max(self.sigma_cov)

        # Define action and observation space
        feature_size = self.num_stocks + len(self.mu) + len(self.sigma_cov.flatten())
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feature_size,),dtype=np.float32,)

        # Initialize state
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks)

    def step(self, action):
        action -= np.mean(action)  # Normalize action to sum to 0

        portfolio_return = np.dot(action, self.stock_data[self.current_step]) + 1  # adjusted by 1 to compensate that sum(action)=0, hence portfolio return 1 is baseline
        self.portfolio_value *= portfolio_return
        terminated = self.current_step + 1 >= len(self.stock_data) - 1
        reward = self._calc_reward(action, terminated, portfolio_return)
        self.current_step += 1

        info, truncated = {}, False
        # obs = self.stock_data[self.current_step]
        obs = self._get_feature_map()
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)
        if self.first_episode:
            self.first_episode = False
        else:  # if it's not the first episode, new data is pulled
            np.random.seed(seed)
            self.stock_data = pull_data(self.data_params, self.args.dataset, self.args.risk_free_rate)
        # obs = np.array(self.stock_data[self.current_step], dtype=np.float32)
        obs = self._get_feature_map()
        info = {}
        return obs, info

    def _get_feature_map(self):
        returns = self.stock_data[self.current_step]
        feature_map = np.concatenate([returns, self.mu, self.sigma_cov.flatten()])
        return feature_map

    def _calc_reward(self, action, terminated, portfolio_return):
        if terminated:  # Terminal utility -> Central Reward-fct.
            reward = (self.portfolio_value ** (1 - self.args.p))  # / (1 - self.args.p) we leave out the constant divisor since it only scales the expectation
        else:
            if portfolio_return <= 0:  # intermediate reward function
                reward = -1000 * abs(portfolio_return)  # punishment for negative performance
            elif self.args.utility_function == "power":
                reward = (portfolio_return ** (1 - self.args.p))  # / (1 - self.args.p) we leave out the constant divisor since it only scales the expectation
            elif self.args.utility_function == "log":
                reward = np.log(portfolio_return)
            reward /= 100  # terminal reward weighted more important
        return reward
