import gymnasium as gym
import numpy as np
from td3_train import pull_data

class PortfolioEnv(gym.Env):
    def __init__(self, args, data_params, stock_data):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.num_stocks = stock_data.shape[1]
        self.portfolio_value = 1
        self.first_episode = True
        self.args = args
        self.data_params = data_params

        self.normalized_stock_data = (self.stock_data - 1) / np.std(self.stock_data, axis=1, keepdims=True)
        self._normalize_parameter()

        # Define action and observation space
        feature_size = self.num_stocks + len(self.mu) + len(self.sigma_cov.flatten())
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feature_size,),dtype=np.float32,)

        # Initialize state
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks)

    def step(self, action):
        action /= sum(action) if sum(action) != 0 else [0.5, 0.5]  # action -= np.mean(action)
        portfolio_return = np.dot(action, self.stock_data[self.current_step])  #+1 # adjusted by 1 to compensate that sum(action)=0, hence portfolio return 1 is baseline
        self.portfolio_value *= portfolio_return

        done = self.current_step + 1 >= len(self.stock_data) - 1
        reward = self._calc_reward(done, portfolio_return)
        self.current_step += 1

        truncated = False
        info = self.portfolio_value if not done and self.args.mode in ['eval', 'compare'] else {}

        # obs = self.stock_data[self.current_step]
        obs = self._get_feature_map()
        return obs, reward, done, truncated, info

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.current_step = 0
        if seed is not None:
            np.random.seed(seed)
        if self.first_episode:
            self.first_episode = False
        else:  # if it's not the first episode, new data is pulled
            np.random.seed(seed)
            if self.args.mode in ['eval', 'compare']:  # pulls new rdm parameters
                from td3_train import generate_random_params
                mu, sigma_cov = generate_random_params(self.num_stocks-1)
                self.data_params = dict(data_params=dict(mu=mu, sigma_cov=sigma_cov, window_size=self.args.window_size, num_paths=self.args.num_paths,grid_points=self.args.window_size))
                self._normalize_parameter()
            self.portfolio_value = 1
            self.stock_data = pull_data(self.data_params, self.args.dataset, self.args.risk_free_rate)
            self.normalized_stock_data = (self.stock_data - 1) / np.std(self.stock_data, axis=1, keepdims=True)
        # obs = np.array(self.stock_data[self.current_step], dtype=np.float32)
        obs = self._get_feature_map()
        info = {}
        return obs, info

    def _get_feature_map(self):
        normalized_returns = self.normalized_stock_data[self.current_step]
        feature_map = np.concatenate([normalized_returns, self.mu, self.sigma_cov.flatten()])
        return feature_map

    def _calc_reward(self, done, portfolio_return):
        if done:  # Terminal utility -> Central Reward-fct.
            reward = (self.portfolio_value ** (1 - self.args.p))  # / (1 - self.args.p) we leave out the constant divisor since it only scales the expectation
            if self.args.mode == 'test':
                print('Terminal Reward is: %s' % reward)
        elif self.args.mode not in ['compare', 'eval']:
            if portfolio_return <= 0:  # intermediate reward function
                reward = -1000 * abs(portfolio_return)  # punishment for negative performance
            elif self.args.utility_function == "power":
                reward = (portfolio_return ** (1 - self.args.p))
            elif self.args.utility_function == "log":
                reward = np.log(portfolio_return)
            reward /= (3 * self.args.window_size)  # terminal reward weighted more important
        else:
            reward = 0
        return reward

    def _normalize_parameter(self):
        self.mu = np.insert(self.data_params['data_params']['mu'], 0, self.args.risk_free_rate)
        self.sigma_cov = np.zeros((self.num_stocks, self.num_stocks))
        self.sigma_cov[1:, 1:] = np.sqrt(self.data_params['data_params']['sigma_cov'])

        # Normalization:
        if self.num_stocks != 2:
            self.mu = (self.mu - np.mean(self.mu)) / np.std(self.mu)
            self.sigma_cov = self.sigma_cov / np.max(self.sigma_cov)
