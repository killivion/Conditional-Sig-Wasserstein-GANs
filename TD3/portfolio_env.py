import gymnasium as gym
import numpy as np
from help_fct import pull_data, analytical_solutions, find_confidence_intervals, action_normalizer
from collections import deque

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
        feature_size = len(self.mu) + len(self.vola_matrix.flatten())  #self.num_stocks + len(self.mu) + len(self.vola_matrix.flatten())
        if args.allow_lending:
            self.action_space = gym.spaces.Box(low=-8, high=8, shape=(self.num_stocks,), dtype=np.float32)
        else:
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(feature_size,), dtype=np.float32,)

        # Initialize state
        self.current_step, self.portfolio_value = 0, 1
        self.first_episode, self.episode_cycle = True, 0
        #self.max_intermediary_reward, self.max_terminal_reward = 0, 0
        #self.reward_normalization_window = args.learning_starts / 2
        #self.reward_window, self.fixed = [], False  # deque(maxlen=1000)
        self.analytical_risky_action, self.analytical_utility = analytical_solutions(self.args, self.data_params)
        self.lower_CI, self.upper_CI = find_confidence_intervals(self.analytical_risky_action, self.data_params, self.args)
        self.step_count, self.i_steps = 0, 2
        self.action = np.zeros(self.num_stocks)


    def step(self, action):
        self.current_step += 1

        #self.step_count += 1
        #if self.step_count >= self.args.total_timesteps/10 * self.i_steps and self.args.mode == 'train':
        #    print(f"{self.i_steps*10}% done")
        #    self.i_steps += 2

        done = self.current_step >= len(self.stock_data)

        self.action = action_normalizer(action)  # normalizes to 1  # if self.num_stocks > 1 else [1 - action, action]
        self.portfolio_value *= self.action @ self.stock_data[self.current_step-1]  # Tipp: adjust by 1 to compensate if sum(action)=0, so portfolio return 1 stays baseline
        self.optimal_portfolio *= np.insert(self.analytical_risky_action, 0, 1 - sum(self.analytical_risky_action)) @ self.stock_data[self.current_step-1]
        reward = self._calc_reward(done)

        obs = self._get_feature_map()
        truncated = False
        info = self.portfolio_value if self.args.mode in ['eval', 'compare'] else {}  #and not done
        return obs, reward, done, truncated, info

    def reset(self, seed=None, test=False, random_actor=False, **kwargs):
        super().reset(seed=seed)
        self.episode_cycle += 1
        self.current_step = 0
        self.portfolio_value, self.optimal_portfolio = 1, 1
        if seed is not None:
            np.random.seed(seed)
        if test:
            self.args.mode = 'test'
        if self.first_episode or random_actor:  # in the random_actor case ensures that in compare random_actor and the trained actor use the same dataset
            self.first_episode = False
        else:  # if not the first episode, new data is pulled
            if self.args.mode in ['eval', 'compare'] or self.episode_cycle == self.args.episode_reset:  # every episode_reset episodes, pulls new rdm parameters
                self.episode_cycle = 0
                from td3_train import generate_random_params
                mu, vola_matrix = generate_random_params(self.num_stocks-1, self.args.num_bm)
                self.data_params = dict(data_params=dict(mu=mu, vola_matrix=vola_matrix, window_size=self.args.window_size, num_paths=self.args.num_paths, num_bm=self.args.num_bm, grid_points=self.args.window_size))
                self.analytical_risky_action, self.analytical_utility = analytical_solutions(self.args, self.data_params)
                self._normalize_parameter()
            self.stock_data = pull_data(self.args, self.data_params)
            # self.normalized_stock_data = (self.stock_data - 1) / np.std(self.stock_data, axis=1, keepdims=True)
        obs = self._get_feature_map()
        info = {}
        return obs, info

    def _get_feature_map(self):
        # normalized_returns = self.normalized_stock_data[self.current_step-1]
        feature_map = np.concatenate([self.mu, self.vola_matrix.flatten()])  #np.concatenate([normalized_returns, self.mu, self.vola_matrix.flatten()])
        return feature_map

    def _calc_reward(self, done):
        if done:  # Terminal utility -> Central Reward-fct.
            if self.portfolio_value <= 0:
                print(f"Careful: negative portfolio value: {self.portfolio_value, self.stock_data[self.current_step-1], self.action}")
            reward = (self.portfolio_value ** (1 - self.args.p)) if not self.portfolio_value <= 0 else 0 * abs(self.portfolio_value)  # / (1 - self.args.p) leave out the constant divisor since it only scales the expectation
            optimal_utility = (self.optimal_portfolio ** (1 - self.args.p)) if not self.optimal_portfolio <= 0 else 0
            reward = reward - optimal_utility
            #self.reward_window.append(reward)
            #if not self.fixed:
            #    self.mean_reward = np.mean(self.reward_window) if self.reward_window else 0.0
            #    self.std_reward = np.std(self.reward_window) if self.reward_window else 1.0
            #if len(self.reward_window) >= 1000:
            #    self.fixed = True
            if self.args.mode not in ['compare', 'eval', 'test']:
                if self.args.allow_lending:
                    normalized_reward = 2 * (reward + 0.1) if self.args.mode not in ['compare', 'eval'] else reward
                else:  # Normalizes via Confidence-Intervals of the extrema: Investing all in stock [in multi-dimensional: uses exemplary CI of the first stock and assumes that all is invested in this one]
                    normalized_reward = (reward - self.lower_CI) / self.upper_CI * 3 - 2  # normalizes that most values lie in [-1, 1]
            else:
                normalized_reward = reward
            #    normalized_reward = (reward - self.mean_reward) / (np.sqrt(self.std_reward) if self.std_reward > 0 else 1.0) if self.args.mode not in ['compare', 'eval', 'tuning'] else reward
            if self.args.mode == 'test':
                print('Terminal Utility is: %s' % ((self.portfolio_value) ** (1 - self.args.p)))
            """
        elif self.args.mode not in ['compare', 'eval']:  # intermediate reward function
            if portfolio_return <= 0:
                reward = -1000 * abs(portfolio_return)  # punishment for negative performance
            elif self.args.utility_function == "power":
                reward = (portfolio_return ** (1 - self.args.p))
            elif self.args.utility_function == "log":
                reward = np.log(portfolio_return)
            """
        else:
            normalized_reward = 0
        return normalized_reward

    def _normalize_parameter(self):  # adds the risk_free parameters and normalizes
        self.mu = np.insert(self.data_params['data_params']['mu'], 0, self.args.risk_free_rate)
        #self.vola_matrix = np.zeros((self.num_stocks, self.num_stocks))
        self.vola_matrix = np.sign(self.data_params['data_params']['vola_matrix']) * np.sqrt(np.abs(self.data_params['data_params']['vola_matrix']))

        # Normalization:
        #if self.num_stocks != 2:
        #    self.mu = (self.mu - np.mean(self.mu)) / np.std(self.mu)
        #   self.vola_matrix = self.vola_matrix / np.max(self.vola_matrix)


