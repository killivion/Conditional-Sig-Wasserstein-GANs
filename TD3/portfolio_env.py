import gymnasium as gym
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, utility_function, alpha, stock_data):
        super.__init__()
        self.stock_data = stock_data
        self.num_stocks = stock_data.shape[1]
        self.utility_function = utility_function
        self.alpha = alpha  # Risk aversion parameter for power utility

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks,), dtype=np.float32)

        # Initialize state
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks)

    def step(self, action):
        # Normalize action to sum to 1
        action = action / np.sum(action)

        # Compute portfolio return
        returns = self.stock_data[self.current_step]
        portfolio_return = np.dot(action, returns)

        if self.utility_function == "power":
            reward = (portfolio_return**self.alpha) / self.alpha if self.alpha != 0 else np.log(portfolio_return)
        elif self.utility_function == "log":
            reward = np.log(portfolio_return)

        self.current_step += 1
        done = self.current_step >= len(self.stock_data) - 1
        info = {}

        # New observation
        obs = self.stock_data[self.current_step]
        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        return self.stock_data[self.current_step]
