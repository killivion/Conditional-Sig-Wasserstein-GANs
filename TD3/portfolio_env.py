import gymnasium as gym
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, utility_function, p, stock_data):
        super(PortfolioEnv, self).__init__()
        self.stock_data = stock_data
        self.num_stocks = stock_data.shape[1]
        self.utility_function = utility_function
        self.p = p  # Risk aversion parameter for power utility

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.num_stocks,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_stocks,), dtype=np.float32)

        # Initialize state
        self.current_step = 0
        self.current_weights = np.zeros(self.num_stocks)

    def step(self, action):
        action -= np.mean(action)  # Normalize action to sum to 0

        returns = self.stock_data[self.current_step]
        portfolio_return = np.dot(action, returns) + 1  # adjusted by 1 to compensate that sum(action)=0, hence portfolio return 1 is baseline
        if portfolio_return <= 0:
            reward = -1000 * abs(portfolio_return)  # punishment for negative performance
        elif self.utility_function == "power":
            reward = (portfolio_return ** (1 - self.p)) / (1 - self.p)
        elif self.utility_function == "log":
            reward = np.log(portfolio_return)

        self.current_step += 1

        terminated = self.current_step >= len(self.stock_data) - 1
        truncated = False  # No truncation logic for now
        info = {}
        obs = self.stock_data[self.current_step]
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        obs = np.array(self.stock_data[self.current_step], dtype=np.float32)
        info = {}  # Empty dictionary, or add custom info if needed
        return obs, info