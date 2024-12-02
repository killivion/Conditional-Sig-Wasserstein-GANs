from stable_baselines3 import TD3
import numpy as np
import yfinance as yf


def main(args):
    from lib.data import get_data
    from train import get_dataset_configuration

    generator = get_dataset_configuration(args.dataset, window_size=args.window_size, num_paths=args.num_paths)
    for spec, data_params in generator:
        if args.dataset == 'YFinance':
            ticker = data_params['data_params']['ticker']
            data = yf.download(ticker, start="2020-01-01", end="2024-01-01")['Adj Close']
        else:
            data = get_data(args.dataset, p=1, q=0, isSigLib=False, **data_params).T

    returns = data.pct_change().dropna().values + 1  # Compute daily change ratio [not daily returns]
    daily_risk_free_rate = (1 + args.risk_free_rate) ** (1/252)
    risk_free_column = np.full((returns.shape[0], 1), daily_risk_free_rate)
    returns = np.hstack((risk_free_column, returns))

    run(args.dataset, spec, returns, args.utility_function, args.p)


def run(dataset, spec, returns, utility_function, p):
    print('Executing TD3 on %s, %s' % (dataset, spec))
    from portfolio_env import PortfolioEnv
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = PortfolioEnv(utility_function, p, stock_data=returns)
    vec_env = DummyVecEnv([lambda: env])

    # Add action noise (exploration)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_env, action_noise=action_noise, verbose=1)

    mode = 'train'
    if mode == 'train':
        model.learn(total_timesteps=1000)
    elif mode == 'test':
        obs = env.reset()
        for _ in range(len(returns) - 1):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                break
    elif mode == 'eval':
        portfolio_value = [1.0]  # Initial portfolio value
        obs = env.reset()
        for _ in range(len(returns) - 1):
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            portfolio_value.append(portfolio_value[-1] * (1 + np.dot(action, returns[env.current_step])))

        # Plot portfolio value
        import matplotlib.pyplot as plt
        plt.plot(portfolio_value, label="TD3 Portfolio")
        plt.title("Portfolio Value Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-utility_function', default="power", type=str)
    parser.add_argument('-p', default=0.5, type=int)
    parser.add_argument('-dataset', default='YFinance', type=str)  # 'Blackscholes', 'Heston', 'VarianceGamma', 'Kou_Jump_Diffusion', 'Levy_Ito', 'YFinance'
    parser.add_argument('-risk_free_rate', default=0.025, type=int)
    parser.add_argument('-window_size', default=1000, type=int)
    parser.add_argument('-num_paths', default=1, type=int)

    args = parser.parse_args()
    main(args)


    """
        env = PortfolioEnv(stock_data=returns, utility_function="power", p=0.5)
        obs, info = env.reset()
        n_steps = 10
        for _ in range(n_steps):
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if done:
                obs, info = env.reset()
                """