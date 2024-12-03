from stable_baselines3 import TD3
import numpy as np
import yfinance as yf
import os


def main(args):
    from train import get_dataset_configuration
    generator = get_dataset_configuration(args.dataset, window_size=args.window_size, num_paths=args.num_paths)
    for spec, data_params in generator:
        returns = pull_data(data_params, args.dataset, args.risk_free_rate)
        run(args, spec, data_params, returns)


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


def run(args, spec, data_params, returns):
    print('Executing TD3 on %s, %s' % (args.dataset, spec))
    from portfolio_env import PortfolioEnv
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(PortfolioEnv(args.utility_function, args.p, stock_data=returns))
    vec_env = DummyVecEnv([lambda: env])

    # Add action noise (exploration)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_env, action_noise=action_noise, verbose=1)

    model_save_path = f"./agent/td3_agent_{args.actor_dataset}"
    if os.path.exists(model_save_path):
        model = TD3.load(model_save_path)
    if args.mode == 'train':
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True, tb_log_name="TD3")
        model.save(model_save_path)
        # tensorboard --logdir ./logs
    elif args.mode == 'test':
        obs, info = env.reset()
        total_reward = 0
        for _ in range(len(returns) - 1):
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated:
                print("Total reward during test:", total_reward)
                break
    elif args.mode == 'eval':
        from tqdm import tqdm
        portfolio_values = []
        for episode in tqdm(range(args.num_episodes), desc="Episodes", leave=False):
            portfolio_value = [1.0]  # Initial portfolio value
            obs, info = env.reset()
            returns = pull_data(data_params, args.dataset, args.risk_free_rate)
            obs = np.array(returns[0], dtype=np.float32)
            for _ in range(len(returns) - 1):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                portfolio_value.append(portfolio_value[-1] * (1 + np.dot(action, returns[env.unwrapped.current_step])))
            portfolio_values.append(portfolio_value)

        portfolio_values = np.array(portfolio_values)
        mean_portfolio_value = portfolio_values.mean(axis=0)

        import matplotlib.pyplot as plt
        plt.plot(range(len(returns)), mean_portfolio_value, color="red")
        plt.plot(portfolio_values.T, color="blue")
        plt.title("Portfolio Value Episodes")
        plt.xlabel("Episode")
        plt.ylabel("Portfolio Value")
        #plt.legend()
        plt.show()
    elif args.mode == 'compare':
        from evaluate_actor import evaluate_actor
        trained_rewards, random_rewards = evaluate_actor(args, data_params, model, env, num_episodes=10, random_actor=False)

        print(f"Trained Actor Average Reward: {np.mean(trained_rewards)}")
        print(trained_rewards)
        print(f"Random Actor Average Reward: {np.mean(random_rewards)}")
        print(random_rewards)
        import matplotlib.pyplot as plt
        plt.boxplot([trained_rewards, random_rewards], labels=['Trained', 'Random'])
        plt.title('Performance Comparison')
        plt.ylabel('Cumulative Reward')
        plt.show()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-utility_function', default="power", type=str)
    parser.add_argument('-p', default=0.5, type=int)
    parser.add_argument('-dataset', default='Blackscholes', type=str)  # 'Blackscholes', 'Heston', 'VarianceGamma', 'Kou_Jump_Diffusion', 'Levy_Ito', 'YFinance'
    parser.add_argument('-actor_dataset', default='Blackscholes', type=str)  # An Actor ID to determine which actor will be loaded (if it exists), then trained or tested/evaluated on
    parser.add_argument('-risk_free_rate', default=0.025, type=int)
    parser.add_argument('-window_size', default=1000, type=int)
    parser.add_argument('-num_paths', default=1000, type=int)
    parser.add_argument('-total_timesteps', default=1000, type=int)
    parser.add_argument('-num_episodes', default=10, type=int)
    parser.add_argument('-mode', default='compare', type=str)  # 'train' 'test' 'eval' 'compare'

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