from td3_train import pull_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_actor(args, data_params, model, env):
    portfolio_values = []
    for episode in tqdm(range(args.num_episodes), desc="Episodes", leave=False):
        portfolio_value = [1.0]  # Initial portfolio value
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            portfolio_value.append(portfolio_value[-1] * (1 + np.dot(action, obs)))
        portfolio_values.append(portfolio_value)

    portfolio_values = np.array(portfolio_values)
    mean_portfolio_value = portfolio_values.mean(axis=0)

    print(f"Trained Actor Average Portfolio: {mean_portfolio_value}")

    plt.plot(range(args.window_size), mean_portfolio_value, color="red")
    plt.plot(portfolio_values.T, color="blue")
    plt.title("Portfolio Value Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Portfolio Value")
    #plt.legend()
    plt.show()


def compare_actor(args, data_params, actor, env):
    trained_cum_rewards, random_cum_rewards, trained_portfolio, random_portfolio = [], [], [], []

    for episode in tqdm(range(args.num_episodes), desc="Episodes", leave=False):
        returns = pull_data(data_params, args.dataset, args.risk_free_rate)
        for random_actor in [False, True]:
            obs, info = env.reset()
            obs = np.array(returns[0], dtype=np.float32)
            portfolio_value = [1.0]
            episode_reward = 0
            done = False
            while not done:
                if random_actor:
                    action = np.zeros(returns.shape[1], dtype=np.float32)  # env.action_space.sample()
                else:
                    action, _ = actor.predict(obs, deterministic=True)

                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                portfolio_value.append(portfolio_value[-1] * (1 + np.dot(action, returns[env.unwrapped.current_step])))

            if random_actor:
                random_cum_rewards.append(episode_reward)
                random_portfolio.append(portfolio_value)
            else:
                trained_cum_rewards.append(episode_reward)
                trained_portfolio.append(portfolio_value)

    #return trained_cum_rewards, random_cum_rewards, trained_portfolio_values, random_portfolio_values

    trained_portfolio = np.array(trained_portfolio)
    random_portfolio = np.array(random_portfolio)

    print(f"Trained Actor Average Reward: {np.mean(trained_cum_rewards)}")
    print(f"Random Actor Average Reward: {np.mean(random_cum_rewards)}")
    print(f"Trained Actor Average Portfolio: {np.mean(trained_portfolio[:,-1])}")
    print(f"Random Actor Average Portfolio: {np.mean(random_portfolio[:,-1])}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(trained_portfolio.T, color="blue", linewidth=0.3)
    axs[0].plot(range(args.window_size), trained_portfolio.mean(axis=0), color="red")
    axs[0].set_title("Trained Portfolio")
    axs[0].set_ylabel("Portfolio Value")

    axs[1].plot(random_portfolio.T, color="blue", linewidth=0.3)
    axs[1].plot(range(args.window_size), random_portfolio.mean(axis=0), color="red")
    axs[1].set_title("Random Portfolio")
    axs[1].set_ylabel("Portfolio Value")

    axs[2].boxplot([trained_cum_rewards, random_cum_rewards], labels=["Trained Actor", "Random Actor"],)
    axs[2].set_title("Performance Comparison")
    axs[2].set_ylabel("Cumulative Reward")

    plt.tight_layout()
    plt.show()
