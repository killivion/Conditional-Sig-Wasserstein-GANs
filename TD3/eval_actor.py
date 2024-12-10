from td3_train import pull_data
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def test_actor(args, data_params, model, env):
    obs, info = env.reset()
    total_reward = 0
    average_riskfree_action, average_risky_action = [], []
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        average_riskfree_action.append(action[0])
        average_risky_action.append(sum(action[1:]))
        print(action)
    print("Total reward during test:", total_reward)
    print("Average Riskfree Action:", np.mean(average_riskfree_action))
    print("Average Risky Action:", np.mean(average_risky_action))

    cholesky = np.linalg.cholesky(data_params['data_params']['sigma_cov'])
    analytical_risky_action = 1/args.p * (data_params['data_params']['mu']-args.risk_free_rate) @ ((cholesky @ cholesky.T) ** (-1))
    print("Analytical Risky Action:", sum(analytical_risky_action))
    print("Analytical Riskfree Action:", 1-sum(analytical_risky_action))


def evaluate_actor(args, data_params, model, env):
    portfolio_values = []
    for episode in tqdm(range(args.num_episodes), desc="Episodes", leave=False):
        portfolio_value = [1.0]  # Initial portfolio value
        obs, info = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            if not done:
                portfolio_value.append(info)
        portfolio_values.append(portfolio_value)

    portfolio_values = np.array(portfolio_values)
    mean_portfolio_value = portfolio_values.mean(axis=0)

    print(f"Trained Actor Average Portfolio: {mean_portfolio_value}")

    plt.plot(range(args.window_size-1), mean_portfolio_value, color="red")
    plt.plot(portfolio_values.T, color="blue")
    plt.title("Portfolio Value Episodes")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    #plt.legend()
    plt.show()


def compare_actor(args, data_params, actor, env):
    trained_cum_rewards, random_cum_rewards, trained_portfolio, random_portfolio = [], [], [], []
    for episode in tqdm(range(args.num_episodes), desc="Episodes", leave=False):
        for random_actor in [False, True]:
            obs, info = env.reset()
            portfolio_value = [1.0]
            episode_reward = 0
            done = False
            while not done:
                if random_actor:  # the null investor
                    action = env.action_space.sample()  # np.zeros(args.num_paths+1, dtype=np.float32)
                else:
                    action, _ = actor.predict(obs, deterministic=True)

                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                if not done:
                    portfolio_value.append(info)

            if random_actor:
                random_cum_rewards.append(episode_reward)
                random_portfolio.append(portfolio_value)
            else:
                trained_cum_rewards.append(episode_reward)
                trained_portfolio.append(portfolio_value)

    #return trained_cum_rewards, random_cum_rewards, trained_portfolio_values, random_portfolio_values

    trained_portfolio = np.array(trained_portfolio)
    random_portfolio = np.array(random_portfolio)

    print(f"Trained Actor Average Terminal Reward: {np.mean(trained_cum_rewards)}")
    print(f"Random Actor Average Terminal Reward: {np.mean(random_cum_rewards)}")
    print(f"Trained Actor Average Portfolio: {np.mean(trained_portfolio[:,-1])}")
    print(f"Random Actor Average Portfolio: {np.mean(random_portfolio[:,-1])}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].plot(trained_portfolio.T, color="blue", linewidth=0.3)
    axs[0].plot(range(args.window_size-1), trained_portfolio.mean(axis=0), color="red")
    axs[0].set_title("Trained Portfolio")
    axs[0].set_ylabel("Portfolio Value")

    axs[1].plot(random_portfolio.T, color="blue", linewidth=0.3)
    axs[1].plot(range(args.window_size-1), random_portfolio.mean(axis=0), color="red")
    axs[1].set_title("Random Portfolio")
    axs[1].set_ylabel("Portfolio Value")

    axs[2].boxplot([trained_cum_rewards, random_cum_rewards], labels=["Trained Actor", "Random Actor"],)
    axs[2].set_title("Performance Comparison")
    axs[2].set_ylabel("Cumulative Reward")

    plt.tight_layout()
    plt.show()
