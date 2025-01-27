import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from help_fct import analytical_solutions, analytical_entry_wealth_offset


def test_actor(args, data_params, model, env):
    obs = env.reset()  # obs, info = env.reset(test=True)
    total_reward = 0
    average_riskfree_action, average_risky_action = [], []
    actions = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated = env.step(action)  #obs, reward, done, truncated, info = env.step(action)
        actions.append(action[0][0]) if not done else print(actions)
        #total_reward += reward
        average_risky_action.append(action[0])
        average_riskfree_action.append(sum(action[1:])) if args.num_paths > 1 else average_riskfree_action.append(1 - action[0])

    analytical_risky_action, analytical_utility = analytical_solutions(args, data_params)

    # print("Total reward during test:", total_reward)
    print("Average Riskfree Action:", np.mean(average_riskfree_action))
    print("Average Risky Action:", np.mean(average_risky_action))
    print("Analytical Risky Action:", sum(analytical_risky_action))
    print("Analytical Riskfree Action:", 1-sum(analytical_risky_action))
    print("Analytical expected Utility:", analytical_utility)


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

    plt.plot(portfolio_values.T, color="blue")
    plt.plot(range(args.window_size - 1), mean_portfolio_value, color="red")
    plt.title("Portfolio Value Episodes")
    plt.xlabel("Time")
    plt.ylabel("Portfolio Value")
    #plt.legend()
    plt.show()


def compare_actor(args, data_params, actor, env):
    trained_cum_rewards, random_cum_rewards, trained_portfolio, random_portfolio = [], [], [], []
    average_risky_action = []
    first = True
    analytical_risky_action, analytical_utility = analytical_solutions(args, data_params)
    for episode in tqdm(range(args.num_episodes), desc="Episodes", leave=False):
        for random_actor in [False, True]:
            obs, info = env.reset(random_actor=random_actor)
            portfolio_value = [1.0]
            episode_reward = 0
            done = False
            while not done:
                if random_actor:  # or perfect_actor, or null-actor
                    #action = env.action_space.sample()
                    #action = np.zeros(args.num_paths+1, dtype=np.float32)
                    action = analytical_risky_action
                else:
                    action, _ = actor.predict(obs, deterministic=True)
                    average_risky_action.append(action[0])
                    if first:
                        entry_wealth_offset = analytical_entry_wealth_offset(action, args, data_params)
                        first = False

                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
                #if not done:
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

    perfect_portfolio = random_portfolio
    perfect_portfolio[perfect_portfolio < 0] = 0
    power_utility = (perfect_portfolio[:, -1] ** (1 - args.p))

    analytical_entry_wealth_offset(action, args, data_params)

    print(f"Trained Actor Average Portfolio: {np.mean(trained_portfolio[:,-1])}")
    print(f"Random Actor Average Portfolio: {np.mean(random_portfolio[:,-1])}")
    print("_____")
    print(f"Trained Actor Average Terminal Reward: {np.mean(trained_cum_rewards)}")
    print(f"Random Actor Average Terminal Reward: {np.mean(random_cum_rewards)}")
    print("Analytical Perfect Utility:", analytical_utility)
    print("Analytical minus Simulated Perfect Utility:", analytical_utility - np.mean(power_utility))
    print("_____")
    print("Additional Entry-Wealth to offset Error:", analytical_utility/np.mean(trained_cum_rewards))
    print("Analytically: Additional Entry-Wealth to offset Error:", entry_wealth_offset)
    print("_____")
    print("Analytical Risky Action:", analytical_risky_action)
    print("Average Risky Action:", np.mean(average_risky_action))

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].plot(trained_portfolio.T, color="blue", linewidth=0.3)
    axs[0].plot(range(trained_portfolio.shape[1]), trained_portfolio.mean(axis=0), color="red")
    axs[0].set_title("Trained Portfolio")
    axs[0].set_ylabel("Portfolio Value")

    axs[1].plot(random_portfolio.T, color="blue", linewidth=0.3)
    axs[1].plot(range(random_portfolio.shape[1]), random_portfolio.mean(axis=0), color="red")
    axs[1].set_title("Random Portfolio")
    axs[1].set_ylabel("Portfolio Value")

    #axs[2].boxplot([trained_cum_rewards, random_cum_rewards], labels=["Trained Actor", "Random Actor"],)
    #axs[2].set_title("Performance Comparison")
    #axs[2].set_ylabel("Cumulative Reward")

    from scipy.stats import gaussian_kde

    axs[2].hist(power_utility.flatten(), bins=50, color="green", alpha=0.7)
    axs[2].axvline(np.mean(power_utility), color="red", linestyle="--", linewidth=2, label="Perfect Actor")
    kde_power = gaussian_kde(power_utility.flatten())
    x_vals_power = np.linspace(min(power_utility.flatten()), max(power_utility.flatten()), 500)
    axs[2].plot(x_vals_power, kde_power(x_vals_power) * len(power_utility.flatten()) * (
                max(power_utility.flatten()) - min(power_utility.flatten())) / 50, color="orange", label="Density")
    axs[2].set_title("Power Utility of Perfect Portfolio")
    axs[2].set_ylabel("Frequency")
    axs[2].set_xlabel("Utility")
    axs[2].legend()

    axs[3].hist(trained_cum_rewards, bins=50, color="blue", alpha=0.7, label="Trained Actor")
    axs[3].axvline(random_cum_rewards[0], color="red", linestyle="--", linewidth=2, label="Perfect Actor")
    axs[3].axvline(np.mean(trained_cum_rewards), color="green", linestyle="--", linewidth=2, label="Mean Trained Actor")
    kde_trained = gaussian_kde(trained_cum_rewards)
    x_vals = np.linspace(min(trained_cum_rewards), max(trained_cum_rewards), 500)
    axs[3].plot(x_vals, kde_trained(x_vals) * len(trained_cum_rewards) * (
                max(trained_cum_rewards) - min(trained_cum_rewards)) / 50, color="orange", label="Density")
    axs[3].set_title("Performance Comparison")
    axs[3].set_ylabel("Frequency")
    axs[3].set_xlabel("Deviation from Perfect Actor")
    axs[3].legend()

    plt.tight_layout()
    plt.show()

