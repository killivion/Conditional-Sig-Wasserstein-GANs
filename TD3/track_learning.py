import pandas as pd
import numpy as np

def monitor_plot(args, i, rewards):
    log_data = pd.read_csv("log.monitor.csv", skiprows=1)  # Skip the header
    reward = log_data["r"]  # Rewards per episode
    if i < args.laps:
        return rewards.append(reward)
    else:
        rolling_mean = pd.Series(rewards).rolling(window=50).mean()
        shortened_rewards = rewards[50:]
        shortened_rolling = rolling_mean[50:]
        time = np.arange(len(shortened_rewards))

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(time, shortened_rewards, label="Episode Reward")
        plt.plot(time, shortened_rolling, label=f"Rolling Mean", color='orange', linewidth=2)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Rewards per Episode")
        plt.legend()
        plt.show()
        return []

if __name__ == '__main__':
    monitor_plot()