import pandas as pd
import numpy as np

def monitor_plot():
    log_data = pd.read_csv("log.monitor.csv", skiprows=1)  # Skip the header
    rewards = log_data["r"]  # Rewards per episode
    episode_lengths = log_data["l"]  # Episode lengths

    rolling_mean = pd.Series(rewards).rolling(window=40).mean()
    time = np.arange(len(rewards))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(time, rewards, label="Episode Reward")
    plt.plot(time, rolling_mean, label=f"Rolling Mean", color='orange', linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    monitor_plot()