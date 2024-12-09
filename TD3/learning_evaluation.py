import pandas as pd

def monitor_plot():
    log_data = pd.read_csv("monitor_log.csv", skiprows=1)  # Skip the header
    rewards = log_data["r"]  # Rewards per episode
    episode_lengths = log_data["l"]  # Episode lengths

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode")
    plt.legend()
    plt.show()