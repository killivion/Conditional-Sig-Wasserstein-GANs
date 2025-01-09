from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import time


def optimize_td3(trial, args, data_params, returns):
    start_time = time.time()

    #learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    #gamma = trial.suggest_float("gamma", 0.95, 0.99)
    # net_arch = trial.suggest_categorical("net_arch", [1, 2, 3])
    # net_arch_options = {1: [128, 128, 128], 2: [256, 256], 3: [400, 300]}
    # net_arch = net_arch_options[net_arch]
    #buffer_size = trial.suggest_categorical("buffer_size", [1000000, 2000000, 5000000, 10000000])

    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    tau = trial.suggest_float("tau", 0.001, 0.01)
    action_noise_std = trial.suggest_float("action_noise_std", 0.01, 0.2)

    train_freq = trial.suggest_int("train_freq", 1, 100)
    args.window_size = trial.suggest_int("window_size", 1, 50, log=True)
    args.grid_points = args.window_size

    env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns))
    vec_env = DummyVecEnv([lambda: env])
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_std * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_env,
                learning_starts=args.learning_starts, learning_rate=0.001, batch_size=batch_size, buffer_size=1000000,
                gamma=1, tau=tau, action_noise=action_noise, verbose=0, train_freq=(train_freq, "episode"))
    eval_callback = EvalCallback(vec_env, best_model_save_path="./evalLogs/",
                                 log_path="./evalLogs/", eval_freq=10000,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=20000*args.window_size, callback=eval_callback)

    print(f"Training time: {time.time() - start_time}")
    start_time = time.time()
    mean_reward, std = evaluate_policy(model, vec_env, n_eval_episodes=1000)
    print(f"Evaluation time: {time.time() - start_time}")
    print(f"Mean reward: {mean_reward}, Std reward: {std}, Window_size: {args.window_size}")

    #print(f"Mean reward: {mean_reward}, Std reward: {std}")
    return mean_reward

def test_optimized_td3(args, data_params, returns):
    best_model = TD3.load("./evalLogs/best_model.zip")

    env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns))  # Replace with your environment
    vec_env = DummyVecEnv([lambda: env])

    mean_reward, std_reward = evaluate_policy(best_model, vec_env, n_eval_episodes=1000, render=False)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")


    data = np.load("./evalLogs/evaluations.npz")

    timesteps = data["timesteps"]
    results = data["results"]  # Evaluation results (mean rewards)

    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, mean_rewards, label="Mean Reward", color="blue")
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        color="blue", alpha=0.2, label="Std Dev"
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation of TD3 Performance Over Time")
    plt.legend()
    plt.grid()
    plt.show()