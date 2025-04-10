#learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    #gamma = trial.suggest_float("gamma", 0.95, 0.99)
    # net_arch = trial.suggest_categorical("net_arch", [1, 2, 3])
    # net_arch_options = {1: [128, 128, 128], 2: [256, 256], 3: [400, 300]}
    # net_arch = net_arch_options[net_arch]
    #buffer_size = trial.suggest_categorical("buffer_size", [1000000, 2000000, 5000000, 10000000])

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import pandas as pd
from portfolio_env import PortfolioEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import time
import matplotlib.cm as cm


def optimize_td3(trial, args, data_params, returns, stock_data):
    start_time = time.time()

    batch_size = trial.suggest_categorical("batch_size", [64, 256, 512, 1024, 2048])
    tau = trial.suggest_float("tau", 0.001, 0.01)
    action_noise_std = trial.suggest_float("action_noise_std", 0.01, 0.2)
    train_freq = trial.suggest_int("train_freq", 1, 100)
    learning_starts_factor = trial.suggest_categorical("learning_starts_factor", [2, 4, 1000])
    #args.window_size = trial.suggest_int("window_size", 1, 50, log=True)

    args.grid_points = args.window_size
    total_timesteps = 20000 * args.window_size
    learning_starts = int(total_timesteps / learning_starts_factor)

    env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_returns=returns, stock_data=stock_data))
    vec_env = DummyVecEnv([lambda: env])
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_std * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_env, learning_starts=learning_starts, learning_rate=0.001, buffer_size=1000000, gamma=1, verbose=0,
        batch_size=batch_size,
        tau=tau,
        action_noise=action_noise,
        train_freq=(train_freq, "episode")
    )

    trial_name = f"trial_bs{batch_size}_tau{round(tau, 4)}_noise{round(action_noise_std, 3)}_trainfreq{train_freq}_learnfac{learning_starts_factor}"
    eval_callback = CustomEvalCallback(eval_env=vec_env, eval_freq=total_timesteps/10,
                                       log_path="./evalLogs/", verbose=0)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    eval_callback.save_all_results(f"./evalLogs/{trial_name}_results.csv")

    print(f"Training time: {time.time() - start_time}")
    mean_reward, std = evaluate_policy(model, vec_env, n_eval_episodes=1000)
    print(f"Mean reward: {mean_reward}, Std reward: {std}, Window_size: {args.window_size}")

    return mean_reward


class CustomEvalCallback(BaseCallback):
    global_best_mean_reward = -np.inf
    def __init__(self, eval_env, eval_freq, log_path="./evalLogs/", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.log_path = log_path
        self.results = []
        self.timesteps = []
        self.best_mean_reward = -np.inf  # Initialize best reward
        self.best_model_path = os.path.join(self.log_path, "best_model.zip")

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env, n_eval_episodes=500)

            # Save results
            self.results.append([mean_reward, std_reward])
            self.timesteps.append(self.n_calls)

            if self.verbose == 0:
                print(f"Evaluation at step {self.n_calls}: mean_reward={mean_reward:.2f}, std_reward={std_reward:.2f}")

            if self.n_calls == self.eval_freq*10:
                if mean_reward > CustomEvalCallback.global_best_mean_reward:
                    CustomEvalCallback.global_best_mean_reward = mean_reward
                    self.model.save(self.best_model_path)
                    if self.verbose == 0:
                        print(f"New best model with mean reward {mean_reward:.2f} saved")

        return True

    def _on_training_end(self):
        # Save all results to a CSV file for later analysis
        self.save_all_results()

    def save_all_results(self, file_path=None):
        if not file_path:
            file_path = os.path.join(self.log_path, f"trial_{self.model.seed}_results.csv")

        results_array = np.array(self.results)
        timesteps_array = np.array(self.timesteps)

        np.savetxt(
            file_path,
            np.column_stack((timesteps_array, results_array)),
            delimiter=",",
            header="timesteps,mean_reward,std_reward",
            comments=""
        )
        if self.verbose > 0:
            print(f"Results saved to {file_path}")


def test_optimized_td3(args, data_params, returns, stock_data):
    if os.path.exists("./evalLogs/trial_None_results.csv"):
        os.remove("./evalLogs/trial_None_results.csv")

    best_model = TD3.load("./evalLogs/best_model.zip")
    env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_returns=returns, stock_data=stock_data))
    vec_env = DummyVecEnv([lambda: env])
    mean_reward, std_reward = evaluate_policy(best_model, vec_env, n_eval_episodes=1000, render=False)
    print(best_model.batch_size, best_model.tau, best_model.train_freq)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")

    import glob
    result_files = glob.glob("./evalLogs/*_results.csv")

    bs_colors = {}
    color_map = cm.get_cmap("tab10")

    plt.figure(figsize=(10, 6))
    for file in result_files:

        trial_name = file.split("/")[-1].replace("_results.csv", "")
        bs_value = int(trial_name.split("_")[1].replace("bs", ""))
        if bs_value not in bs_colors:
            bs_colors[bs_value] = color_map(len(bs_colors) / 10)

        data = pd.read_csv(file)

        scale = range(len(data["timesteps"]))

        # timesteps = data["timesteps"]
        mean_rewards = data["mean_reward"]
        std_rewards = data["std_reward"]

        plt.plot(scale, mean_rewards, lw=1.5, color=bs_colors[bs_value]) #label=f"{trial_name}",
        """plt.fill_between(
            scale,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2
        )"""

    legend_handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=f"bs={bs}")
        for bs, color in bs_colors.items()
    ]
    plt.legend(handles=legend_handles, title="Batch Size (bs)")

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Evaluation of TD3 Performance Across Trials")
    #plt.legend()
    plt.grid()
    plt.show()
