from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor
import numpy as np
from portfolio_env import PortfolioEnv
from stable_baselines3.common.callbacks import EvalCallback


def optimize_td3(trial, args, data_params, returns):

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 256, 512])
    gamma = trial.suggest_float("gamma", 0.95, 0.99)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    net_arch = trial.suggest_categorical("net_arch", [1, 2, 3])
    net_arch_options = {1: [128, 128, 128], 2: [256, 256], 3: [400, 300]}
    net_arch = net_arch_options[net_arch]
    action_noise_std = trial.suggest_float("action_noise_std", 0.1, 0.5)

    train_freq = trial.suggest_int("train_freq", 1, 100)
    args.window_size = trial.suggest_int("window_size", 1, 252)
    args.grid_points = args.window_size

    env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns))  # Replace with your environment
    vec_env = DummyVecEnv([lambda: env])
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=action_noise_std * np.ones(n_actions))

    model = TD3("MlpPolicy", vec_env,
                learning_starts=10000, learning_rate=learning_rate, batch_size=batch_size,
                gamma=gamma, tau=tau, policy_kwargs={"net_arch": net_arch},
                action_noise=action_noise, verbose=0, train_freq=(train_freq, "episode"))
    eval_callback = EvalCallback(vec_env, best_model_save_path="./evalLogs/",
                                 log_path="./evalLogs/", eval_freq=10000,
                                 deterministic=True, render=False)
    model.learn(total_timesteps=50000, callback=eval_callback)

    mean_reward, std = evaluate_policy(model, vec_env, n_eval_episodes=50)
    print(f"Mean reward: {mean_reward}, Std reward: {std}, Window_size: {args.window_size}")
    #print(f"Mean reward: {mean_reward}, Std reward: {std}")
    return mean_reward

