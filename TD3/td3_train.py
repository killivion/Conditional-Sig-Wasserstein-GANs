from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

import numpy as np
import os
import torch
import optuna
from functools import partial

import eval_actor
from track_learning import monitor_plot
from help_fct import find_largest_td3_folder, ActionLoggingCallback, generate_random_params, pull_data, fuse_folders, analytical_solutions, action_normalizer, expected_utility, analytical_entry_wealth_offset
from portfolio_env import PortfolioEnv
from hyperparameter_tuning import optimize_td3, test_optimized_td3

"""
tensorboard --logdir ./TD3/logs

Terminal: 
$env:PYTHONPATH="."
python TD3/td3_train.py -mode train -model_ID 3 -total_timesteps 50000 -allow_lending False
tensorboard --logdir ./logs
"""

def main(args, i=0):
    from train import get_dataset_configuration
    if args.dataset == 'correlated_Blackscholes':
        mu, vola_matrix = generate_random_params(args.num_paths, args.num_bm)
        spec = 'args.num_paths={}_window_size={}'.format(args.num_paths, args.window_size)
        data_params = dict(data_params=dict(mu=mu, vola_matrix=vola_matrix, window_size=args.window_size, num_paths=args.num_paths, num_bm=args.num_bm, grid_points=args.grid_points))
    else:
        generator = get_dataset_configuration(args.dataset, window_size=args.window_size, num_paths=args.num_paths, grid_points=args.grid_points)
        for s, d in generator:
            spec, data_params = s, d  # odd way to do it - easiest at the time, works in 1-d

    returns = pull_data(args, data_params)
    if args.mode == 'tuning':
        custom_args = {"args": args, "data_params": data_params, "returns": returns}
        optimize_func = partial(optimize_td3, **custom_args)
        study = optuna.create_study(direction="maximize")
        study.optimize(optimize_func, n_trials=args.n_trials, show_progress_bar=True)
        print("Best hyperparameters:", study.best_params)
    elif args.mode == 'test_tuning':
        test_optimized_td3(args, data_params, returns)
    else:
        run(args, spec, data_params, returns, i)


def run(args, spec, data_params, returns, i=0):
    print('Executing TD3 on %s, %s' % (args.dataset, spec))

    if args.mode == 'train':
        env = Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns))  # Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns), filename='log')
    else:
        env = PortfolioEnv(args=args, data_params=data_params, stock_data=returns)
    vec_env = DummyVecEnv([lambda: env])

    # Add action noise (exploration)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.action_noise_sigma * np.ones(n_actions))


    hardware = 'LRZ' if torch.cuda.is_available() else 'cpu'
    model_save_path = f"./agent/{args.model_ID}_{hardware}_td3_{args.actor_dataset}_assets_{args.num_paths}_window_{args.window_size}"

    # Load Model
    if os.path.exists(f"{model_save_path}.zip"):
        model = TD3.load(model_save_path)
        model.load_replay_buffer(f"{model_save_path}_buffer.pkl") if os.path.exists(f"{model_save_path}_buffer.pkl") else print("No replay buffer found; training from an empty buffer.")
        if os.path.exists(f"{model_save_path}_optimizer.pth"):
            optim_state = torch.load(f"{model_save_path}_optimizer.pth")
            model.actor.optimizer.load_state_dict(optim_state['actor_optimizer'])
            model.critic.optimizer.load_state_dict(optim_state['critic_optimizer'])
        else:
            print("No optimizer found; training from a new optimizer.")
        model.set_env(vec_env)
        print(f"Uses {model_save_path} trained on {model.num_timesteps}")
        already_trained_timesteps = model.num_timesteps
        model.learning_starts = 0
    else:
        print("No saved model found; starting new training.")
        model = TD3("MlpPolicy", vec_env, buffer_size=args.buffer_size, gamma=1, action_noise=action_noise, batch_size=args.batch_size, verbose=0, tensorboard_log="./logs/", train_freq=(args.train_freq, "episode"))
        model.learning_starts = args.total_timesteps / 5
        already_trained_timesteps = 0
    #model.verbose = 0 if hardware == 'cpu' else 0

    # Train, Test, Eval [Evaluate], Compare [with some benchmark]
    if args.mode == 'train':  # tensorboard --logdir ./TD3/logs
        tensorboard_path, number = find_largest_td3_folder(args)
        action_logging_callback = ActionLoggingCallback(log_dir=tensorboard_path)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True, tb_log_name=f"TD3_{number}", callback=action_logging_callback)
        fuse_folders(number, args)
        model.num_timesteps += already_trained_timesteps
        model.save(model_save_path)
        model.save_replay_buffer(f"{model_save_path}_buffer.pkl")
        torch.save({'actor_optimizer': model.actor.optimizer.state_dict(), 'critic_optimizer': model.critic.optimizer.state_dict()}, f"{model_save_path}_optimizer.pth")
        print(f"Model saved at: {model_save_path} with {model.num_timesteps} timesteps trained of which {already_trained_timesteps} were trained before")
        #if i == args.laps - 1:
        #    monitor_plot(args)

    if args.mode == 'test':
    #    print("Params:", data_params)
        eval_actor.test_actor(args, data_params, model, vec_env)
    elif args.mode == 'eval':
        eval_actor.evaluate_actor(args, data_params, model, env)
    elif args.mode == 'compare':
        eval_actor.compare_actor(args, data_params, model, env)
        # trained_rewards, random_rewards, trained_portfolio_values, random_portfolio_values

    elif args.mode == 'test_solution':
        analytical_risky_action, analytical_utility = analytical_solutions(args, data_params)
        print(f"Analytical Actions: {1-sum(analytical_risky_action), analytical_risky_action}, Analytical Utility: {analytical_utility}")
        print(f"Risky Fraciton is {sum(analytical_risky_action)}")
        if already_trained_timesteps > 0:
            obs, info = env.reset()
            action, _ = model.predict(obs, deterministic=True)
            norm_action = action_normalizer(action)
            print(f"Current action: {norm_action}, Expected Utility: {expected_utility(norm_action[1:], args, data_params)}")
            print(f"Risky Fraciton is {sum(norm_action[1:])}")

            print(f"Entry_Wealth_Offset: {analytical_entry_wealth_offset(action, args, data_params)}")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset', default='correlated_Blackscholes', type=str)  # 'Blackscholes', 'Heston', 'VarianceGamma', 'Kou_Jump_Diffusion', 'Levy_Ito', 'YFinance', 'correlated_Blackscholes'
    parser.add_argument('-actor_dataset', default='correlated_Blackscholes', type=str)  # An Actor ID to determine which actor will be loaded (if it exists), then trained or tested/evaluated on
    parser.add_argument('-utility_function', default="power", type=str)
    parser.add_argument('-allow_lending', action='store_true', help="Enable lending")

    parser.add_argument('-episode_reset', default=10000000, type=int)  #currently off
    #parser.add_argument('-learning_starts', default=100000, type=int)
    parser.add_argument('-p', default=0.8, type=float)
    parser.add_argument('-risk_free_rate', default=0.04, type=float)
    parser.add_argument('-grid_points', default=1, type=int)
    parser.add_argument('-window_size', default=1, type=int)
    parser.add_argument('-num_paths', default=2, type=int)
    parser.add_argument('-num_bm', default=2, type=int)  # Number of random sources N

    parser.add_argument('-action_noise_sigma', default=0.05, type=float)
    parser.add_argument('-train_freq', default=1, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-buffer_size', default=1000000, type=int)

    parser.add_argument('-total_timesteps', default=500, type=int)
    parser.add_argument('-num_episodes', default=10000, type=int)
    parser.add_argument('-n_trials', default=50, type=int)

    parser.add_argument('-model_ID', default=5, type=int)
    parser.add_argument('-laps', default=1, type=int)
    parser.add_argument('-mode', default='compare', type=str)  # 'train' 'compare' 'tuning' 'test_tuning' 'test_solution' # 'test' 'eval' are outdated

    args = parser.parse_args()
    if args.mode == 'train':
        for i in range(args.laps):
            #args.model_ID = 6 + i
            #print(f"This is lap {i+1} of {args.laps}")
            main(args, i)
    else:
        main(args)

    """
    Actor-Loss [small neg]: Large -> Instability, Close to 0: Yields high/good Q Values
    Critic-Loss: High -> Instability, Very low<0.02 -> Convergence / Overfitting
    """