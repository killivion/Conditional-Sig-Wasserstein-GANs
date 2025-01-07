from stable_baselines3 import TD3
import numpy as np
import os
import torch
import eval_actor
from track_learning import monitor_plot
from help_fct import find_largest_td3_folder, ActionLoggingCallback, generate_random_params, pull_data, fuse_folders

"""
tensorboard --logdir ./TD3/logs
tensorboard --logdir ./logs

$env:PYTHONPATH="."
python TD3/td3_train.py -mode train -model_ID 2 -total_timesteps 100000
"""

def main(args, i=0):
    from train import get_dataset_configuration
    if args.dataset == 'correlated_Blackscholes':
        mu, sigma_cov = generate_random_params(args.num_paths)
        spec = 'args.num_paths={}_window_size={}'.format(args.num_paths, args.window_size)
        data_params = dict(data_params=dict(mu=mu, sigma_cov=sigma_cov, window_size=args.window_size, num_paths=args.num_paths, grid_points=args.grid_points))
    else:
        generator = get_dataset_configuration(args.dataset, window_size=args.window_size, num_paths=args.num_paths, grid_points=args.grid_points)
        for s, d in generator:
            spec, data_params = s, d  # odd way to do it, works in 1-d

    returns = pull_data(args, data_params)
    run(args, spec, data_params, returns, i)


def run(args, spec, data_params, returns, i=0):
    print('Executing TD3 on %s, %s' % (args.dataset, spec))
    from portfolio_env import PortfolioEnv
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    env = PortfolioEnv(args=args, data_params=data_params, stock_data=returns)  # Monitor(PortfolioEnv(args=args, data_params=data_params, stock_data=returns), filename='log')
    vec_env = DummyVecEnv([lambda: env])

    # Add action noise (exploration)
    n_actions = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    hardware = 'LRZ' if torch.cuda.is_available() else 'cpu'
    model_save_path = f"./agent/{args.model_ID}_{hardware}_td3_{args.actor_dataset}_assets_{args.num_paths}_window_{args.window_size}"

    # Load Model
    if os.path.exists(f"{model_save_path}.zip"):
        model = TD3.load(model_save_path)
        model.set_env(vec_env)
        print(f"Uses {model_save_path} trained on {model.num_timesteps}")
        already_trained_timesteps = model.num_timesteps
    else:
        print("No saved model found; starting new training.")
        model = TD3("MlpPolicy", vec_env, action_noise=action_noise, batch_size=args.batch_size, verbose=0, learning_starts=1000, tensorboard_log="./logs/", train_freq=(args.train_freq, "episode"))
        already_trained_timesteps = 0
    #model.verbose = 0 if hardware == 'cpu' else 0

    # Train, Test, Eval [Evaluate], Compare [with some benchmark]
    if args.mode == 'train':  # tensorboard --logdir ./TD3/logs
        tensorboard_path, number = find_largest_td3_folder(args)
        action_logging_callback = ActionLoggingCallback(log_dir=tensorboard_path)
        model.learn(total_timesteps=args.total_timesteps, progress_bar=True, tb_log_name="TD3", callback=action_logging_callback)
        fuse_folders(number, args)
        model.num_timesteps += already_trained_timesteps
        model.save(model_save_path)
        print(f"Model saved at: {model_save_path} with {model.num_timesteps} timesteps trained of which {already_trained_timesteps} were trained before")
        #if i == args.laps - 1:
        #    monitor_plot(args)
    #if args.mode in ['test', 'train']:
    #    print("Params:", data_params)
    #    eval_actor.test_actor(args, data_params, model, vec_env)
    elif args.mode == 'eval':
        eval_actor.evaluate_actor(args, data_params, model, env)
    elif args.mode == 'compare':
        eval_actor.compare_actor(args, data_params, model, env)
        # trained_rewards, random_rewards, trained_portfolio_values, random_portfolio_values

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-utility_function', default="power", type=str)
    parser.add_argument('-episode_reset', default=10000000, type=int)  #currently off
    parser.add_argument('-p', default=0.8, type=float)
    parser.add_argument('-dataset', default='correlated_Blackscholes', type=str)  # 'Blackscholes', 'Heston', 'VarianceGamma', 'Kou_Jump_Diffusion', 'Levy_Ito', 'YFinance', 'correlated_Blackscholes'
    parser.add_argument('-actor_dataset', default='correlated_Blackscholes', type=str)  # An Actor ID to determine which actor will be loaded (if it exists), then trained or tested/evaluated on
    parser.add_argument('-risk_free_rate', default=0.04, type=float)
    parser.add_argument('-grid_points', default=1, type=int)
    parser.add_argument('-window_size', default=1, type=int)
    parser.add_argument('-num_paths', default=1, type=int)

    parser.add_argument('-train_freq', default=20, type=int)
    parser.add_argument('-total_timesteps', default=1000000, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-num_episodes', default=3000, type=int)

    parser.add_argument('-model_ID', default=2, type=int)
    parser.add_argument('-laps', default=1, type=int)
    parser.add_argument('-mode', default='compare', type=str)  # 'train' 'test' 'eval' 'compare'

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