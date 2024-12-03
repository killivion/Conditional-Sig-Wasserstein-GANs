from td3_train import pull_data
import numpy as np

def evaluate_actor(args, data_params, actor, env, num_episodes, random_actor=False):

    trained_cum_rewards = []
    random_cum_rewards = []

    for episode in range(num_episodes):
        returns = pull_data(data_params, args.dataset, args.risk_free_rate)
        for random_actor in [False, True]:
            obs, info = env.reset()
            obs = np.array(returns[0], dtype=np.float32)
            episode_reward = 0
            done = False
            while not done:
                if random_actor:
                    action = env.action_space.sample()  # Random action
                else:
                    action, _ = actor.predict(obs, deterministic=True)  # Trained actor action

                obs, reward, done, _, info = env.step(action)
                episode_reward += reward
            if random_actor:
                random_cum_rewards.append(episode_reward)
            else:
                trained_cum_rewards.append(episode_reward)

    return trained_cum_rewards, random_cum_rewards