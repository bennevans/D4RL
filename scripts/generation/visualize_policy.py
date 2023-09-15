
import gym
import d4rl
from stable_baselines3 import PPO
from generate_maze2d_datasets import sample_target

import numpy as np

if __name__ == '__main__':
    policy = "theta_1m_20ep.zip"
    env_name = "maze2d-theta-umaze-v0"

    # Load policy
    policy = PPO.load(policy)
    env = gym.make(env_name)

    # Visualize policy'

    done = False
    obs = env.reset()
    for i in range(10000):
        action, _states = policy.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        done = np.linalg.norm(obs[0:2] - env.env._target) <= 0.5

        if done:
            target_location = sample_target()
            env.set_target(target_location=target_location)
            done = False
            ts = 0

        env.render()