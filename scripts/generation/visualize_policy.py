
import gym
import d4rl
from stable_baselines3 import PPO
from generate_maze2d_datasets import sample_target

import numpy as np
import imageio

from tqdm import tqdm

if __name__ == '__main__':
    policy = "theta_10k_20ep_global"
    env_name = "maze2d-theta-umaze-v0"

    # Load policy
    policy = PPO.load(policy)
    env = gym.make(env_name, obscure_mode=None, invisible_target=False)

    # Visualize policy'

    all_frames = []

    done = False
    obs = env.reset()
    for i in tqdm(range(200)):
        action, _states = policy.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        
        done = np.linalg.norm(obs[0:2] - obs[-2:]) <= 0.5

        print(i, rewards, done, obs[0:2], obs[-2:])
        if done:
            target_location = sample_target()
            env.set_target(target_location=target_location)
            done = False
            ts = 0

        rgb = env.render(mode='rgb_array', camera_name='topview')
        all_frames.append(rgb)
    
    imageio.mimwrite("policy.mp4", all_frames, fps=100)