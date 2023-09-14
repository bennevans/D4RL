# import gymnasium as gym
import gym

from stable_baselines3 import PPO
from d4rl.pointmaze.maze_model_theta import MazeEnv

env = gym.make('maze2d-theta-umaze-v0')
print(env.reward_type)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    # action = env.action_space.sample()
    # obs, reward, done, info = vec_env.step([action])
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()