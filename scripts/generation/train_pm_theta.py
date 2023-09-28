# import gymnasium as gym
import gym

from stable_baselines3 import PPO
from d4rl.pointmaze.maze_model_theta import MazeEnv
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv


if __name__ == '__main__':
    # env = gym.make('maze2d-theta-umaze-v0')
    vec_env = make_vec_env('maze2d-theta-umaze-v0', n_envs=64, vec_env_cls=SubprocVecEnv)
    # print(env.reward_type)
    print(vec_env)
    policy_kwargs = dict(log_std_init=-1.5, net_arch=dict(pi=[128, 128], vf=[512, 512]))
    model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs, batch_size=256, n_steps=1024, n_epochs=20, tensorboard_log="./logs")
    model.learn(total_timesteps=10_000_000)
    model.save("theta_10m_20ep_global_std-1.5")
    # vec_env = model.get_env()
    # obs = vec_env.reset()
    # for i in range(1000):
    #     # action = env.action_space.sample()
    #     # obs, reward, done, info = vec_env.step([action])
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = vec_env.step(action)
    #     vec_env.render()
    #     # VecEnv resets automatically
    #     # if done:
    #     #   obs = env.reset()

    # vec_env.close()
