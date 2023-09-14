import gym
import logging
from d4rl.pointmaze import waypoint_controller
from d4rl.pointmaze import maze_model
import numpy as np
import pickle
import gzip
import h5py
import argparse
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os

from stable_baselines3 import PPO

def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            'rand_acts': [],
            }

def append_data(data, s, a, tgt, done, env_data, rand):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(0.0)
    data['terminals'].append(done)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())
    data['rand_acts'].append(rand)

def npify(data):
    for k in data:
        if k == 'terminals':
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)

def in_wall(x, y, buffer):
    left_of_wall = x < 2.4 + buffer
    right_of_wall = x > 1.2 - buffer
    under_wall = y < 2.4 + buffer
    return left_of_wall and right_of_wall and under_wall

def sample_target(buffer=0.1):
    x = np.random.uniform(0.4 + buffer, 3.2 - buffer)
    y = np.random.uniform(0.4 + buffer, 3.2 - buffer)

    if in_wall(x, y, buffer):
        return sample_target()
    return np.array([x, y])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-theta-umaze-v0', help='Maze type', choices=['maze2d-umaze-v1','maze2d-theta-umaze-v0'])
    parser.add_argument('--num_samples', type=int, default=int(500000), help='Num samples to collect')
    parser.add_argument('--fname', type=str, default=None, required=True, help='filename to save to')
    parser.add_argument('--policy', type=str, default=None, help='filename to load from')
    parser.add_argument('--camera', type=str, default='topview', help='Camera name', choices=['topview', 'fpv', 'top_rotate'])
    args = parser.parse_args()
    print('args.policy', args.policy)
    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name
    else:
        if args.fname is None:
            fname = '{}-{}-fs-{}-p-{}-d-{}-s-{}-ts-{}-int-{}-hw-{}-pre-cover.hdf5'.format(args.env_name, args.num_samples, frame_skip, p_gain, d_gain, solve_thresh, time_step, integrator, im_shape[0])
        else:
            fname = args.fname + '.hdf5'
        # fname = '{}-{}-fs-{}-p-{}-d-{}-s-{}-ts-{}-int-{}-hw-{}-noisy-cover.hdf5'.format(args.env_name, args.num_samples, frame_skip, p_gain, d_gain, solve_thresh, time_step, integrator, im_shape[0])

    if os.path.exists(fname):
        print('File exists, exiting.')
        exit(0)

    frame_skip = 5
    p_gain = 5.0
    d_gain = -1
    solve_thresh = 0.1
    time_step = "0.01"
    integrator = "Euler"

    env = gym.make(args.env_name, frame_skip=frame_skip, integrator=integrator, time_step=time_step)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    if args.policy is not None:
        model = PPO.load(args.policy)
    else:
        controller = waypoint_controller.WaypointController(maze, solve_thresh=solve_thresh, p_gain=p_gain, d_gain=d_gain)
    # env = maze_model.MazeEnv(maze, frame_skip=frame_skip, integrator=integrator, time_step=time_step)

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    rand_act_prob = 0.2
    zero_act_prob = 0.1
    regular_act_prob = 1 - rand_act_prob - zero_act_prob

    im_shape = (100, 100)
    images = np.zeros((args.num_samples, *im_shape, 3), dtype=np.uint8)

    for i in tqdm(range(args.num_samples)):
        position = s[0:2]
        velocity = s[2:4]
        if args.policy is not None:
            print(s.shape)
            print(env)
            act = model.predict(s)[0]
        else:
            act, done = controller.get_action(position, velocity, env._target)

        act = act + np.random.randn(*act.shape)*0.5

        type_act = np.random.choice(['regular', 'zero', 'random'], p=[regular_act_prob, zero_act_prob, rand_act_prob])
        if type_act == 'zero':
            act = np.zeros_like(act)
            random = False
        elif type_act == 'random':
            act = env.action_space.sample()
            random = True
        else:
            act = act
            random = False

        act = np.clip(act, -1.0, 1.0)
        if ts >= max_episode_steps:
            done = True

        if '_target' in dir(env):
            target = env._target
        else:
            target = env.env._target

        append_data(data, s, act, target, done, env.sim.data, random)

        rgb = env.render(mode='rgb_array', camera_name=args.camera)
        resized = cv2.resize(rgb, im_shape, interpolation=cv2.INTER_CUBIC)
        images[i] = resized

        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            target_location = sample_target()
            env.set_target(target_location=target_location)
            done = False
            ts = 0
        else:
            s = ns

        # rgb = env.render(mode='rgb_array', camera_name='topview')
        # resized = cv2.resize(rgb, im_shape, interpolation=cv2.INTER_CUBIC)
        # images[i] = resized

        
    
    
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

    dataset.create_dataset('images', data=images, compression='gzip', chunks=True)

    dataset.attrs['rand_act_prob'] = rand_act_prob
    dataset.attrs['zero_act_prob'] = zero_act_prob
    dataset.attrs['regular_act_prob'] = regular_act_prob
    dataset.attrs['frame_skip'] = frame_skip
    dataset.attrs['p_gain'] = p_gain
    dataset.attrs['d_gain'] = d_gain
    dataset.attrs['solve_thresh'] = solve_thresh
    dataset.attrs['time_step'] = time_step
    dataset.attrs['integrator'] = integrator
    dataset.attrs['num_samples'] = args.num_samples
    dataset.attrs['env_name'] = args.env_name
    dataset.attrs['image_size'] = im_shape
    

    dataset.close()

if __name__ == "__main__":
    main()
