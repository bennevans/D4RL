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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render trajectories')
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--env_name', type=str, default='maze2d-umaze-v1', help='Maze type')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    args = parser.parse_args()

    frame_skip = 3
    p_gain = 5.0
    d_gain = -1
    solve_thresh = 0.1
    time_step = "0.02"
    integrator = "RK4"

    env = gym.make(args.env_name, frame_skip=frame_skip, integrator=integrator, time_step=time_step)
    maze = env.str_maze_spec
    max_episode_steps = env._max_episode_steps

    controller = waypoint_controller.WaypointController(maze, solve_thresh=solve_thresh, p_gain=p_gain, d_gain=d_gain)
    env = maze_model.MazeEnv(maze, frame_skip=frame_skip, integrator=integrator, time_step=time_step)

    env.set_target()
    s = env.reset()
    act = env.action_space.sample()
    done = False

    data = reset_data()
    ts = 0
    rand_act_prob = 0.3
    zero_act_prob = 0.1
    regular_act_prob = 1 - rand_act_prob - zero_act_prob

    im_shape = (150, 150)
    images = np.zeros((args.num_samples, *im_shape, 3), dtype=np.uint8)

    for i in tqdm(range(args.num_samples)):
        position = s[0:2]
        velocity = s[2:4]
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
        append_data(data, s, act, env._target, done, env.sim.data, random)

        rgb = env.render(mode='rgb_array', camera_name='topview')
        resized = cv2.resize(rgb, im_shape, interpolation=cv2.INTER_CUBIC)
        images[i] = resized


        ns, _, _, _ = env.step(act)

        if len(data['observations']) % 10000 == 0:
            print(len(data['observations']))

        ts += 1
        if done:
            env.set_target()
            done = False
            ts = 0
        else:
            s = ns
        """
        # if args.render:
        # env.render()
        rgb = env.render(mode='rgb_array', camera_name='topview')
        # w = 256
        # h = 256

        # # center crop
        # center = rgb.shape
        # x = center[1]/2 - w/2
        # y = center[0]/2 - h/2

        # rgb = rgb[int(y):int(y+h), int(x):int(x+w)]
        resized = cv2.resize(rgb, im_shape, interpolation=cv2.INTER_CUBIC)
        # plt.imshow(rgb)
        # plt.figure()
        # plt.imshow(resized)
        # plt.show()
        images[i] = resized
            # plt.imsave('{}.png'.format(im_shape[0]), resized)
            # # plt.show()
            # exit()
        """
    
    if args.noisy:
        fname = '%s-noisy.hdf5' % args.env_name
    else:
        fname = '{}-{}-fs-{}-p-{}-d-{}-s-{}-ts-{}-int-{}-pre.hdf5'.format(args.env_name, args.num_samples, frame_skip, p_gain, d_gain, solve_thresh, time_step, integrator)
    dataset = h5py.File(fname, 'w')
    npify(data)
    for k in data:
        dataset.create_dataset(k, data=data[k], compression='gzip')

    dataset.create_dataset('images', data=images, compression='gzip', chunks=True)

if __name__ == "__main__":
    main()
