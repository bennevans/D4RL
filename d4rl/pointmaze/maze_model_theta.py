""" A pointmass maze env."""
from gym.envs.mujoco import mujoco_env
from gym import utils
from d4rl import offline_env
from d4rl.pointmaze.dynamic_mjc import MJCModel
import numpy as np
import random


WALL = 10
EMPTY = 11
GOAL = 12


def parse_maze(maze_str):
    lines = maze_str.strip().split('\\')
    width, height = len(lines), len(lines[0])
    maze_arr = np.zeros((width, height), dtype=np.int32)
    for w in range(width):
        for h in range(height):
            tile = lines[w][h]
            if tile == '#':
                maze_arr[w][h] = WALL
            elif tile == 'G':
                maze_arr[w][h] = GOAL
            elif tile == ' ' or tile == 'O' or tile == '0':
                maze_arr[w][h] = EMPTY
            else:
                raise ValueError('Unknown tile type: %s' % tile)
    return maze_arr

OBSCURE_CENTER = 'center'
OBSCURE_3 = '3'

ANGLE_ACCEL = 'angle_accel'
XY_ACCEL = 'xy_accel'

def point_maze(maze_str, time_step="0.01", integrator="Euler", obscure_mode=OBSCURE_3, control_mode=ANGLE_ACCEL):
    maze_arr = parse_maze(maze_str)

    mjcmodel = MJCModel('point_maze')
    mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
    # mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
    mjcmodel.root.option(timestep=time_step, gravity="0 0 0", iterations="20", integrator=integrator)
    default = mjcmodel.root.default()
    default.joint(damping=1, limited='false')
    default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

    asset = mjcmodel.root.asset()
    asset.texture(type="2d",name="groundplane",builtin="checker",rgb1="0.2 0.3 0.4",rgb2="0.1 0.2 0.3",width=100,height=100)
    asset.texture(name="skybox",type="skybox",builtin="gradient",rgb1=".4 .6 .8",rgb2="0 0 0",
               width="800",height="800",mark="random",markrgb="1 1 1")
    asset.material(name="groundplane",texture="groundplane",texrepeat="20 20")
    asset.material(name="wall",rgba=".7 .5 .3 1")
    asset.material(name="target",rgba=".6 .3 .3 1")

    visual = mjcmodel.root.visual()
    visual.headlight(ambient=".4 .4 .4",diffuse=".8 .8 .8",specular="0.1 0.1 0.1")
    visual.map(znear=.01)
    visual.quality(shadowsize=2048)

    worldbody = mjcmodel.root.worldbody()
    worldbody.geom(name='ground',size="40 40 0.25",pos="0 0 -0.1",type="plane",contype=1,conaffinity=0,material="groundplane")

    top_camera = worldbody.camera(name='topview', pos=[3.0,3.0,7], fovy=30, xyaxes=[1,0,0,0,1,0])

    particle = worldbody.body(name='particle', pos=[1.2,1.2,0])
    particle.geom(name='particle_geom', type='sphere', size=0.1, rgba='0.0 0.0 1.0 0.0', contype=1)
    particle.site(name='particle_site', pos=[0.0,0.0,0], size=0.2, rgba='0.3 0.6 0.3 1')

    fpv_camera = particle.camera(name="fpv", pos=[0.0,0.0,0.0], xyaxes=[0,-1,0,0,0,1], fovy=60)
    top_rotate_camera = particle.camera(name="top_rotate", pos=[0.0,0.0,7.0], xyaxes=[1,0,0,0,1,0], fovy=60)


    if control_mode == ANGLE_ACCEL:
        particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
        particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])
        particle.joint(name='ball_rot', type='hinge', pos=[0,0,0], axis=[0,0,1])
    elif control_mode == XY_ACCEL:
        particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
        particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

    worldbody.site(name='target_site', pos=[0.0,0.0,0], size=0.2, material='target')

    width, height = maze_arr.shape
    for w in range(width):
        for h in range(height):
            if maze_arr[w,h] == WALL:
                worldbody.geom(conaffinity=1,
                               type='box',
                               name='wall_%d_%d'%(w,h),
                               material='wall',
                               pos=[w+1.0,h+1.0,0],
                               size=[0.5,0.5,0.2])


    actuator = mjcmodel.root.actuator()
    if control_mode == ANGLE_ACCEL:
        actuator.velocity(joint="ball_rot", ctrllimited=False)
        actuator.motor(joint="ball_x", ctrllimited=False)
    elif control_mode == XY_ACCEL:
        actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)
        actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], ctrllimited=True, gear=100)

    if obscure_mode == OBSCURE_CENTER:
        worldbody.geom(conaffinity=1,
                          type='box',
                            name='obscure1',
                            material='wall',
                            pos=[width/2.0+0.5,height/2.0+1.5,0.5],
                            size=[0.5,0.5,0.2])
    elif obscure_mode == OBSCURE_3:
        worldbody.geom(conaffinity=1,
                          type='box',
                            name='obscure1',
                            material='wall',
                            pos=[width/2.0+0.5,height/2.0+1.5,0.5],
                            size=[0.5,0.5,0.2])
        worldbody.geom(conaffinity=1,
                          type='box',
                            name='obscure2',
                            material='wall',
                            pos=[width/2.0 - 0.5,height/2.0-0.5,0.5],
                            size=[0.5,0.5,0.2])
        worldbody.geom(conaffinity=1,
                          type='box',
                            name='obscure3',
                            material='wall',
                            pos=[width/2.0+1.5,height/2.0-0.5,0.5],
                            size=[0.5,0.5,0.2])

    return mjcmodel


LARGE_MAZE = \
        "############\\"+\
        "#OOOO#OOOOO#\\"+\
        "#O##O#O#O#O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O####O###O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "##O#O#O#O###\\"+\
        "#OO#OOO#OGO#\\"+\
        "############"

LARGE_MAZE_EVAL = \
        "############\\"+\
        "#OO#OOO#OGO#\\"+\
        "##O###O#O#O#\\"+\
        "#OO#O#OOOOO#\\"+\
        "#O##O#OO##O#\\"+\
        "#OOOOOO#OOO#\\"+\
        "#O##O#O#O###\\"+\
        "#OOOO#OOOOO#\\"+\
        "############"

MEDIUM_MAZE = \
        '########\\'+\
        '#OO##OO#\\'+\
        '#OO#OOO#\\'+\
        '##OOO###\\'+\
        '#OO#OOO#\\'+\
        '#O#OO#O#\\'+\
        '#OOO#OG#\\'+\
        "########"

MEDIUM_MAZE_EVAL = \
        '########\\'+\
        '#OOOOOG#\\'+\
        '#O#O##O#\\'+\
        '#OOOO#O#\\'+\
        '###OO###\\'+\
        '#OOOOOO#\\'+\
        '#OO##OO#\\'+\
        "########"

SMALL_MAZE = \
        "######\\"+\
        "#OOOO#\\"+\
        "#O##O#\\"+\
        "#OOOO#\\"+\
        "######"

U_MAZE = \
        "#####\\"+\
        "#GOO#\\"+\
        "###O#\\"+\
        "#OOO#\\"+\
        "#####"

U_MAZE_EVAL = \
        "#####\\"+\
        "#OOG#\\"+\
        "#O###\\"+\
        "#OOO#\\"+\
        "#####"

OPEN = \
        "#######\\"+\
        "#OOOOO#\\"+\
        "#OOGOO#\\"+\
        "#OOOOO#\\"+\
        "#######"


class MazeEnv(mujoco_env.MujocoEnv, utils.EzPickle, offline_env.OfflineEnv):
    def __init__(self,
                 maze_spec=U_MAZE,
                 reward_type='dense',
                 reset_target=False,
                 frame_skip=1,
                 time_step="0.01",
                 integrator="Euler",
                 obscure_mode=OBSCURE_3,
                 control_mode=ANGLE_ACCEL,
                 theta_scale=2000,
                 x_scale=1000,
                 **kwargs):
        offline_env.OfflineEnv.__init__(self, **kwargs)

        self.render_mode = None
        self.reset_target = reset_target
        self.str_maze_spec = maze_spec
        self.maze_arr = parse_maze(maze_spec)
        self.reward_type = reward_type
        self.reset_locations = list(zip(*np.where(self.maze_arr == EMPTY)))
        self.reset_locations.sort()
        self.frame_skip = frame_skip
        self.time_step = time_step
        self.integrator = integrator
        self.theta_scale = theta_scale
        self.x_scale = x_scale

        self._target = np.array([0.0,0.0])
        print(time_step, integrator)
        model = point_maze(maze_spec, time_step=self.time_step, integrator=self.integrator, obscure_mode=obscure_mode, control_mode=control_mode)
        with model.asfile() as f:
            print(f.name)
            # import ipdb; ipdb.set_trace()
            mujoco_env.MujocoEnv.__init__(self, model_path=f.name,
                                          frame_skip=frame_skip)
        utils.EzPickle.__init__(self)

        # Set the default goal (overriden by a call to set_target)
        # Try to find a goal if it exists
        self.goal_locations = list(zip(*np.where(self.maze_arr == GOAL)))
        if len(self.goal_locations) == 1:
            self.set_target(self.goal_locations[0])
        elif len(self.goal_locations) > 1:
            raise ValueError("More than 1 goal specified!")
        else:
            # If no goal, use the first empty tile
            self.set_target(np.array(self.reset_locations[0]).astype(self.observation_space.dtype))
        self.empty_and_goal_locations = self.reset_locations + self.goal_locations

        self.frame_skip = frame_skip

        # import ipdb; ipdb.set_trace()
        self.action_space.low = np.array([-1.0, -1.0])
        self.action_space.high = np.array([1.0, 1.0])

    def step(self, action): # [change in theta and acceleration in x]

        # action = np.clip(action, -1.0, 1.0)
        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        theta = qpos[2]
        xdot, ydot = qvel[0:2]

        theta_vel = action[0] * self.theta_scale
        x_accel = action[1] * self.x_scale

        # print('theta_vel: ', theta_vel)
        # print('x_accel: ', x_accel)
        # print('qpos', qpos)
        # print('qvel', qvel)

        dx = xdot + x_accel * np.cos(theta) * self.dt
        dy = ydot + x_accel * np.sin(theta) * self.dt
        target_qvel = np.array([dx, dy, theta_vel])
        self.set_state(qpos, target_qvel)

        # print('target_qvel', target_qvel)

        self.clip_velocity()
        # self.do_simulation(action, self.frame_skip)
        self.sim.step()
        self.set_marker()
        ob = self._get_obs()
        if self.reward_type == 'sparse':
            reward = 1.0 if np.linalg.norm(ob[0:2] - self._target) <= 0.5 else 0.0
        elif self.reward_type == 'dense':
            reward = np.exp(-np.linalg.norm(ob[0:2] - self._target))
        else:
            raise ValueError('Unknown reward type %s' % self.reward_type)
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel, self._target]).ravel()

    def get_target(self):
        return self._target

    def set_target(self, target_location=None):
        if target_location is None:
            idx = self.np_random.choice(len(self.empty_and_goal_locations))
            reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
            target_location = reset_location + self.np_random.uniform(low=-.1, high=.1, size=len(reset_location))
        self._target = target_location

    def set_marker(self):
        self.data.site_xpos[self.model.site_name2id('target_site')] = np.array([self._target[0]+1, self._target[1]+1, 0.0])

    def clip_velocity(self):
        qvel = np.clip(self.sim.data.qvel, -5.0, 5.0)
        self.set_state(self.sim.data.qpos, qvel)

    def reset_model(self):
        idx = self.np_random.choice(len(self.empty_and_goal_locations))
        reset_location = np.array(self.empty_and_goal_locations[idx]).astype(self.observation_space.dtype)
        reset_angle = self.np_random.uniform(low=-np.pi, high=np.pi)
        reset_location = np.concatenate([reset_location, [reset_angle]])
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        if self.reset_target:
            self.set_target()
        return self._get_obs()

    def reset_to_location(self, location):
        self.sim.reset()
        reset_location = np.array(location).astype(self.observation_space.dtype)
        qpos = reset_location + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass

if __name__ == '__main__':
    import gym
    print_xml = False
    if print_xml:
        model = point_maze(U_MAZE)
        with model.asfile() as f:
            for line in f.file.readlines():
                print(line, end='')
    else:
        # env = MazeEnv(maze_spec=U_MAZE, reward_type='dense', obscure_mode=None)
        env = gym.make("maze2d-theta-umaze-v0", reward_type='dense', obscure_mode=None)
        env.reset()
        forward_action = np.array([0.0, 0.1])
        rotate_action = np.array([1.0, 0.0])
        zero_action = np.array([0.0, 0.0])
        while True:
            for i in range(100):
                env.render(camera_name="fpv")
                env.step(forward_action)
                print('fwd', env.data.qpos, env.data.qvel)

            for i in range(100):
                env.render(camera_name="fpv")
                env.step(-forward_action)
                print('fwd', env.data.qpos, env.data.qvel)
            
            for i in range(100):
                env.render()
                env.step(zero_action)
                print('stop', i, env.data.qpos, env.data.qvel)

            for i in range(100):
                env.render(camera_name="fpv")
                env.step(rotate_action)
                print('turn', i, env.data.qpos, env.data.qvel)