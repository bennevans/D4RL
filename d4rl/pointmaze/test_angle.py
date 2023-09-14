import mujoco_py
import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

model = mujoco_py.load_model_from_path("test.xml")
sim = mujoco_py.MjSim(model)
sim.reset()

sim.data.qpos[:] = np.array([0.0, 0.0, 0.78])

dt = sim.model.opt.timestep

def custom_step(sim, action, forward_scale=100.0, turn_scale=400.0):
    qpos = sim.data.qpos.copy()
    qvel = sim.data.qvel.copy()
    theta = qpos[2]
    xdot, ydot = qvel[0:2]

    theta_vel = action[0] * turn_scale
    x_accel = action[1] * forward_scale

    # print('theta_vel: ', theta_vel)
    # print('x_accel: ', x_accel)
    # print('qpos', qpos)
    # print('qvel', qvel)

    dx = xdot + x_accel * np.cos(theta) * dt
    dy = ydot + x_accel * np.sin(theta) * dt
    target_qvel = np.array([dx, dy, theta_vel])

    current_state = sim.get_state()
    current_state.qvel[:] = target_qvel

    sim.set_state(current_state)
    sim.step()
    # target_image = sim.render(255,255)
    # print('qpos after', sim.data.qpos)
    # print('target_qvel', target_qvel)
    # print('qvel after', sim.data.qvel)
    return sim.data.qpos.copy(), sim.data.qvel.copy()

forward_action = np.array([0.0, 1.0]) # theta_vel, x_accel
turn_action = np.array([1.0, 0.0]) # theta_vel, x_accel
zero_action = np.array([0.0, 0.0]) # theta_vel, x_accel

all_qpos = []
all_qvel = []
all_x_pos = []
all_x_mat = []
global_theta = []
for i in range(100):
    qpos, qvel = custom_step(sim, forward_action)
    all_qpos.append(qpos)
    all_qvel.append(qvel)
    all_x_pos.append(sim.data.body_xpos[1].copy())
    xmat = sim.data.body_xmat[1].copy()
    all_x_mat.append(xmat)
    r = R.from_matrix(xmat.reshape(3,3))
    global_theta.append(r.as_euler('xyz')[2])

for i in range(100):
    qpos, qvel = custom_step(sim, turn_action)
    all_qpos.append(qpos)
    all_qvel.append(qvel)
    all_x_pos.append(sim.data.body_xpos[1].copy())
    all_x_mat.append(sim.data.body_xmat[1].copy())
    xmat = sim.data.body_xmat[1].copy()
    r = R.from_matrix(xmat.reshape(3,3))
    global_theta.append(r.as_euler('xyz')[2])

for i in range(100):
    qpos, qvel = custom_step(sim, zero_action)
    all_qpos.append(qpos)
    all_qvel.append(qvel)
    all_x_pos.append(sim.data.body_xpos[1].copy())
    all_x_mat.append(sim.data.body_xmat[1].copy())
    xmat = sim.data.body_xmat[1].copy()
    r = R.from_matrix(xmat.reshape(3,3))
    global_theta.append(r.as_euler('xyz')[2])

for i in range(100):
    qpos, qvel = custom_step(sim, -forward_action)
    all_qpos.append(qpos)
    all_qvel.append(qvel)
    all_x_pos.append(sim.data.body_xpos[1].copy())
    all_x_mat.append(sim.data.body_xmat[1].copy())
    xmat = sim.data.body_xmat[1].copy()
    r = R.from_matrix(xmat.reshape(3,3))
    global_theta.append(r.as_euler('xyz')[2])

for i in range(200):
    qpos, qvel = custom_step(sim, -forward_action)
    all_qpos.append(qpos)
    all_qvel.append(qvel)
    all_x_pos.append(sim.data.body_xpos[1].copy())
    all_x_mat.append(sim.data.body_xmat[1].copy())
    xmat = sim.data.body_xmat[1].copy()
    r = R.from_matrix(xmat.reshape(3,3))
    global_theta.append(r.as_euler('xyz')[2])

all_qpos = np.array(all_qpos)
all_qvel = np.array(all_qvel)
all_x_pos = np.array(all_x_pos)
all_x_mat = np.array(all_x_mat)
global_theta = np.array(global_theta)

plt.figure("xpos")
plt.scatter(all_x_pos[:, 0], all_x_pos[:, 1])

plt.figure("qpos")
plt.scatter(all_qpos[:, 0], all_qpos[:, 1])

plt.figure("qvel")
plt.plot(all_qvel[:, 0], label='xvel')
plt.plot(all_qvel[:, 1], label='yvel')
plt.plot(all_qvel[:, 2], label='theta_vel')
plt.legend()
plt.show()
