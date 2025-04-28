import time

import numpy as np
from dm_control import mjcf
import mujoco
import mujoco.viewer
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed


class BaseEnv(gym.Env):
    def __init__(self, render_mode="gui", real_time=False, max_timesteps=10000) -> None:
        self._render_mode = render_mode
        self._real_time = real_time
        self._max_timesteps = max_timesteps

        self.viewer = None
        self._joint_names = [
            "fl_hx",
            "fl_hy",
            "fl_kn",
            "fr_hx",
            "fr_hy",
            "fr_kn",
            "hl_hx",
            "hl_hy",
            "hl_kn",
            "hr_hx",
            "hr_hy",
            "hr_kn",
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
            "arm_f1x",
        ]
        self._n_joints = len(self._joint_names)
        self.action_space = gym.spaces.Box(
            low=-3.14,
            high=3.14,
            shape=(self._n_joints,),
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Box(
            low=-100,
            high=100,
            shape=(self._n_joints*2+7,),
            dtype=np.float64,
        )
        self.reset()

    @property
    def observation(self):
        qpos = self._get_joint_position()
        qvel = self._get_joint_velocity()
        body_xpos = self.data.body("body").xpos
        body_xquat = self.data.body("body").xquat
        obs = np.concatenate([qpos, qvel, body_xpos, body_xquat])
        return obs

    @property
    def reward(self):
        body_xpos = self.data.body("body").xpos
        body_xquat = self.data.body("body").xquat
        goal_xpos = [2, 0, 0.65222]
        goal_xquat = [1, 0, 0, 0]
        xpos_reward = -np.linalg.norm(body_xpos - goal_xpos)
        xquat_reward = -np.linalg.norm(body_xquat - goal_xquat)
        reward = xpos_reward + xquat_reward
        return reward

    @property
    def terminated(self):
        return bool(self.reward > -0.1)

    @property
    def truncated(self):
        return self._t >= self._max_timesteps

    @property
    def info(self):
        return {}

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "data"):
            del self.data
        if self.viewer is not None:
            if self._render_mode == "offscreen":
                del self.viewer
            else:
                self.viewer.close()

        scene = self._create_scene()
        xml_string = scene.to_xml_string()
        assets = scene.get_assets()
        self.model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
        self.data = mujoco.MjData(self.model)
        if self._render_mode == "gui":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        elif self._render_mode == "offscreen":
            self.viewer = mujoco.Renderer(self.model, 128, 128)

        self._t = 0
        return self.observation, self.info

    def step(self, action):
        self._t += 1
        self._set_joint_position(action)
        observation = self.observation
        reward = self.reward
        terminated = self.terminated
        truncated = self.truncated
        info = self.info
        return observation, reward, terminated, truncated, info

    def _create_scene(self):
        scene = mjcf.from_path("mujoco_menagerie/boston_dynamics_spot/scene_arm.xml")
        create_visual(scene, "sphere", pos=[2, 0, 0.65222], quat=[1, 0, 0, 0],
                      size=[0.05], rgba=[0.2, 1.0, 0.2, 1], name="goal")
        return scene

    def _step(self):
        step_time = time.time()
        mujoco.mj_step(self.model, self.data)
        if self._render_mode == "gui":
            self.viewer.sync()
        if self._real_time:
            time_until_next_step = self.model.opt.timestep - (time.time() - step_time)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def _get_joint_position(self):
        position = np.zeros(self._n_joints)
        for i, name in enumerate(self._joint_names):
            position[i] = self.data.joint(name).qpos[0]
        return position

    def _get_joint_velocity(self):
        velocity = np.zeros(self._n_joints)
        for i, name in enumerate(self._joint_names):
            velocity[i] = self.data.joint(name).qvel[0]
        return velocity

    def _set_joint_position(self, joint_values, max_iters=1, threshold=0.05):
        self.data.ctrl[:] = joint_values
        self._step()
        current_position = self._get_joint_position()
        max_error = max(abs(current_position - joint_values))
        it = 0
        while max_error > threshold:
            it += 1
            self._step()
            max_error = 0
            current_position = self._get_joint_position()
            error = max(abs(current_position - joint_values))
            if error > max_error:
                max_error = error

            if it > max_iters:
                break


def create_object(root, obj_type, pos, quat, size, rgba, friction=[0.5, 0.005, 0.0001], density=1000,
                  name=None, static=False):
    body = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    if not static:
        body.add("joint", type="free")
    body.add("geom", type=obj_type, size=size, rgba=rgba, friction=friction, name=name, density=density)
    return root


def create_box(root, pos, quat, size, width, rgba, friction=[0.5, 0.005, 0.0001],
               lid_type="slide", name=None, static=False):
    base = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    if not static:
        base.add("joint", type="free")
    base.add("geom", type="box", size=[size[0] + width, size[1] + width, width/2],
             rgba=rgba, friction=friction, pos=[0, 0, -(size[2]+width/2)], mass=0.025)
    base.add("geom", type="box", size=[width/2, size[1] + width, size[2]],
             rgba=rgba, friction=friction, pos=[size[0]+width/2, 0, 0], mass=0.025)
    base.add("geom", type="box", size=[width/2, size[1] + width, size[2]],
             rgba=rgba, friction=friction, pos=[-(size[0]+width/2), 0, 0], mass=0.025)
    base.add("geom", type="box", size=[size[0], width/2, size[2]],
             rgba=rgba, friction=friction, pos=[0, size[1]+width/2, 0], mass=0.025)
    base.add("geom", type="box", size=[size[0], width/2, size[2]],
             rgba=rgba, friction=friction, pos=[0, -(size[1]+width/2), 0], mass=0.025)
    lid = base.add("body", pos=[0, 0, size[2] + width/2])
    if lid_type == "slide":
        lid.add("joint", type="slide", axis=[0, 1, 0], range=[-2*size[1], 2*size[1]], damping=0.1)
        lid.add("geom", type="box", size=[size[0]+width, size[1]+width, width/2],
                rgba=[rgba[0]*0.8, rgba[1]*0.8, rgba[2]*0.8, 1], friction=friction, pos=[0, 0, 0],
                mass=0.025)
        lid.add("geom", type="cylinder", size=[0.005, 0.01], rgba=[0.8, 0.8, 0.8, 1.0],
                pos=[0, -0.02, 0.01+width/2], mass=0.025)
        lid.add("geom", type="cylinder", size=[0.005, 0.01], rgba=[0.8, 0.8, 0.8, 1.0],
                pos=[0, 0.02, 0.01+width/2], mass=0.025)
        lid.add("geom", type="capsule", size=[0.005, 0.02], rgba=[0.8, 0.8, 0.8, 1.0],
                friction=[1., 0.005, 0.0001],
                pos=[0, 0, 0.02+width/2], quat=[0.7071068, 0.7071068, 0, 0], mass=0.025)
    if lid_type == "hinge":
        lid.add("joint", type="hinge", axis=[1, 0, 0], range=[0, np.pi], pos=[0, -(size[1]+width/2), 0])
        lid.add("geom", type="box", size=[size[0]+width, size[1]+width, width/2],
                rgba=[rgba[0]*0.8, rgba[1]*0.8, rgba[2]*0.8, 1], friction=friction, pos=[0, 0, 0],
                mass=0.025)
        lid.add("geom", type="box", size=[0.0075, 0.0075, 0.0075], rgba=[0.8, 0.8, 0.8, 1.0],
                friction=[1., 0.005, 0.0001],
                pos=[0, size[1]-0.0075, 0.0075+width/2], mass=0.025)

    return root


def create_visual(root, obj_type, pos, quat, size, rgba, name=None):
    body = root.worldbody.add("body", pos=pos, quat=quat, name=name)
    body.add("site", type=obj_type, size=size, rgba=rgba, name=name)
    return root


def create_base(root, position, height, rgba=[0.5, 0.5, 0.5, 1.0]):
    body = root.worldbody.add("body", pos=position, name="groundbase")
    body.add("geom", type="cylinder", size=[0.1, height], rgba=rgba, name="groundbase")
    body.add("site", pos=[0, 0, height], name="attachment_site")
    return root


def add_camera_to_scene(root, name, position, target):
    target_dummy = root.worldbody.add("body", pos=target)
    root.worldbody.add("camera", name=name, mode="targetbody", pos=position, target=target_dummy)
    return root


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1[0], point1[1], point1[2],
                             point2[0], point2[1], point2[2])


if __name__ == "__main__":
    env = BaseEnv("offscreen")
    check_env(env)
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000000)
    model.save("spot_arm")

# set_random_seed(4)
# env = MNISTHyperGrid(dimensions=(6, 6), eps=0.05, max_episode_steps=20)
# eval_env = MNISTHyperGrid(dimensions=(6, 6), eps=0.05, max_episode_steps=20)
# check_env(env)

# logger = configure("out/logs/", ["csv", "stdout"])
# model = DQN("MlpPolicy", env, verbose=1)
# model.set_logger(logger)
# eval_callback = EvalCallback(eval_env, eval_freq=100, n_eval_episodes=10)
# model.learn(total_timesteps=40_000, callback=eval_callback)

# env.close()
# eval_env.close()
