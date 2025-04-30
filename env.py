import time
from copy import copy

import numpy as np
from dm_control import mjcf
import mujoco
import mujoco.viewer
import mujoco.rollout


class BaseEnv:
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
        self._ee_site = "gripper_site"
        self._max_leg_angle = 0.1235335
        self._leg_length = 0.6888

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
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                       show_left_ui=False, show_right_ui=False)
            self.viewer.lock()
            self.viewer.cam.fixedcamid = 0
            self.viewer.cam.type = 2
        elif self._render_mode == "offscreen":
            self.viewer = mujoco.Renderer(self.model, 128, 128)

        self.data.ctrl[1] = np.pi/4
        self.data.ctrl[2] = -np.pi/2
        self.data.ctrl[4] = np.pi/4
        self.data.ctrl[5] = -np.pi/2
        self.data.ctrl[7] = np.pi/4
        self.data.ctrl[8] = -np.pi/2
        self.data.ctrl[10] = np.pi/4
        self.data.ctrl[11] = -np.pi/2

        self.data.ctrl[13] = -np.pi
        self.data.ctrl[14] = np.pi

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
        add_camera_to_scene(scene, "camera", position=[0.5, 3, 0.75], target=[1.75, 0, 1])
        gripper = scene.find("body", "arm_link_fngr")
        gripper.add("site", name="gripper_site", pos=[0.05, 0, -0.03], size=[0.02, 0.02, 0.02], rgba=[1, 0, 0, 0])
        size = np.zeros((3,))
        size[0] = np.random.uniform(0.05, 0.35)
        size[1] = np.random.uniform(0.05, 0.35)
        size[2] = np.random.uniform(0.005, 0.15)
        self._platform_size = size*2
        create_object(scene, "box", pos=[2, 0, size[2]], quat=[1, 0, 0, 0],
                      size=size, rgba=[0.8, 0.3, 0.3, 1], name="platform", static=False)
        scene.find("key", "home").remove()
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

    def _set_hip_angle(self, angle, n_step=1):
        angle = max(angle, self._max_leg_angle)
        q = np.zeros(self._n_joints)
        q = self.data.ctrl[:]
        curr_angle = q[7]
        intpl = np.linspace(curr_angle, angle, n_step+1)[1:]
        for q_i in intpl:
            q[7] = q_i
            q[8] = -2*q_i
            q[10] = q_i
            q[11] = -2*q_i
            for _ in range(100):
                self._step()

    def _set_chest_angle(self, angle, n_step=1):
        angle = max(angle, self._max_leg_angle)
        q = np.zeros(self._n_joints)
        q = self.data.ctrl[:]
        curr_angle = q[1]
        intpl = np.linspace(curr_angle, angle, n_step+1)[1:]
        for q_i in intpl:
            q[1] = q_i
            q[2] = -2*q_i
            q[4] = q_i
            q[5] = -2*q_i
            for _ in range(100):
                self._step()

    def _set_hip_height(self, height):
        max_height = self._leg_length * np.cos(self._max_leg_angle)
        height = min(height, max_height)
        angle = np.arccos(height/self._leg_length)
        self._set_hip_angle(angle)

    def _set_chest_height(self, height):
        max_height = self._leg_length * np.cos(self._max_leg_angle)
        height = min(height, max_height)
        angle = np.arccos(height/self._leg_length)
        self._set_chest_angle(angle)

    def _set_front_leg_angle(self, angle):
        self.data.ctrl[0] = -angle
        self.data.ctrl[3] = angle

    def get_gripper_position(self):
        return self.data.site(self._ee_site).xpos

    def reach(self, n_step=100):

        q = np.zeros(self._n_joints)
        q = self.data.ctrl[:]
        q_13_c = q[13]
        q_14_c = q[14]

        q13_intpl = np.linspace(q_13_c, -1.6, n_step+1)[1:]
        q14_intpl = np.linspace(q_14_c, 0.4, n_step+1)[1:]
        for i in range(n_step):
            q[1] = q[1] + 0.2/n_step
            q[4] = q[4] + 0.2/n_step
            q[7] = q[7] + 0.2/n_step
            q[10] = q[10] + 0.2/n_step
            q[13] = q13_intpl[i]
            q[14] = q14_intpl[i]
            self._set_joint_position(q)

    def teleport(self, x=None, y=None, z=None, qw=None, qx=None, qy=None, qz=None):
        if x is not None:
            self.data.qpos[0] = x
        if y is not None:
            self.data.qpos[1] = y
        if z is not None:
            self.data.qpos[2] = z
        if qw is not None:
            self.data.qpos[3] = qw
        if qx is not None:
            self.data.qpos[4] = qx
        if qy is not None:
            self.data.qpos[5] = qy
        if qz is not None:
            self.data.qpos[6] = qz
        self._step()

    def get_mjstate(self, nbatch=1):
        full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
        state = np.zeros((mujoco.mj_stateSize(self.model, full_physics),))
        mujoco.mj_getState(self.model, self.data, state, full_physics)
        return np.tile(state, (nbatch, 1))


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
    N_EXP = 100
    N_BATCH = 100
    N_STEP = 500

    env = BaseEnv(render_mode="gui", real_time=False)
    results = np.zeros((N_EXP, 4))

    for i in range(N_EXP):
        env.reset()
        results[i, :3] = env._platform_size
        print(results[i])

        env.teleport(x=1.75-env._platform_size[0]/2, y=0, z=0.72+env._platform_size[2])
        env._set_front_leg_angle(0.25)
        env._set_hip_height(0.683)
        env._set_chest_height(0.683-env._platform_size[2]-0.01)
        for _ in range(1000):
            env._step()
        env.reach()
        for _ in range(1000):
            env._step()
        print(env.get_gripper_position())
