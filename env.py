import collections
import time
from copy import copy, deepcopy

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from dm_control import mjcf
import mujoco
import mujoco.viewer
import mujoco.minimize
import mujoco.rollout

IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


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

    def reset(self, path=None):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "data"):
            del self.data
        if self.viewer is not None:
            if self._render_mode == "offscreen":
                del self.viewer
            else:
                self.viewer.close()

        scene = self._create_scene(path)
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
            self.viewer = mujoco.Renderer(self.model, 256, 256)

        self.data.ctrl[1] = np.pi/4
        self.data.ctrl[2] = -np.pi/2
        self.data.ctrl[4] = np.pi/4
        self.data.ctrl[5] = -np.pi/2
        self.data.ctrl[7] = 1.03
        self.data.ctrl[8] = -1.
        self.data.ctrl[10] = 1.03
        self.data.ctrl[11] = -1.

        self.data.ctrl[13] = -np.pi
        self.data.ctrl[14] = 7*np.pi/8

        self.data.ctrl[16] = -1.57
        for _ in range(2000):
            self._step()

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

    def _create_scene(self, path=None):
        scene = mjcf.from_path("mujoco_menagerie/boston_dynamics_spot/scene_arm.xml")
        add_camera_to_scene(scene, "camera", position=[0.5, -2, 2.25], target=[1.75, -0.25, 1])
        gripper = scene.find("body", "arm_link_wr1")
        gripper.add("site", name="gripper_site", pos=[0.2, 0., -0.025], size=[0.02, 0.02, 0.02], rgba=[0, 1, 1, 0.4])

        scene.worldbody.add("site", name="goal", pos=[2.025, -0.3, 1.66528183], type="box",
                            quat=[1, 0, 0, 0], size=[0.02, 0.02, 0.02], rgba=[0.8, 0.3, 0.3, 1.0])

        bookshelf = mjcf.from_path("bookshelf/bookshelf_scaled.xml")
        q = R.from_euler("xyz", np.array([np.pi/2, 0, -np.pi/2])).as_quat(scalar_first=True)
        scene.worldbody.add("site", type="sphere", pos=[2.2055, 0, 0], size=[0.01, 0.01, 0.01], quat=q, rgba=[1, 0, 0, 0], name="bookshelf_site")
        scene.find("site", "bookshelf_site").attach(bookshelf)
        scene.find("key", "home").remove()

        cardboard = mjcf.from_path("cardboard/cardboard.xml")
        scene.worldbody.add("site", type="sphere", pos=[2-0.381/2, 0.05, 0.381/2], size=[0.01, 0.01, 0.01], rgba=[1, 0, 0, 0], name="cardboard_site")
        scene.find("site", "cardboard_site").attach(cardboard)

        desk = mjcf.from_path("desk/desk.xml")
        q = R.from_euler("xyz", np.array([np.pi/2, 0, 0])).as_quat(scalar_first=True)
        scene.worldbody.add("site", type="sphere", pos=[2.2055, -0.8095, 0.], size=[0.01, 0.01, 0.01], quat=q, rgba=[1, 0, 0, 0], name="desk_site")
        scene.find("site", "desk_site").attach(desk)

        if path is not None:
            cube = mjcf.from_xml_string(path)
            cube.worldbody.body[0].geom[1].friction = [10.0, 10.0, 10.0]
            q = R.from_euler("xyz", np.array([np.pi/2, 0, -np.pi/2])).as_quat(scalar_first=True)
            site = scene.worldbody.add("site", type="sphere", pos=[1.75, -0.3, 1], size=[0.01, 0.01, 0.01], quat=q, rgba=[1, 0, 0, 0], name="cube_site")
            body = site.attach(cube)
            body.add("joint", type="free", name="cube_freejoint")
            create_object(scene, "box", pos=[1.75, -0.3, 1.65], quat=[1, 0, 0, 0],
                          size=[0.001, 0.001, 0.001], rgba=[0.8, 0.3, 0.3, 1], name="dummy", static=False)

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

    def _set_chest_and_hip_angle(self, qc, qh, n_step=1):
        qc = max(qc, self._max_leg_angle)
        qh = max(qh, self._max_leg_angle)
        q = np.zeros(self._n_joints)
        q = self.data.ctrl[:]
        curr_qc = q[1]
        curr_qh = q[7]
        c_int = np.linspace(curr_qc, qc, n_step+1)[1:]
        h_int = np.linspace(curr_qh, qh, n_step+1)[1:]
        for i in range(len(c_int)):
            q[1] = c_int[i]
            q[2] = -2*c_int[i]
            q[4] = c_int[i]
            q[5] = -2*c_int[i]
            q[7] = h_int[i]
            q[8] = -2*h_int[i]
            q[10] = h_int[i]
            q[11] = -2*h_int[i]
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

    def _set_chest_and_hip_height(self, ch, hh):
        max_height = self._leg_length * np.cos(self._max_leg_angle)
        ch = min(ch, max_height)
        hh = min(hh, max_height)
        qc = np.arccos(ch/self._leg_length)
        qh = np.arccos(hh/self._leg_length)
        self._set_chest_and_hip_angle(qc, qh)

    def _set_front_leg_angle(self, angle):
        self.data.ctrl[0] = -angle
        self.data.ctrl[3] = angle
        for _ in range(5):
            self._step()

    def _open_gripper(self):
        self.data.ctrl[18] = -1.57
        for _ in range(5):
            self._step()

    def _close_gripper(self):
        self.data.ctrl[18] = 0.0
        for _ in range(5):
            self._step()

    def get_ee_pose(self):
        position = self.data.site(self._ee_site).xpos.copy()
        rotation = self.data.site(self._ee_site).xmat.copy()
        orientation = np.zeros(4)
        mujoco.mju_mat2Quat(orientation, rotation)
        return position, orientation

    def move_to_ee_pose(self, position, orientation=None, T=1.0):
        pos, quat = self.get_ee_pose()
        n_step = int(np.round(T / self.model.opt.timestep))
        pos_traj = np.linspace(pos, position, n_step+1)[1:]
        quat_traj = None

        if orientation is not None:
            cq = R.from_quat(quat, scalar_first=True)
            tq = R.from_quat(orientation, scalar_first=True)
            slerp = Slerp([0, T], R.concatenate([cq, tq]))
            quat_traj = slerp(np.linspace(0, T, n_step+1)[1:]).as_quat(scalar_first=True)

        for j, p_j in enumerate(pos_traj):
            q_j = quat_traj[j] if quat_traj is not None else None
            self._set_ee_pose(p_j, orientation=q_j, max_iters=1, threshold=0.001)

    def _set_ee_pose(self, position, rotation=None, orientation=None, max_iters=100, threshold=0.02):
        if rotation is not None and orientation is not None:
            raise Exception("Either rotation or orientation can be set.")

        quat = None
        if rotation is not None:
            quat = R.from_euler("xyz", rotation, degrees=True).as_quat()
        elif orientation is not None:
            quat = orientation

        dof_indices = [self.model.joint(name).qposadr for name in self._joint_names[12:]]
        result = qpos_from_site_pose(self.model, self.data, self._ee_site, position, quat,
                                     self._joint_names[12:])
        qpos = result.qpos
        qdict = {i: qpos[q_idx] for i, q_idx in enumerate(dof_indices)}

        max_error = 100*threshold
        it = 0
        while max_error > threshold:

            it += 1
            self._step()
            max_error = 0
            curr_pos, curr_quat = self.get_ee_pose()
            max_error += np.linalg.norm(np.array(position) - curr_pos)

            # this part is taken from dm_control
            # https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py#L165
            if quat is not None:
                neg_quat = np.zeros(4)
                mujoco.mju_negQuat(neg_quat, curr_quat)
                error_quat = np.zeros(4)
                mujoco.mju_mulQuat(error_quat, quat, neg_quat)
                error_vel = np.zeros(3)
                mujoco.mju_quat2Vel(error_vel, error_quat, 1)
                max_error += np.linalg.norm(error_vel)
            for idx in qdict:
                self.data.ctrl[idx+12] = qdict[idx]
            if it >= max_iters:
                break

    def _set_ee_pose_fullbody(self, target_pos=None, target_quat=None, n_via=10, T=3.0):
        if target_pos is None and target_quat is None:
            raise Exception("Both position and orientation are None.")
        assert n_via >= 1, "n_via should be greater or equal than 1"

        pos, quat = self.get_ee_pose()
        delta_time = T / n_via
        pos_traj = None
        quat_traj = None

        if target_pos is not None:
            pos_traj = np.linspace(pos, target_pos, n_via+1)[1:]

        if target_quat is not None:
            cq = R.from_quat(quat, scalar_first=True)
            tq = R.from_quat(target_quat, scalar_first=True)
            slerp = Slerp([0, T], R.concatenate([cq, tq]))
            quat_traj = slerp(np.linspace(0, T, n_via+1)[1:]).as_quat(scalar_first=True)

        lower = self.model.actuator_ctrlrange[:, 0].reshape(-1, 1)
        upper = self.model.actuator_ctrlrange[:, 1].reshape(-1, 1)

        for j in range(n_via):
            x0 = self.data.ctrl.copy().reshape(-1, 1)

            def reach_target(x):
                return ik_residual(x, self.model, self.data, "gripper_site",
                                   target_pos=pos_traj[j] if pos_traj is not None else None,
                                   target_quat=quat_traj[j] if quat_traj is not None else None,
                                   T=delta_time, rot_weight=1.0)

            x, _ = mujoco.minimize.least_squares(x0, reach_target, [lower, upper], xtol=1e-3, gtol=1e-3, verbose=False)
            x0 = self.data.ctrl.copy()
            xT = x[:, 0]
            delta_nstep = int(np.round(delta_time / self.model.opt.timestep))
            ctrl = np.linspace(x0, xT, delta_nstep+1)[1:]
            for ctrl_i in ctrl:
                self.data.ctrl[:] = ctrl_i
                self._step()

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


def build_sinusoidal_traj(params, env, n_step, n_repeat):
    """
    params: [gait_period, amplitude, leg_hy, leg_kn]
    returns control sequence shape (1, n_step*n_repeat, n_joints)
    """
    gait_period, amplitude, leg_hy, leg_kn = params
    n_joints = env.action_space.shape[0]
    dt = env.model.opt.timestep
    # initialize ctrl buffer
    ctrl = np.zeros((1, n_step * n_repeat, n_joints), dtype=np.float64)

    for t in range(n_step):
        phase = 2 * np.pi * ((t * dt) % gait_period) / gait_period
        sinp = np.sin(phase)

        # one step of controls
        step_ctrl = np.zeros((n_joints,), dtype=np.float64)
        # hip-pitch trot: FL/HR in phase, FR/HL out of phase
        step_ctrl[1] = amplitude * sinp   # fl_hy
        step_ctrl[4] = -amplitude * sinp   # fr_hy
        step_ctrl[7] = -amplitude * sinp   # hl_hy
        step_ctrl[10] = amplitude * sinp   # hr_hy

        # fix knee angles for support
        for _, idx_kn in [(1, 2), (4, 5), (7, 8), (10, 11)]:
            step_ctrl[idx_kn] = leg_kn

        # optionally set roll joints to some offset (e.g. leg_hy)
        for idx_hy, _ in [(1, 2), (4, 5), (7, 8), (10, 11)]:
            step_ctrl[idx_hy] = leg_hy

        # repeat this step n_repeat times
        for r in range(n_repeat):
            ctrl[0, t * n_repeat + r, :] = step_ctrl

    return ctrl


def body_res(params, env, x=None, y=None, z=None, n_step=100, initial_state=None):
    ctrl_b = np.array(params).reshape(4, 1, -1)
    ctrl_b = ctrl_b.repeat(n_step, axis=1)
    ctrl_b = np.transpose(ctrl_b, (2, 1, 0))
    n_batch = ctrl_b.shape[0]
    ctrl = np.zeros((n_batch, n_step, env.model.nu), dtype=np.float64)
    ctrl[:, :] = env.data.ctrl.copy()
    ctrl[:, :, 1] = ctrl_b[:, :, 0]
    ctrl[:, :, 2] = ctrl_b[:, :, 1]
    ctrl[:, :, 4] = ctrl_b[:, :, 0]
    ctrl[:, :, 5] = ctrl_b[:, :, 1]
    ctrl[:, :, 7] = ctrl_b[:, :, 2]
    ctrl[:, :, 8] = ctrl_b[:, :, 3]
    ctrl[:, :, 10] = ctrl_b[:, :, 2]
    ctrl[:, :, 11] = ctrl_b[:, :, 3]

    # prepare state
    scratch = [copy(env.data) for _ in range(n_batch)]
    if initial_state is None:
        initial_state = env.get_mjstate(nbatch=n_batch)

    # perform rollout
    state, _ = mujoco.rollout.rollout(model=env.model,
                                      data=scratch,
                                      initial_state=initial_state,
                                      control=ctrl)

    # compute error against target
    errors = []
    for i, s in enumerate(state):
        mujoco.mj_setState(env.model, scratch[i], s[-1],
                           mujoco.mjtState.mjSTATE_FULLPHYSICS)
        pos = scratch[i].body("body").xpos
        err = 0
        if x is not None:
            err += (pos[0] - x) ** 2
        if y is not None:
            err += (pos[1] - y) ** 2
        if z is not None:
            err += (pos[2] - z) ** 2
        errors.append(err)
    return np.array(errors).reshape(1, -1)


def ik_residual(u,
                model,
                data,
                site_name,
                target_pos=None,
                target_quat=None,
                T=1.0,
                rot_weight=1.0):
    scratch = deepcopy(data)
    dtype = scratch.qpos.dtype
    if target_pos is not None and target_quat is not None:
        err = np.empty(6, dtype=dtype)
        err_pos, err_rot = err[:3], err[3:]
    else:
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            err_pos, err_rot = err, None
        elif target_quat is not None:
            err_pos, err_rot = None, err
        else:
            raise ValueError("At least one of `target_pos` or `target_quat` must be specified.")

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    # get the state
    full_physics = mujoco.mjtState.mjSTATE_FULLPHYSICS
    mjstate = np.zeros((mujoco.mj_stateSize(model, full_physics),))
    mujoco.mj_getState(model, scratch, mjstate, full_physics)

    # interpolate the control input
    ctrlT = u.T
    ctrl0 = scratch.ctrl.copy().reshape(1, -1)
    ctrl0 = np.tile(ctrl0, (ctrlT.shape[0], 1))

    # Interpolate and stack the control sequences
    nstep = int(np.round(T / model.opt.timestep))
    ctrl = np.stack(np.linspace(ctrl0, ctrlT, nstep), axis=1)

    # perform rollout
    state, _ = mujoco.rollout.rollout(model=model,
                                      data=scratch,
                                      initial_state=mjstate,
                                      control=ctrl)

    errors = []
    for i, s in enumerate(state):
        # compute the error
        mujoco.mj_setState(model, scratch, s[-1], full_physics)
        mujoco.mj_forward(model, scratch)
        err_norm = 0.0
        if target_pos is not None:
            # translational error.
            err_pos[:] = target_pos - scratch.site(site_name).xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # rotational error.
            mujoco.mju_mat2Quat(site_xquat, scratch.site(site_name).xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight
        errors.append(err_norm)
    errors = np.array(errors).reshape(1, -1)
    return errors


# modified from https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
def qpos_from_site_pose(model,
                        data,
                        site_name,
                        target_pos=None,
                        target_quat=None,
                        joint_names=None,
                        tol=1e-14,
                        rot_weight=1.0,
                        regularization_threshold=0.1,
                        regularization_strength=3e-2,
                        max_update_norm=2.0,
                        progress_thresh=20.0,
                        max_steps=100,
                        inplace=False):
    dtype = data.qpos.dtype

    if target_pos is not None and target_quat is not None:
        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]
    else:
        jac = np.empty((3, model.nv), dtype=dtype)
        err = np.empty(3, dtype=dtype)
        if target_pos is not None:
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("At least one of `target_pos` or `target_quat` must be specified.")

    update_nv = np.zeros(model.nv, dtype=dtype)

    if target_quat is not None:
        site_xquat = np.empty(4, dtype=dtype)
        neg_site_xquat = np.empty(4, dtype=dtype)
        err_rot_quat = np.empty(4, dtype=dtype)

    if not inplace:
        data = deepcopy(data)

    mujoco.mj_fwdPosition(model, data)
    site_id = model.site(site_name).id

    # todo: check they are indeed updated in place

    if joint_names is None:
        dof_indices = slice(None)
    elif isinstance(joint_names, (list, np.ndarray, tuple)):
        if isinstance(joint_names, tuple):
            joint_names = list(joint_names)
        dof_indices = [model.joint(name).dofadr[0] for name in joint_names]
    else:
        raise ValueError(f"`joint_names` must be either None, a list, a tuple, or a numpy array; "
                         f"got {type(joint_names)}.")

    success = False
    qpos = data.qpos.copy()
    for steps in range(max_steps):
        err_norm = 0.0

        if target_pos is not None:
            # translational error.
            err_pos[:] = target_pos - data.site(site_name).xpos
            err_norm += np.linalg.norm(err_pos)
        if target_quat is not None:
            # rotational error.
            mujoco.mju_mat2Quat(site_xquat, data.site(site_name).xmat)
            mujoco.mju_negQuat(neg_site_xquat, site_xquat)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_xquat)
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1)
            err_norm += np.linalg.norm(err_rot) * rot_weight

        if err_norm < tol:
            success = True
            break
        else:
            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
            jac_joints = jac[:, dof_indices]
            reg_strength = regularization_strength if err_norm > regularization_threshold else 0.0
            update_joints = nullspace_method(jac_joints, err, regularization_strength=reg_strength)
            update_norm = np.linalg.norm(update_joints)

        progress_criterion = err_norm / update_norm
        if progress_criterion > progress_thresh:
            break

        if update_norm > max_update_norm:
            update_joints *= max_update_norm / update_norm

        update_nv[dof_indices] = update_joints
        mujoco.mj_integratePos(model, data.qpos, update_nv, 1)
        mujoco.mj_fwdPosition(model, data)

        if not inplace:
            qpos = data.qpos.copy()
        else:
            qpos = data.qpos

    return IKResult(qpos, err_norm, steps, success)


# modified from https://github.com/deepmind/dm_control/blob/main/dm_control/utils/inverse_kinematics.py
def nullspace_method(jac_joints, delta, regularization_strength=0.0):
    hess_approx = jac_joints.T.dot(jac_joints)
    joint_delta = jac_joints.T.dot(delta)
    if regularization_strength > 0:
        # L2 regularization
        hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
        return np.linalg.solve(hess_approx, joint_delta)
    else:
        return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]
