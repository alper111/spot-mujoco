import json
import os
from PIL import Image

import mujoco
import numpy as np

from env import BaseEnv


def get_xml(name, objfile):
    return f"""
        <mujoco model="{name}">
        <default>
            <default class="visual">
            <geom group="2" type="mesh" contype="0" conaffinity="0"/>
            </default>
            <default class="collision">
            <geom group="3" type="mesh"/>
            </default>
        </default>
        <asset>
            <mesh file="{objfile}"/>
        </asset>
        <worldbody>
            <body name="{name}">
            <geom mesh="{name}" class="visual"/>
            <geom mesh="{name}" class="collision"/>
            </body>
        </worldbody>
        </mujoco>
"""


if __name__ == "__main__":
    N_EXP = 100
    N_BATCH = 100
    N_STEP = 500

    env = BaseEnv(render_mode="offscreen", real_time=False)
    results = np.zeros((N_EXP, 4))
    dimensions = ["footprint_area", "mass", "platform_depth",
                  "platform_height", "platform_width", "top_surface_flatness"]
    print("Dimension; Iter; SliderValue; DistanceToGoal; Success", file=open("spot_out.csv", "w"))

    for dim in dimensions:
        for i in range(0, 10):
            # generate the platform
            folder = f"data/reach/{dim}/iter-{i}"
            objname = f"cube_master_deformed_{i}.obj"
            objpath = os.path.join(folder, objname)
            properties = json.load(open(os.path.join(folder, f"properties_value_{i}.json"), "r"))

            obj = get_xml(f"cube_master_deformed_{i}", objpath)
            env.reset(obj)
            # env.reset(f"data/cube_dataset/{dim}/iter-{i}/objs_def/0_{i}/0_{i}.xml")
            for _ in range(1000):
                env._step()

            # measure the platform's height with the dummy
            height = env.data.body("dummy").xpos[2]

            # teleport by the platform
            env.teleport(x=1.52, y=-0.3, z=0.72+height)
            for _ in range(10):
                env._step()

            # lean forward
            env.data.ctrl[1] = 1.
            env.data.ctrl[2] = -1.5
            env.data.ctrl[4] = 1.
            env.data.ctrl[5] = -1.5
            for _ in range(1000):
                env._step()

            pos, quat = env.get_ee_pose()
            target_xpos1 = env.data.site("goal").xpos.copy()
            target_xpos2 = target_xpos1.copy()
            target_xpos1[0] -= 0.15
            target_xpos1[2] += 0.05
            target_quat = np.zeros(4)
            mujoco.mju_euler2Quat(target_quat, [0, -np.pi/4, 0], "zyx")
            print(f"Target pos: {target_xpos2}, target quat: {target_quat}")
            print(f"Before end-effector pos: {pos} quat: {quat}")

            env.move_to_ee_pose(target_xpos1, T=1)
            env.move_to_ee_pose(target_xpos2, orientation=target_quat, T=1)
            # env._set_ee_pose(target_xpos1, threshold=0.0001, max_iters=2000)
            # env._set_ee_pose(target_xpos2, orientation=target_quat, threshold=0.0001, max_iters=2000)
            for _ in range(1000):
                env._step()

            pos, quat = env.get_ee_pose()
            distance = np.linalg.norm(pos - target_xpos2)
            success = distance < 0.05
            print(f"After end-effector pos: {pos} quat: {quat}")
            print(f"{dim}; {i}; {properties['slider_value']}; {distance}; {success}",
                  file=open("spot_out.csv", "a"))

            # only for offscreen rendering
            env.viewer.update_scene(env.data, camera="camera")
            pixels = env.viewer.render()
            img = Image.fromarray(pixels)
            img.save(f"out/{dim}_{i}.png")
