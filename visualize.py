from etils import epath

import mujoco
import numpy as np
import subprocess
import sys
import argparse

from builderbench.constants import _CUSTOM_COLORS, _TIME_STEPS, _MJX_PARAMS
from builderbench.create_task_data import cube_4_tasks, cube_5_tasks

def _add_objects(spec, positions):
    object_names = []
    num_cubes = len(positions)
    for i in range(num_cubes):

        # stacked placement
        x = positions[i][0]
        y = positions[i][1]
        z = positions[i][2] + 0.02

        yaw = np.random.uniform(-1, 1) * np.pi * 0
        quat = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

        body = spec.worldbody.add_body(
            name=f"block_{i}",
            pos=[x, y, z],
            quat=quat,
        )
        body.add_joint(
            name=f"block_joint_{i}",
            type=mujoco.mjtJoint.mjJNT_FREE,
            damping=0.1, 
            armature=0.001,
        )
        body.add_geom(
            name=f"block_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            contype=3,   
            conaffinity=1,
            solref=[0.01, 1],
            size=[0.02, 0.02, 0.02],
            rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)],
            density=1240,
        )

        # adding actuators for each cube
        actuator_x = spec.add_actuator(
            name=f"x_block_{i}",
            target=f"block_joint_{i}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gear=[1, 0, 0, 0, 0, 0],
            ctrllimited=True,
            ctrlrange=(-0.1, 0.1),
        )
        actuator_x.set_to_motor()
        
        actuator_y = spec.add_actuator(
            name=f"y_block_{i}",
            target=f"block_joint_{i}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gear=[0, 1, 0, 0, 0, 0],
            ctrllimited=True,
            ctrlrange=(-0.1, 0.1),
        )
        actuator_y.set_to_motor()

        actuator_z = spec.add_actuator(
            name=f"z_block_{i}",
            target=f"block_joint_{i}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gear=[0, 0, 1, 0, 0, 0],
            ctrllimited=True,
            ctrlrange=(0.0, 1.0),
        )
        actuator_z.set_to_motor()

        actuator_yaw = spec.add_actuator(
            name=f"yaw_block_{i}",
            target=f"block_joint_{i}",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            gear=[0, 0, 0, 1, 0, 0],
            ctrllimited=True,
            ctrlrange=(-0.1, 0.1),
        )
        actuator_yaw.set_to_motor()

        # adding target position for cube
        body = spec.worldbody.add_body(
            name=f"target_mocap_{i}",
            mocap=True,
            pos=[x, y, z],
            quat=quat,   
        )
        body.add_geom(
            name=f"target_mocap_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.01, 0.01, 0.01],
            rgba=_CUSTOM_COLORS[i % len(_CUSTOM_COLORS)] - np.array([0.0, 0.0, 0.0, 0.8]),
            contype=0,
            conaffinity=0,
        )

        object_names.append(f"block_{i}")

    return spec, object_names

def create_scene_xml(task_id=1, num_cubes=5, xml_path="builderbench/xmls/creative_scene.xml"):
    """Create scene XML file with specified parameters."""
    env_id = f"creative-{num_cubes}-task{task_id}"
    if num_cubes == 4:
        tasks = cube_4_tasks
    elif num_cubes == 5:
        tasks = cube_5_tasks
    else:
        raise ValueError("Only 4 or 5 cubes are supported in this script.")

    spec = mujoco.MjSpec.from_file(str(xml_path))
    spec.stat.center = np.array([0.4, 0.0 , 0.4])
    spec.stat.extent = 1.2
    spec.visual.global_.elevation = -30.0
    spec.visual.global_.azimuth = 180
    spec, _ = _add_objects(spec, tasks["goals"][task_id-1])
    spec.option.timestep = 0.005

    output_path = f'tmp/{num_cubes}-{task_id}.xml'
    with open(output_path, 'w') as f:
        f.write(spec.to_xml())
    
    return output_path, _MJX_PARAMS[env_id][0], _MJX_PARAMS[env_id][1]

def launch_mjwarp_viewer(xml_path, engine="warp", **kwargs):
    """Launch MuJoCo Warp viewer with specified arguments."""
    cmd = ["python", "-m", "mujoco_warp.viewer", xml_path]
    
    # Add engine flag
    cmd.extend(["--engine", engine])
    
    # Add other optional arguments
    if "nconmax" in kwargs and kwargs["nconmax"] is not None:
        cmd.extend(["--nconmax", str(kwargs["nconmax"])])
    
    if "njmax" in kwargs and kwargs["njmax"] is not None:
        cmd.extend(["--njmax", str(kwargs["njmax"])])
    
    if "device" in kwargs and kwargs["device"] is not None:
        cmd.extend(["--device", kwargs["device"]])
    
    print(f"Launching viewer with command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Create scene and launch MuJoCo Warp viewer")
    parser.add_argument("--task_id", type=int, default=1, help="Task ID (default: 1)")
    parser.add_argument("--num_cubes", type=int, default=5, choices=[4, 5], help="Number of cubes (default: 5)")
    parser.add_argument("--device", type=str, default='cuda', help="Override default Warp device")
    
    args = parser.parse_args()
    
    # Create the scene XML
    xml_path, nconmax, njmax = create_scene_xml(args.task_id, args.num_cubes)
    print(f"Created scene XML: {xml_path}")
    
    viewer_kwargs = {
        "nconmax": nconmax,
        "njmax": njmax,
        "device": args.device,
    }
    launch_mjwarp_viewer(xml_path, **viewer_kwargs)

if __name__ == "__main__":
    main()