import jax
import jax.numpy as jnp
import optax
import numpy as np
import mujoco
import mujoco.mjx as mjx
from mujoco.mjx._src import math as mjx_math

from etils import epath
from ml_collections import config_dict
from typing import Optional

from builderbench.env_utils import State, mjx_make_data, mjx_step_data
from builderbench.constants import _CUSTOM_COLORS

def default_config() -> config_dict.ConfigDict:
    config = config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.005,
        nconmax=16 * 1024, 
        njmax=32,
        num_cubes=1,
        action_scale = config_dict.create(
            xy_scale = 0.1,
            yaw_scale = 0.1,
            select_scale = np.pi,
            ),
        delta_control=False,
        episode_length=150,
        task_id=0,
        reward_sensitivity=5.0,
        success_threshold=0.02,
        easy_success_threshold=0.05,
        env_early_termination=True,
        permutation_invariant_reward=True,
        impl='warp'
    )
    return config

class PlanarCube():
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
    ):
        
        xml_path = (
            epath.Path(__file__).resolve().parent
            / "xmls"
            / "creative_scene.xml"
        )
                        
        self._init(xml_path=xml_path, config=config)

        self._config = config
        self._ctrl_dt = config.ctrl_dt
        self._sim_dt = config.sim_dt
        self._episode_length = config.episode_length

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def spec(self) -> mujoco.MjSpec:
        return self._spec

    @property
    def dt(self) -> float:
        """Control timestep for the environment."""
        return self._ctrl_dt

    @property
    def sim_dt(self) -> float:
        """Simulation timestep for the environment."""
        return self._sim_dt
    
    @property
    def n_substeps(self) -> int:
        """Number of sim steps per control step."""
        return int(round(self.dt / self.sim_dt))
    
    @property
    def action_size(self):
        """Size of the action space."""
        return 5
    
    @property
    def observation_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        obs = abstract_state.obs
        return obs.shape[-1]
    
    @property
    def goal_size(self):
        abstract_state = jax.eval_shape(self.reset, jax.random.PRNGKey(0))
        goal = abstract_state.info["achieved_goal"]
        return goal.shape[-1]

    @property
    def unwrapped(self):
        return self

    def _init(self, xml_path, config):

        # prepare spec and add objects to the spec
        spec = self._prepare_spec(xml_path, config)
        spec, self._object_names = self._add_objects(spec, config.num_cubes)
        self._spec = spec

        # compile spec and create mujoco model and data
        self._mj_model = self._spec.compile()
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)

        # get dimensions
        self._sensor_dim = self._mj_model.nsensordata
        self._pstate_dim = mujoco.mj_stateSize(self._mj_model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._qpos_dim = self._mj_model.nq
        self._qvel_dim = self._mj_model.nv
        self._ctrl_dim = self._mj_model.nu

        # get mocap ids
        self._mocap_targets = np.array([
            self._mj_model.body( f"target_mocap_{i}" ).mocapid[0]
            for i in range( config.num_cubes )
        ])
        self._mocap_targets_geom = np.array([
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_GEOM, f"target_mocap_{i}")
            for i in range( config.num_cubes )
        ])

        # get object ids
        self._objs_ids = np.array([
            self._mj_model.body(obj_name).id
            for obj_name in self._object_names
        ])

        # get start indices in qpos and qvel
        self._objs_qposadr = np.array([
            self._mj_model.jnt_qposadr[ self._mj_model.body(obj_name).jntadr[0] ]
            for obj_name in self._object_names
        ])
        self._objs_qveladr = np.array([
            self._mj_model.jnt_dofadr[self._mj_model.body(obj_name).jntadr[0]]
            for obj_name in self._object_names
        ])

        # # get object pos and quat indices in qpos
        # self._objs_pos_qpos_idxs = np.concatenate([
        #     obj_adr + np.arange(3)
        #     for obj_adr in self._objs_qposadr
        # ])
        # self._objs_quat_qpos_idxs = np.concatenate([
        #     obj_adr + 3 + np.arange(4)
        #     for obj_adr in self._objs_qposadr
        # ])

        # get start indices in actuators
        self._objs_actuator_adr = np.stack([
            np.array([
                mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
                for actuator_name in [f"x_block_{i}", f"y_block_{i}"]
            ])
            for i in range(config.num_cubes)
        ])
        self._objs_actuator_adr = jnp.array(self._objs_actuator_adr)

        # get constants
        _data = mujoco.MjData(self._mj_model)
        _data.ctrl = np.array([0, 0, 0, 0] * config.num_cubes)

        # get constants
        self._init_q = jnp.array(_data.qpos, dtype=jnp.float32)
        self._init_v = jnp.array(_data.qvel, dtype=jnp.float32) * 0
        self._init_ctrl = jnp.array(_data.ctrl, dtype=jnp.float32)
        
        # set action scale
        self._action_scale =  np.array([config.action_scale.xy_scale]*2 + [config.action_scale.select_scale])

        # set bounds       
        self._workspace_bounds = jnp.array([ [ -0.12, -0.12, 0.03 ], [ 0.12, 0.12, 0.05 ] ])
        self._target_sampling_bounds = jnp.array([ [ -0.1, -0.1, 0.0 ], [ 0.1, 0.1, 0.0 ] ])
        self._ctrl_bounds = jnp.array( self._mj_model.actuator_ctrlrange.T )
        self._ctrl_median = (self._ctrl_bounds[1] + self._ctrl_bounds[0]) / 2
        self._ctrl_halfspan = (self._ctrl_bounds[1] - self._ctrl_bounds[0]) / 2

        # get task data
        task_data = np.load( epath.Path(__file__).resolve().parent / f'tasks/creative-{config.num_cubes}.npz')
        self._starts_data = jnp.array( task_data['starts'][config.task_id] )

    def _add_objects(self, spec, num_cubes):
        object_names = []
        for i in range(num_cubes):

            # stacked placement
            x = 0.1
            y = (0.06 * i)
            z = 0.04

            body = spec.worldbody.add_body(
                name=f"block_{i}",
                pos=[x, y, z], 
            )

            body.add_joint(
                name=f"x_block_joint_{i}",
                type=mujoco.mjtJoint.mjJNT_FREE,
                axis=(1, 0, 0),
                range=(-0.1, 0.1),
            )
            body.add_joint(
                name=f"y_block_joint_{i}",
                type=mujoco.mjtJoint.mjJNT_FREE,
                axis=(0, 1, 0),
                range=(-0.1, 0.1),
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
                target=f"x_block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[1, 0, 0, 0, 0, 0],
                ctrllimited=True,
                ctrlrange=(-0.1, 0.1),
            )
            actuator_x.set_to_motor()

            actuator_y = spec.add_actuator(
                name=f"y_block_{i}",
                target=f"y_block_joint_{i}",
                trntype=mujoco.mjtTrn.mjTRN_JOINT,
                gear=[0, 1, 0, 0, 0, 0],
                ctrllimited=True,
                ctrlrange=(-0.1, 0.1),
            )
            actuator_y.set_to_motor()

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

    def _prepare_spec(self, xml_path, config):
        spec = mujoco.MjSpec.from_file(str(xml_path))

        spec.option.timestep = config.sim_dt
        
        spec.stat.center = np.array([0.4, 0.0 , 0.4])
        spec.stat.extent = 1.2
        spec.visual.global_.elevation = -30.0
        spec.visual.global_.azimuth = 180

        return spec