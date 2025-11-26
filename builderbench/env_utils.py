import re
import jax
import mujoco
import mujoco.mjx as mjx

from flax import struct
from etils import epath
from typing import Any, Dict, Optional

from builderbench.constants import _TIME_STEPS, _MJX_PARAMS

@struct.dataclass
class State:
    """Environment state for training and inference."""
    data: mjx.Data                                   # mjx data
    obs: jax.Array                                   # observation for the RL agent
    reward: jax.Array                                # reward for the RL agent
    done: jax.Array                                  # terminal signal for the RL agent
    metrics: Dict[str, jax.Array]                    # metrics to store during training 
    info: Dict[str, Any]                             # misc information

def mjx_make_data(
	model: mujoco.MjModel,
	qpos: Optional[jax.Array] = None,
	qvel: Optional[jax.Array] = None,
	ctrl: Optional[jax.Array] = None,
	act: Optional[jax.Array] = None,
	mocap_pos: Optional[jax.Array] = None,
	mocap_quat: Optional[jax.Array] = None,
	impl: Optional[str] = None,
	nconmax: Optional[int] = None,
	njmax: Optional[int] = None,
	device: Optional[jax.Device] = None,
) -> mjx.Data:
	"""Initialize MJX Data."""
	data = mjx.make_data(
		model, impl=impl, nconmax=nconmax, njmax=njmax, device=device
	)
	if qpos is not None:
		data = data.replace(qpos=qpos)
	if qvel is not None:
		data = data.replace(qvel=qvel)
	if ctrl is not None:
		data = data.replace(ctrl=ctrl)
	if act is not None:
		data = data.replace(act=act)
	if mocap_pos is not None:
		data = data.replace(mocap_pos=mocap_pos.reshape(model.nmocap, -1))
	if mocap_quat is not None:
		data = data.replace(mocap_quat=mocap_quat.reshape(model.nmocap, -1))
	return data

def mjx_step_data(
    model: mjx.Model,
    data: mjx.Data,
    action: jax.Array,
    n_substeps: int = 1,
) -> mjx.Data:
	def single_step(data, _):
		data = data.replace(ctrl=action)
		data = mjx.step(model, data)
		return data, None

	return jax.lax.scan(single_step, data, (), n_substeps)[0]


def make_env(args):
	# Initialize environment
	if re.fullmatch(r"creative-\d+-task\d+", args.env_id):
		num_cubes = int( re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int( re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.creative_block import CreativeCube, default_config
		env_class = CreativeCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 50

	elif re.fullmatch(r"sparse-creative-\d+-task\d+", args.env_id):
		num_cubes = int( re.search(r"creative-(\d+)", args.env_id).group(1))
		task_id = int( re.search(r"task(\d+)", args.env_id).group(1)) - 1

		from builderbench.creative_block import SparseCreativeCube, default_config
		env_class = SparseCreativeCube
		default_config = default_config()
		default_config.num_cubes = num_cubes
		default_config.task_id = task_id
		episode_length = 100 + num_cubes * 100

	else:
		raise ValueError(f"Environment {args.env_id} not supported")

	default_config.episode_length = args.env_episode_length if args.env_episode_length is not None else episode_length
	default_config.sim_dt, default_config.ctrl_dt = _TIME_STEPS[args.env_id]
	default_config.env_early_termination = args.env_early_termination
	default_config.permutation_invariant_reward = args.permutation_invariant_reward
	default_config.nconmax, default_config.njmax = _MJX_PARAMS[args.env_id][0] * args.num_envs, _MJX_PARAMS[args.env_id][1] 
	return env_class, default_config