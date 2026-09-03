import os
import sys

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
if "MUJOCO_GL" not in os.environ and sys.platform.startswith("linux"):
    os.environ["MUJOCO_GL"] = "egl"

import jax
jax.config.update(
    "jax_compilation_cache_dir",
    os.environ.get("JAX_COMPILATION_CACHE_DIR", "/tmp/builderbench_jax_cache"),
)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import time
import json
import tyro
import numpy as np
import functools
import pprint
import wandb
import wandb_osh

import jax
import flax
import optax
import distrax
import flax.linen as nn
import jax.numpy as jnp

from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from pathlib import Path
from dataclasses import dataclass, field
from typing import Literal, NamedTuple
from wandb_osh.hooks import TriggerWandbSyncHook

from utils.wrapper import wrap_env, PDWrapper
from utils.evaluation import Evaluator, get_video
from utils.networks import MLP, save_params
from utils.jax import count_parameters
from builderbench.env_utils import make_env
from utils.buffer import TrajectoryUniformSamplingQueue
from continual_auc import OnlineNormalizedAUC
from continual.dcc_networks import make_dcc_networks
from continual.flat_upstream_networks import make_flat_upstream_networks
from continual.grouped_layout import GroupedPadLayout
from continual.grouped_wrapper import GroupedPadWrapper
from continual.vanilla_networks import make_vanilla_crl_networks
from continual.semantic_layout import SemanticLayout
from continual.semantic_wrapper import SemanticPadWrapper


def save_gif(frames, path, fps=10):
    from PIL import Image
    imgs = [Image.fromarray(np.asarray(f).astype(np.uint8)) for f in frames]
    duration_ms = max(1, int(round(1000 / fps)))
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def render_cube_positions(env, cube_pos_vec, mocap_pos_vec=None):
    """Render a (num_cubes*3,) cube-position vector as a mujoco scene.

    Fills a copy of the env's init qpos with the given cube XYZs (leaving
    quats at init) and zero velocities. If `mocap_pos_vec` is given (same
    shape), it's drawn as the scene's target markers; otherwise mocap is
    pushed off-screen.
    """
    num_cubes = env._config.num_cubes
    qpos = np.array(env._init_q, copy=True)
    qpos[np.asarray(env._objs_pos_qpos_idxs)] = np.asarray(cube_pos_vec).reshape(-1)
    qvel = np.zeros_like(np.array(env._init_v))
    if mocap_pos_vec is None:
        mocap_pos = np.tile(np.array([10.0, 10.0, 10.0]), num_cubes)
    else:
        mocap_pos = np.asarray(mocap_pos_vec).reshape(-1)
    identity_quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), num_cubes)
    return env.render_from_info(qpos, qvel, mocap_pos, identity_quats)


def visualize_crl_batch(env, achieved_goals, future_goals, num_anchors=4, num_negatives=4):
    """Render a grid showing anchor state, positive goal, and negative goals.

    Col 0: anchor cubes at `achieved[i]` with mocap targets placed at `future[i]`
           (positive goal) -- mirrors what the critic sees: (state, goal).
    Col 1: cubes at `future[i]` rendered alone -- the positive goal configuration.
    Col 2..: cubes at `future[j!=i]` rendered alone -- negatives pulled from the
           same minibatch (exactly as InfoNCE uses them).

    Subplot titles include the L2 cube-position distance from the anchor, so it's
    visible when positive/negatives are numerically close (common early in training).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    achieved = np.asarray(achieved_goals)
    future = np.asarray(future_goals)
    B = achieved.shape[0]
    num_anchors = min(num_anchors, B)
    num_negatives = min(num_negatives, B - 1)

    # cache pure-goal renders (no mocap) keyed by row index in `future`
    goal_cache = {}
    def render_goal(j):
        if j not in goal_cache:
            goal_cache[j] = render_cube_positions(env, future[j])
        return goal_cache[j]

    print("[sample viz] anchor -> {positive, neg...} L2 distances (cube XYZ):")
    ncols = 2 + num_negatives
    fig, axes = plt.subplots(num_anchors, ncols, figsize=(2.4 * ncols, 2.4 * num_anchors), squeeze=False)
    for i in range(num_anchors):
        # anchor cubes with positive goal overlaid as mocap targets
        axes[i, 0].imshow(render_cube_positions(env, achieved[i], mocap_pos_vec=future[i]))
        axes[i, 0].set_ylabel(f"sample {i}")
        axes[i, 0].set_xticks([]); axes[i, 0].set_yticks([])

        d_pos = float(np.linalg.norm(achieved[i] - future[i]))
        axes[i, 1].imshow(render_goal(i))
        axes[i, 1].set_title(f"pos  Δ={d_pos:.3f}")
        axes[i, 1].set_xticks([]); axes[i, 1].set_yticks([])

        neg_idxs = [j for j in range(B) if j != i][:num_negatives]
        neg_dists = []
        for k, j in enumerate(neg_idxs):
            d_neg = float(np.linalg.norm(achieved[i] - future[j]))
            neg_dists.append(d_neg)
            axes[i, 2 + k].imshow(render_goal(j))
            axes[i, 2 + k].set_title(f"neg{k} Δ={d_neg:.3f}")
            axes[i, 2 + k].set_xticks([]); axes[i, 2 + k].set_yticks([])

        print(f"  sample {i}: pos={d_pos:.4f}  negs={['%.4f' % d for d in neg_dists]}")

        if i == 0:
            axes[i, 0].set_title("anchor (cubes) + positive (mocap)")
    fig.tight_layout()
    return fig

@dataclass
class Args:
    # experiment
    agent: str = "crl"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]

    # logging and checkpointing
    track: bool = True
    wandb_project_name: str = "builderbench-stablecrl-dcc"
    wandb_entity: str | None = None
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50             # number of evaluation / logging / saving steps
    num_reset_steps: int = 50             # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # eval-time video recording
    record_videos: bool = True            # render one eval episode per eval step
    video_every_n_evals: int = 1          # record every N eval steps (1 = every eval)
    video_fps: int = 10

    # training-sample visualization (anchor / positive / negatives from CRL batch)
    visualize_samples: bool = True
    viz_every_n_evals: int = 1
    viz_num_anchors: int = 4
    viz_num_negatives: int = 4

    # environment
    env_id: str = 'creative-1-task1'
    num_envs: int = 1024
    num_eval_envs: int = 128
    env_early_termination: bool = False # note!
    env_episode_length: int | None = None
    permutation_invariant_reward: bool = True   # invariance to the order of cubes in any structure

    # algorithm
    num_timesteps: int = 200_000_000
    policy_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    encoder_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    rollout_length: int = 64
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    entropy_cost: float = 0.1
    entropy_cost_final: float | None = None  # linearly anneal entropy_cost to this value when set
    entropy_decay_fraction: float = 1.0      # fraction of total gradient steps used for the anneal
    logsumexp_cost: float = 0.1
    rep_size: int = 64
    architecture: Literal["block", "default"] = "block"  # "default" matches the actor/critic MLPs in crl.py
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    diagnostic: bool = False
    repetition_factor: int = 1  # CRTR: >1 repeats each sampled trajectory this many times in the batch (1 = plain CRL)
    num_blocks: int = 8
    hidden_dim: int = 1024
    scale_actor_residual_by_depth: bool = True  # keep deep actor residual stacks on the same scale as the 1-block baseline
    scale_critic_residual_by_depth: bool = True  # keep deep critic encoder residual stacks on the same scale as the 1-block baseline
    use_non_residual_critic_encoders: bool = False  # block architecture only: use the SA/G encoder MLPs from crl.py
    use_pd: bool = True
    pd_duration: int = 5
    mjx_impl: Literal["warp", "jax"] = "warp"

    # continual DCC
    max_cubes: int = 8
    dcc_task_width: int = 256
    dcc_task_depth: int = 4
    dcc_combine_mode: Literal["add", "concat"] = "add"
    dcc_goal_encoder_mode: Literal["shared", "projected"] = "shared"
    dcc_carry_shared: bool = True

    # semantic-wrapper control: ordinary CRL without DCC decomposition
    critic_family: Literal["dcc", "vanilla"] = "dcc"
    observation_layout: Literal["semantic", "grouped"] = "semantic"
    vanilla_network_type: Literal["set", "flat_upstream"] = "set"
    vanilla_width: int = 512
    vanilla_depth: int = 3
    vanilla_carry_critic: bool = True

@flax.struct.dataclass
class CRLTrainingState:
    """Contains training state for the learner"""
    env_steps: np.ndarray
    gradient_steps: np.ndarray
    actor_state: TrainState
    critic_state: TrainState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    extras: jnp.ndarray = ()

class ResidualBlock(nn.Module):
    hidden_dim: int
    norm_type: str = "layer_norm"
    residual_scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)

        residual = x
        for _ in range(4):
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = nn.swish(x)
        return residual + self.residual_scale * x


class SA_encoder(nn.Module):
    rep_size: int
    residual: bool = True
    num_blocks: int = 1
    hidden_dim: int = 1024
    norm_type: str = "layer_norm"
    scale_residual_by_depth: bool = True

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)

        x = jnp.concatenate([s, a], axis=-1)
        if self.residual:
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = nn.swish(x)

            branch_scale = 1.0
            if self.scale_residual_by_depth and self.num_blocks > 1:
                branch_scale = 1.0 / np.sqrt(self.num_blocks)

            for _ in range(self.num_blocks):
                x = ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    norm_type=self.norm_type,
                    residual_scale=branch_scale,
                )(x)
        else:
            for _ in range(4):
                x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
                x = normalize(x)
                x = nn.swish(x)

        x = nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class G_encoder(nn.Module):
    rep_size: int
    residual: bool = True
    num_blocks: int = 1
    hidden_dim: int = 1024
    norm_type: str = "layer_norm"
    scale_residual_by_depth: bool = True

    @nn.compact
    def __call__(self, g: jnp.ndarray):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)

        x = g
        if self.residual:
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = nn.swish(x)

            branch_scale = 1.0
            if self.scale_residual_by_depth and self.num_blocks > 1:
                branch_scale = 1.0 / np.sqrt(self.num_blocks)

            for _ in range(self.num_blocks):
                x = ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    norm_type=self.norm_type,
                    residual_scale=branch_scale,
                )(x)
        else:
            for _ in range(4):
                x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
                x = normalize(x)
                x = nn.swish(x)

        x = nn.Dense(self.rep_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class Actor(nn.Module):
    action_size: int
    residual: bool = True
    num_blocks: int = 1
    hidden_dim: int = 1024
    norm_type: str = "layer_norm"
    scale_residual_by_depth: bool = True

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, s, g_repr):
        lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        normalize = (lambda y: nn.LayerNorm()(y)) if self.norm_type == "layer_norm" else (lambda y: y)

        x = jnp.concatenate([s, g_repr], axis=-1)
        if self.residual:
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = normalize(x)
            x = nn.swish(x)

            # Keep deeper actors close to the 1-block initialization/gradient scale so
            # the policy heads do not saturate as num_blocks grows.
            branch_scale = 1.0
            if self.scale_residual_by_depth and self.num_blocks > 1:
                branch_scale = 1.0 / np.sqrt(self.num_blocks)

            for _ in range(self.num_blocks):
                x = ResidualBlock(
                    hidden_dim=self.hidden_dim,
                    norm_type=self.norm_type,
                    residual_scale=branch_scale,
                )(x)
        else:
            for _ in range(4):
                x = nn.Dense(1024, kernel_init=lecun_uniform, bias_init=bias_init)(x)
                x = normalize(x)
                x = nn.swish(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


def make_inference_fn(contrastive_networks):
    """Creates params and inference function for the CRL agent."""
    def make_policy(params, deterministic: bool = False):

        def policy(observations, goals, key_sample):

            goals = contrastive_networks.apply_goal(params['critic'], goals)
            means, log_stds = contrastive_networks.actor.apply(
                params['actor'], observations, goals
            )

            if deterministic:
                return nn.tanh( means ), {}

            stds = jnp.exp(log_stds)
            raw_actions = means + stds * jax.random.normal(key_sample, shape=means.shape, dtype=means.dtype)
            postprocessed_actions = nn.tanh(raw_actions)

            log_prob = jax.scipy.stats.norm.logpdf(raw_actions, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(postprocessed_actions)) + 1e-6)
            log_prob = log_prob.sum(-1)

            return postprocessed_actions, {
                'log_prob': log_prob,
                'raw_action': raw_actions,
            }

        return policy

    return make_policy

def main(args: Args, carry: dict | None = None, task_index: int = 0):

    if args.repetition_factor < 1:
        raise ValueError("repetition_factor must be positive")
    if args.num_envs < args.repetition_factor:
        raise ValueError(
            f"num_envs={args.num_envs} must be at least "
            f"repetition_factor={args.repetition_factor}; otherwise CRTR "
            "would create an empty training batch"
        )

    args.num_training_step = args.num_timesteps // ( args.num_envs * args.rollout_length )
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)
    num_sgd_steps_per_training_step = (args.rollout_length * args.num_envs) // args.num_envs
    total_gradient_steps = args.num_training_step * num_sgd_steps_per_training_step

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = {num_sgd_steps_per_training_step}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Data to update ratio = {  ( args.num_envs * args.rollout_length ) / num_sgd_steps_per_training_step}")

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"

    # Initialize wandb if tracking is enabled
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity or None,
            mode=args.wandb_mode,
            dir=args.wandb_dir,
            group=args.wandb_group,
            name=args.exp_name,
            config=vars(args),
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, key_buffer, key_env, key_eval, key_actor, key_sa, key_g = jax.random.split(key, 7)

    # Initialize environment
    env_class, default_config = make_env(args)
    default_config.impl = args.mjx_impl
    if args.critic_family == "dcc" and args.observation_layout != "semantic":
        raise ValueError("DCC requires observation_layout='semantic'")
    if (
        args.critic_family == "vanilla"
        and args.vanilla_network_type == "set"
        and args.observation_layout != "semantic"
    ):
        raise ValueError(
            "vanilla_network_type='set' requires observation_layout='semantic'"
        )
    layout = (
        SemanticLayout(max_cubes=args.max_cubes)
        if args.observation_layout == "semantic"
        else GroupedPadLayout(max_cubes=args.max_cubes)
    )
    wrapper_type = (
        SemanticPadWrapper
        if args.observation_layout == "semantic"
        else GroupedPadWrapper
    )
    semantic_env = wrapper_type(
        env_class(config=default_config),
        num_cubes=default_config.num_cubes,
        max_cubes=args.max_cubes,
    )
    semantic_eval_env = wrapper_type(
        env_class(config=default_config),
        num_cubes=default_config.num_cubes,
        max_cubes=args.max_cubes,
    )
    if args.use_pd:
        assert default_config.episode_length % args.pd_duration == 0, "Environment episode length must be divisible by pd_duration"
        episode_length = default_config.episode_length // args.pd_duration
        env = wrap_env(PDWrapper(semantic_env, duration=args.pd_duration), episode_length)
        eval_env = wrap_env(PDWrapper(semantic_eval_env, duration=args.pd_duration), episode_length)
        print(f"Control mode: PD waypoint controls (pd_duration={args.pd_duration})")
    else:
        episode_length = default_config.episode_length
        env = wrap_env(semantic_env, episode_length)
        eval_env = wrap_env(semantic_eval_env, episode_length)
        print("Control mode: raw controls")

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    key_envs = jax.random.split(key_env, args.num_envs)
    env_state = reset_fn(key_envs)
    obs_size = env.observation_size
    action_size = env.action_size
    goal_size = env.goal_size

    log_data_metric_keys = []
    for k in ("obj_reached_once", "obj_lifted", "obj_moved"):
        if k in env_state.metrics.keys():
            log_data_metric_keys.append(k)
    log_data_metric_keys = tuple(log_data_metric_keys)

    if obs_size != layout.observation_size or goal_size != layout.goal_size:
        raise ValueError(
            f"semantic wrapper contract mismatch: obs={obs_size}/"
            f"{layout.observation_size}, goal={goal_size}/{layout.goal_size}"
        )

    if args.critic_family == "dcc":
        networks = make_dcc_networks(
            layout=layout,
            action_size=action_size,
            rep_size=args.rep_size,
            architecture=args.architecture,
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            scale_actor_residual_by_depth=(
                args.scale_actor_residual_by_depth
            ),
            scale_critic_residual_by_depth=(
                args.scale_critic_residual_by_depth
            ),
            use_non_residual_critic_encoders=(
                args.use_non_residual_critic_encoders
            ),
            task_width=args.dcc_task_width,
            task_depth=args.dcc_task_depth,
            combine_mode=args.dcc_combine_mode,
            goal_encoder_mode=args.dcc_goal_encoder_mode,
        )
    elif args.critic_family == "vanilla":
        if args.vanilla_network_type == "set":
            networks = make_vanilla_crl_networks(
                layout=layout,
                action_size=action_size,
                rep_size=args.rep_size,
                width=args.vanilla_width,
                depth=args.vanilla_depth,
            )
        else:
            networks = make_flat_upstream_networks(
                observation_size=layout.observation_size,
                goal_size=layout.goal_size,
                action_size=action_size,
                rep_size=args.rep_size,
                architecture=args.architecture,
                num_blocks=args.num_blocks,
                hidden_dim=args.hidden_dim,
                scale_actor_residual_by_depth=(
                    args.scale_actor_residual_by_depth
                ),
                scale_critic_residual_by_depth=(
                    args.scale_critic_residual_by_depth
                ),
                use_non_residual_critic_encoders=(
                    args.use_non_residual_critic_encoders
                ),
            )
    else:
        raise ValueError(f"unknown critic_family={args.critic_family!r}")

    initialization_key = (
        (key_actor, key_sa, key_g)
        if args.critic_family == "dcc"
        or (
            args.critic_family == "vanilla"
            and args.vanilla_network_type == "flat_upstream"
        )
        else key_sa
    )
    initialized = networks.init_params(initialization_key)
    actor_state = TrainState.create(
        apply_fn=networks.actor.apply,
        params=initialized.pop("actor"),
        tx=optax.adam(learning_rate=args.actor_learning_rate)
    )
    critic_state = TrainState.create(
        apply_fn=None,
        params=initialized,
        tx=optax.adam(learning_rate=args.critic_learning_rate),
    )
    if carry is not None:
        if carry.get("actor") is not None:
            actor_state = actor_state.replace(params=carry["actor"])
        critic_params = dict(critic_state.params)
        if args.critic_family == "dcc" and args.dcc_carry_shared:
            for group in networks.shared_groups:
                critic_params[group] = carry["critic_shared"][group]
        elif (
            args.critic_family == "vanilla"
            and args.vanilla_carry_critic
            and carry.get("critic") is not None
        ):
            critic_params = carry["critic"]
        critic_state = critic_state.replace(params=critic_params)
        if args.critic_family == "dcc":
            print(
                "Continual transfer: "
                f"{'restored' if args.dcc_carry_shared else 'reset'} "
                f"shared DCC groups for task_index={task_index}; reset "
                "phi_task and optimizer state."
            )
        else:
            print(
                "Continual transfer: "
                f"{'restored' if args.vanilla_carry_critic else 'reset'} "
                f"the vanilla critic for task_index={task_index}; "
                "optimizer state is fresh."
            )
    if args.critic_family == "dcc":
        architecture_description = (
            f"flat upstream {args.architecture}, "
            f"blocks={args.num_blocks}, hidden_dim={args.hidden_dim}, "
            f"task_blocks={args.dcc_task_depth}, "
            f"task_hidden_dim={args.dcc_task_width}, "
            f"combine={args.dcc_combine_mode}, "
            f"goal_encoder={args.dcc_goal_encoder_mode}, "
            "no dynamics head"
        )
    else:
        if args.vanilla_network_type == "set":
            architecture_description = (
                f"semantic set width={args.vanilla_width}, "
                f"depth={args.vanilla_depth}"
            )
        else:
            architecture_description = (
                f"flat upstream {args.architecture}, "
                f"blocks={args.num_blocks}, hidden_dim={args.hidden_dim}"
            )
        architecture_description += (
            f", layout={args.observation_layout}, no task/dynamics head"
        )
    print(
        f"\n{args.critic_family.upper()} parameters: "
        f"actor={count_parameters(actor_state.params)}, "
        f"critic={count_parameters(critic_state.params)}; "
        f"max_cubes={args.max_cubes}, {architecture_description}\n"
    )

    # Trainstate
    training_state = CRLTrainingState(
        env_steps=np.zeros((), dtype=np.float64),
        gradient_steps=np.zeros((), dtype=np.float64),
        actor_state=actor_state,
        critic_state=critic_state,
    )

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_goal = jnp.zeros((goal_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        achieved_goal=dummy_goal,
        action=dummy_action,
        extras={
            "state_extras": {
                "traj_id": 0.0,
            }
        },
    )
    def jit_wrap(buffer):
        buffer.insert = jax.jit(buffer.insert)
        buffer.sample = jax.jit(buffer.sample)
        return buffer

    replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=args.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=args.num_envs,
                num_envs=args.num_envs,
                sequence_length=episode_length+1,
            )
        )
    buffer_state = jax.jit(replay_buffer.init)(key_buffer)

    make_policy = make_inference_fn(networks)

    entropy_cost_final = args.entropy_cost if args.entropy_cost_final is None else args.entropy_cost_final
    entropy_decay_steps = max(1, int(total_gradient_steps * args.entropy_decay_fraction))

    def get_entropy_cost(gradient_steps):
        if args.entropy_cost_final is None:
            return jnp.asarray(args.entropy_cost, dtype=jnp.float32)
        progress = jnp.clip(gradient_steps / entropy_decay_steps, 0.0, 1.0)
        entropy_cost = args.entropy_cost + progress * (entropy_cost_final - args.entropy_cost)
        return jnp.asarray(entropy_cost, dtype=jnp.float32)

    if args.entropy_cost_final is None:
        print(f'Entropy config: fixed entropy_cost = {args.entropy_cost}\n')
    else:
        print(
            f'Entropy config: entropy_cost anneals {args.entropy_cost} -> {entropy_cost_final} '
            f'over {entropy_decay_steps} gradient steps '
            f'({100 * args.entropy_decay_fraction:.1f}% of training)\n'
        )

    # Initialize evaluators
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=key_eval,
    )

    def actor_step(training_state, env, env_state, key, extra_fields, metrics_fields):
        g_repr = networks.apply_goal(
            training_state.critic_state.params,
            env_state.info['target_goal'],
        )
        means, log_stds = networks.actor.apply(
            training_state.actor_state.params, env_state.obs, g_repr
        )
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )

        next_env_state = env.step(env_state, actions)

        state_extras = {x: next_env_state.info[x] for x in extra_fields}
        metrics = {x: next_env_state.metrics[x] for x in metrics_fields}

        return training_state, next_env_state, Transition(
                                            observation=env_state.obs,
                                            achieved_goal=env_state.info['achieved_goal'],
                                            action=actions,
                                            extras={"state_extras": state_extras},
                                        ), metrics

    @jax.jit
    def data_collect_step(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused_t):
            training_state, env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            training_state, env_state, transition, metrics = actor_step(
                training_state,
                env,
                env_state,
                current_key,
                extra_fields=("traj_id",),
                metrics_fields=log_data_metric_keys,
            )
            return (training_state, env_state, next_key), (transition, metrics)

        (training_state, env_state, _), (data, metrics) = jax.lax.scan(f, (training_state, env_state, key), (), length=args.rollout_length)

        training_state = training_state.replace(
            env_steps=training_state.env_steps + (args.num_envs * args.rollout_length),
        )

        buffer_state = replay_buffer.insert(buffer_state, data)
        return training_state, env_state, buffer_state, metrics

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            training_state, env_state, buffer_state, _ = data_collect_step(
                training_state,
                env_state,
                buffer_state,
                key,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=np.ceil(args.min_replay_size / args.rollout_length))[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        def actor_loss(actor_params, critic_params, transitions, key, gradient_steps):
            state = transitions.observation
            goal = transitions.extras['future_goal']
            frozen_critic = jax.lax.stop_gradient(critic_params)
            g_repr = networks.apply_goal(frozen_critic, goal)
            means, log_stds = networks.actor.apply(actor_params, state, g_repr)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)           # dimension = B

            sa_repr = networks.apply_sa(frozen_critic, state, action)
            qf_pi = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))

            entropy_cost = get_entropy_cost(gradient_steps)
            entropy_loss = entropy_cost * log_prob
            q_loss = -qf_pi
            actor_loss = jnp.mean(entropy_loss + q_loss)
            gaussian_entropy = jnp.sum(0.5 * (1.0 + jnp.log(2 * jnp.pi)) + log_stds, axis=-1)

            return actor_loss, (log_prob, log_stds, means, action, qf_pi, entropy_cost, entropy_loss, q_loss, gaussian_entropy)

        (
            actorloss,
            (log_prob, log_stds, means, action, qf_pi, entropy_cost, entropy_loss, q_loss, gaussian_entropy),
        ), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            transitions,
            key,
            training_state.gradient_steps,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        training_state = training_state.replace(actor_state=new_actor_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "actor_log_std": log_stds,
            "actor_log_std_min": jnp.min(log_stds),
            "actor_log_std_max": jnp.max(log_stds),
            "actor_mean_abs": jnp.mean(jnp.abs(means), axis=-1),
            "actor_action_abs": jnp.mean(jnp.abs(action), axis=-1),
            "actor_q": qf_pi,
            "actor_q_loss": q_loss,
            "actor_entropy_loss": entropy_loss,
            "actor_pre_tanh_entropy": gaussian_entropy,
            "entropy_cost": entropy_cost,
        }

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, transitions, key):
            state = transitions.observation
            action = transitions.action
            goal = transitions.extras['future_goal']

            sa_repr = networks.apply_sa(critic_params, state, action)
            g_repr = networks.apply_goal(critic_params, goal)

            # InfoNCE
            logits = -jnp.sqrt(jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1)) #shape = BxB

            info_nce_loss = -jnp.mean(
                jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1)
            )

            # logsumexp regularisation
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            logsumexp_loss = args.logsumexp_cost * jnp.mean(logsumexp**2)

            critic_loss = info_nce_loss + logsumexp_loss

            I = jnp.eye(logits.shape[0])
            correct = jnp.argmax(logits, axis=1) == jnp.argmax(I, axis=1)
            logits_pos = jnp.sum(logits * I) / jnp.sum(I)
            logits_neg = jnp.sum(logits * (1 - I)) / jnp.maximum(
                jnp.sum(1 - I), 1.0
            )

            return critic_loss, (
                logsumexp, correct, logits_pos, logits_neg,
                info_nce_loss, logsumexp_loss,
            )

        (loss, aux), grad = jax.value_and_grad(
            critic_loss, has_aux=True
        )(training_state.critic_state.params, transitions, key)
        (
            logsumexp, correct, logits_pos, logits_neg,
            info_nce_loss, logsumexp_loss,
        ) = aux
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state = new_critic_state)

        metrics = {
            "categorical_accuracy": jnp.mean(correct),
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "logsumexp": logsumexp.mean(),
            "critic_loss": loss,
            "critic_info_nce_loss": info_nce_loss,
            "critic_logsumexp_loss": logsumexp_loss,
        }

        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, buffer_state, key = carry

        key, key_critic, key_actor, key_sampling1, key_sampling2 = jax.random.split(key, 5)

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        if args.repetition_factor > 1:
            # CRTR: each unique trajectory appears `repetition_factor` times in the batch, so InfoNCE
            # negatives include same-trajectory samples. The per-element keys below pick independent
            # (anchor t0, future t1) for each repetition, matching the CRTR algorithm.
            n_unique = transitions.observation.shape[0] // args.repetition_factor
            transitions = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x[:n_unique], args.repetition_factor, axis=0), transitions
            )
        batch_keys = jax.random.split(key_sampling1, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
            (args.discount,), transitions, batch_keys
        )
        random_indices = jax.random.randint(key_sampling2, (transitions.action.shape[0],), minval=0, maxval=transitions.action.shape[1])
        transitions = jax.tree_util.tree_map(lambda x: x[jnp.arange(x.shape[0]), random_indices], transitions)

        training_state, actor_metrics = update_actor_and_alpha(transitions, training_state, key_actor)

        training_state, critic_metrics = update_critic(transitions, training_state, key_critic)

        training_state = training_state.replace(gradient_steps = training_state.gradient_steps + 1)

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)

        return (training_state, buffer_state, key,), metrics

    @jax.jit
    def learn_step(training_state, buffer_state, key):

        (training_state, buffer_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, buffer_state, key), (), length=num_sgd_steps_per_training_step)

        return training_state, buffer_state, metrics

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    initial_eval_metrics = evaluator.run_evaluation(
        policy_params={
            "actor": training_state.actor_state.params,
            "critic": training_state.critic_state.params,
        },
        training_metrics={},
    )
    initial_strict_success = float(
        np.asarray(initial_eval_metrics["eval/episode_success_rate"])
    )
    initial_easy_success = float(
        np.asarray(initial_eval_metrics["eval/episode_easy_success_rate"])
    )
    strict_success_auc = OnlineNormalizedAUC(
        initial_value=initial_strict_success
    )
    easy_success_auc = OnlineNormalizedAUC(
        initial_value=initial_easy_success
    )
    task_adaptation = {
        "task_index": task_index,
        "env_id": args.env_id,
        "forward_transfer/initial_success_rate": initial_strict_success,
        "forward_transfer/initial_easy_success_rate": initial_easy_success,
        "adaptation/final_success_rate": initial_strict_success,
        "adaptation/final_easy_success_rate": initial_easy_success,
        "adaptation/success_rate_auc": initial_strict_success,
        "adaptation/easy_success_rate_auc": initial_easy_success,
        "adaptation/budget_env_steps": int(args.num_timesteps),
    }
    print(
        "Initial task performance before data collection: "
        f"success={initial_strict_success:.6f}, "
        f"easy_success={initial_easy_success:.6f}"
    )
    if args.track:
        wandb.log({
            "forward_transfer/initial_success_rate": (
                initial_strict_success
            ),
            "forward_transfer/initial_easy_success_rate": (
                initial_easy_success
            ),
            "training/env_steps": 0,
        }, step=0)
    xt = time.time()
    metrics = None

    print('prefilling replay buffer....')
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    for ts in range(1, args.num_training_step + 1):

        if ts == 1 or ts % max(1, args.num_training_step // 100) == 0:
            print(f'[{ts}/{args.num_training_step}] {100 * ts / args.num_training_step:.1f}%', flush=True)

        key_sgd, key_generate_rollout, key = jax.random.split(key, 3)

        data_collect_start = time.time()
        training_state, env_state, buffer_state, data_metrics = data_collect_step(training_state, env_state, buffer_state, key_generate_rollout)
        data_collect_step_time += time.time() - data_collect_start

        learn_step_start = time.time()
        training_state, buffer_state, training_metrics = learn_step(training_state, buffer_state, key_sgd)
        learn_step_time += time.time() - learn_step_start

        if metrics is None:
            metrics = data_metrics | training_metrics
        else:
            metrics = jax.tree_util.tree_map(
                lambda x, y: x + y, metrics, (data_metrics | training_metrics)
            )

        if args.num_reset_steps > 0 and ts % args.num_training_steps_per_real_reset == 0:
            key_env, key = jax.random.split(key, 2)
            key_envs = jax.random.split(key_env, args.num_envs)
            env_state = reset_fn(key_envs)

        if ts % args.num_training_steps_per_eval == 0:
            es = ts // args.num_training_steps_per_eval

            metrics = jax.tree_util.tree_map(
                lambda x: x / args.num_training_steps_per_eval, metrics
            )
            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            training_step_time = time.time() - xt
            training_walltime += training_step_time

            sps = (
                args.num_training_steps_per_eval
                * args.num_envs * args.rollout_length
            ) / training_step_time

            metrics = {
                'training/sps': sps,
                'training/walltime': training_walltime,
                'training/data_collection_time_fraction' : data_collect_step_time / training_step_time,
                'training/learning_time_fraction' : learn_step_time / training_step_time,
                'training/env_steps': training_state.env_steps,
                **{f'training/{name}': value for name, value in metrics.items()},
                'buffer_current_size': replay_buffer.size(buffer_state),
            }

            metrics = evaluator.run_evaluation(
                policy_params={
                    "actor": training_state.actor_state.params,
                    "critic": training_state.critic_state.params,
                },
                training_metrics=metrics,
            )
            env_steps = float(np.asarray(metrics["training/env_steps"]))
            strict_success = float(
                np.asarray(metrics["eval/episode_success_rate"])
            )
            easy_success = float(
                np.asarray(metrics["eval/episode_easy_success_rate"])
            )
            metrics["eval/episode_success_rate_auc"] = (
                strict_success_auc.update(env_steps, strict_success)
            )
            metrics["eval/episode_easy_success_rate_auc"] = (
                easy_success_auc.update(env_steps, easy_success)
            )
            task_adaptation.update({
                "adaptation/final_success_rate": strict_success,
                "adaptation/final_easy_success_rate": easy_success,
                "adaptation/success_rate_auc": float(
                    metrics["eval/episode_success_rate_auc"]
                ),
                "adaptation/easy_success_rate_auc": float(
                    metrics["eval/episode_easy_success_rate_auc"]
                ),
            })

            video_path_file = None
            if args.record_videos and es % max(1, args.video_every_n_evals) == 0:
                try:
                    key, video_key = jax.random.split(key, 2)
                    policy_params = {
                        "actor": training_state.actor_state.params,
                        "critic": training_state.critic_state.params,
                    }
                    inference_fn = make_policy(policy_params, deterministic=True)
                    video_frames = get_video(args.env_id, inference_fn, eval_env, video_key, episode_length)
                    if args.save_checkpoint:
                        video_dir = f"{save_path}/videos"
                        os.makedirs(video_dir, exist_ok=True)
                        video_path_file = f"{video_dir}/eval_{es}.gif"
                        save_gif(video_frames, video_path_file, fps=args.video_fps)
                        print(f"Saved eval video to {video_path_file}")
                except Exception as e:
                    print(f"Video recording failed at eval {es}: {e}")
                    video_path_file = None

            viz_path_file = None
            if args.visualize_samples and args.save_checkpoint and es % max(1, args.viz_every_n_evals) == 0:
                try:
                    key, viz_key = jax.random.split(key, 2)
                    k_flatten, k_idx = jax.random.split(viz_key, 2)
                    # Sample a batch exactly like sgd_step, but only for visualization.
                    _, viz_trans = replay_buffer.sample(buffer_state)
                    batch_keys = jax.random.split(k_flatten, viz_trans.observation.shape[0])
                    viz_trans = jax.vmap(TrajectoryUniformSamplingQueue.flatten_crl_fn, in_axes=(None, 0, 0))(
                        (args.discount,), viz_trans, batch_keys
                    )
                    random_indices = jax.random.randint(
                        k_idx, (viz_trans.action.shape[0],),
                        minval=0, maxval=viz_trans.action.shape[1],
                    )
                    viz_trans = jax.tree_util.tree_map(
                        lambda x: x[jnp.arange(x.shape[0]), random_indices], viz_trans
                    )
                    achieved_np = np.asarray(viz_trans.achieved_goal)
                    future_np = np.asarray(viz_trans.extras["future_goal"])
                    achieved_np = layout.unpack_goal(
                        achieved_np, default_config.num_cubes
                    )
                    future_np = layout.unpack_goal(
                        future_np, default_config.num_cubes
                    )
                    fig = visualize_crl_batch(
                        eval_env, achieved_np, future_np,
                        num_anchors=args.viz_num_anchors,
                        num_negatives=args.viz_num_negatives,
                    )
                    viz_dir = f"{save_path}/sample_viz"
                    os.makedirs(viz_dir, exist_ok=True)
                    viz_path_file = f"{viz_dir}/samples_{es}.png"
                    fig.savefig(viz_path_file, dpi=100, bbox_inches="tight")
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    print(f"Saved sample viz to {viz_path_file}")
                except Exception as e:
                    print(f"Sample viz failed at eval {es}: {e}")
                    viz_path_file = None

            print(f'\nEvaluation step {es}:\n')
            pprint.pprint(metrics)
            if args.track:
                wandb_payload = dict(metrics)
                if video_path_file is not None and os.path.exists(video_path_file):
                    try:
                        wandb_payload["eval/video"] = wandb.Video(
                            video_path_file,
                            fps=args.video_fps,
                            format="gif",
                        )
                    except Exception as e:
                        print(f"wandb video log failed at eval {es}: {e}")
                if viz_path_file is not None and os.path.exists(viz_path_file):
                    try:
                        wandb_payload["train/samples"] = wandb.Image(
                            viz_path_file
                        )
                    except Exception as e:
                        print(f"wandb sample viz log failed at eval {es}: {e}")
                wandb.log(wandb_payload, step=es)
                if args.wandb_mode == 'offline':
                    trigger_sync()

            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl",
                    params = (
                        training_state.actor_state.params,
                        training_state.critic_state.params,
                    )
                )

                def _to_jsonable(v):
                    if isinstance(v, (jnp.ndarray, np.ndarray)):
                        return v.item() if v.ndim == 0 else v.tolist()
                    if isinstance(v, (np.floating, np.integer, np.bool_)):
                        return v.item()
                    return v
                log_entry = {"eval_step": es, **{k: _to_jsonable(v) for k, v in metrics.items()}}
                with open(f"{save_path}/eval_log.jsonl", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            metrics = None
            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.save_checkpoint:
        log_path = f"{save_path}/eval_log.jsonl"
        if os.path.exists(log_path):
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                steps, success_rates, rewards = [], [], []
                with open(log_path) as f:
                    for line in f:
                        entry = json.loads(line)
                        steps.append(entry.get("eval_step"))
                        success_rates.append(
                            entry.get("eval/episode_success_rate")
                        )
                        rewards.append(entry.get("eval/episode_reward"))

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(steps, success_rates, marker="o")
                axes[0].set_xlabel("Eval step")
                axes[0].set_ylabel("eval/episode_success_rate")
                axes[0].set_title("Episode success rate")
                axes[0].grid(True, alpha=0.3)
                axes[1].plot(
                    steps, rewards, marker="o", color="tab:orange"
                )
                axes[1].set_xlabel("Eval step")
                axes[1].set_ylabel("eval/episode_reward")
                axes[1].set_title("Episode reward")
                axes[1].grid(True, alpha=0.3)
                fig.suptitle(args.exp_name)
                fig.tight_layout()
                plot_path = f"{save_path}/metrics.png"
                fig.savefig(plot_path, dpi=120)
                plt.close(fig)
                print(f"Saved metrics plot to {plot_path}")
            except Exception as error:
                print(
                    "metrics plot skipped; checkpoints and eval_log.jsonl "
                    f"are valid: {error}"
                )

    if args.save_checkpoint:
        with open(f"{save_path}/task_adaptation.json", "w") as stream:
            json.dump(task_adaptation, stream, indent=2, sort_keys=True)
            stream.write("\n")

    if args.track:
        if wandb.run is not None:
            for name, value in task_adaptation.items():
                if name not in {"task_index", "env_id"}:
                    wandb.run.summary[name] = value
        wandb.finish()

    task_adaptation_history = (
        list(carry.get("task_adaptation", ()))
        if carry is not None
        else []
    )
    while len(task_adaptation_history) < task_index:
        task_adaptation_history.append(None)
    if task_index < len(task_adaptation_history):
        task_adaptation_history[task_index] = task_adaptation
    else:
        task_adaptation_history.append(task_adaptation)

    if args.critic_family == "vanilla":
        return {
            "actor": training_state.actor_state.params,
            "critic": training_state.critic_state.params,
            "task_index": task_index,
            "env_id": args.env_id,
            "task_adaptation": task_adaptation_history,
        }

    task_bank = list(carry.get("task_bank", ())) if carry is not None else []
    task_snapshot = {
        "task_index": task_index,
        "env_id": args.env_id,
        "phi_task": training_state.critic_state.params["phi_task"],
    }
    if task_index < len(task_bank):
        task_bank[task_index] = task_snapshot
    else:
        task_bank.append(task_snapshot)

    return {
        "actor": training_state.actor_state.params,
        "critic_shared": {
            group: training_state.critic_state.params[group]
            for group in networks.shared_groups
        },
        "task_index": task_index,
        "env_id": args.env_id,
        "task_bank": task_bank,
        "task_adaptation": task_adaptation_history,
    }

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
