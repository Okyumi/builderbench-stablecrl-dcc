import os

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["MUJOCO_GL"] = "egl"

import time
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
from typing import Any, NamedTuple, Sequence
from wandb_osh.hooks import TriggerWandbSyncHook

from utils.wrapper import wrap_env
from utils.evaluation import Evaluator
from utils.networks import MLP, save_params
from utils.jax import count_parameters
from builderbench.env_utils import make_env
from utils.buffer import TrajectoryUniformSamplingQueue

@dataclass
class Args:
    # experiment
    agent: str = "sac-her"
    seed: int = 1
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    
    # logging and checkpointing
    track: bool = False
    wandb_project_name: str = "builderbench"
    wandb_entity: str = 'raj19'
    wandb_mode: str = 'online'
    wandb_dir: str = './'
    wandb_group: str = 'default'
    wandb_name_tag: str = ''

    num_eval_steps: int = 50             # number of evaluation / logging / saving steps
    num_reset_steps: int = 50             # number of times to call true resets (env.reset) instead of soft resets (AutoResetWrapper)

    save_checkpoint: bool = True

    # environment
    env_id: str = 'creative-1-task1'
    num_envs: int = 1024
    num_eval_envs: int = 128
    env_early_termination: bool = True
    env_episode_length: int = None
    permutation_invariant_reward: bool = True   # invariance to the order of cubes in any structure

    # algorithm
    num_timesteps: int = 50000000
    policy_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    value_hidden_sizes: list = field(default_factory=lambda: [256, 256, 256, 256])
    rollout_length: int = 64
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    discount: float = 0.99
    entropy_cost: float = 0.1
    tau: float = 0.01
    relabelling_prob: float = 0.8
    max_replay_size: int = 10000
    min_replay_size: int = 1000
    relabelling_prob: float = 0.8

    diagnostic: bool = False

class CriticTrainState(TrainState):
    """trainstate for critic that also stores target parameters"""
    target_params: Any

@flax.struct.dataclass
class SACTrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    commanded_goal: jnp.ndarray
    achieved_goal: jnp.ndarray
    action: jnp.ndarray
    done: jnp.ndarray
    next_observation: jnp.ndarray
    extras: jnp.ndarray = ()

class Critic(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = True
    kernel_init: Any = variance_scaling(1/3, "fan_in", "uniform")
    bias_init: Any = nn.initializers.zeros
    final_kernet_init: Any = nn.initializers.zeros
    final_bias_init: Any = nn.initializers.zeros

    def setup(self):
        self.q1 = MLP(
            layer_sizes=self.layer_sizes, 
            activation=self.activation, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init,
            final_kernet_init=self.final_kernet_init, 
            final_bias_init=self.final_bias_init, 
            layer_norm=self.layer_norm
        )

        self.q2 = MLP(
            layer_sizes=self.layer_sizes, 
            activation=self.activation, 
            kernel_init=self.kernel_init, 
            bias_init=self.bias_init,
            final_kernet_init=self.final_kernet_init, 
            final_bias_init=self.final_bias_init, 
            layer_norm=self.layer_norm
        )
    def __call__(self, observations, actions, goals):

        inputs = jnp.concatenate([observations, actions, goals], axis=-1)
        q1 = self.q1(inputs)
        q2 = self.q2(inputs)
        return jnp.concatenate([q1, q2], axis=-1)

class Actor(nn.Module):
    layer_sizes: Sequence[int]
    activation: Any = nn.swish
    layer_norm: bool = True
    kernel_init: Any = variance_scaling(1/3, "fan_in", "uniform")
    bias_init: Any = nn.initializers.zeros

    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def setup(self):
        self.actor_net = MLP(layer_sizes=self.layer_sizes, activation=self.activation, kernel_init=self.kernel_init, bias_init=self.bias_init, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        x = self.actor_net(x)
        mean, log_std = jnp.split(x, 2, axis=-1)
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std

def make_inference_fn(policy_network):
    """Creates params and inference function for the SAC agent."""
    def make_policy(params, deterministic: bool = False):

        def policy(observations, goals, key_sample):
            
            means, log_stds = policy_network.apply(params, observations, goals)
                
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

def main(args: Args):
    args.num_training_step = args.num_timesteps // ( args.num_envs * args.rollout_length )
    args.num_training_steps_per_eval = args.num_training_step // args.num_eval_steps
    args.num_training_steps_per_real_reset = args.num_training_step // max(1, args.num_reset_steps)

    print(f"Total number of training steps = {args.num_training_step}")
    print(f"Total number of gradient steps per training step = { (args.rollout_length * args.num_envs) // args.num_envs}")
    print(f"Total number of env steps per training step = {args.num_envs * args.rollout_length}")
    print(f"Data to update ratio = {  ( args.num_envs * args.rollout_length ) / ( args.rollout_length * args.num_envs // args.num_envs )}")

    args.exp_name = f"{args.wandb_name_tag + '__' if args.wandb_name_tag != '' else ''}{args.env_id}__{args.seed}__{os.path.basename(__file__)[: -len('.py')]}__{int(time.time())}"
    
    # Initialize wandb if tracking is enabled
    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
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
    key, key_buffer, key_env, key_eval, key_actor, key_critic = jax.random.split(key, 6)

    # Initialize environment
    env_class, default_config = make_env(args)
    env = wrap_env( env_class(config=default_config), default_config.episode_length )
    eval_env = wrap_env( env_class(config=default_config), default_config.episode_length )  
    episode_length = default_config.episode_length

    # Initialize checkpoint folder
    if args.save_checkpoint:
        save_path = Path(args.wandb_dir) / f"checkpoints/{args.exp_name}/"
        os.makedirs(save_path, exist_ok=True)

    reset_fn = jax.jit(env.reset)
    her_reward_fn = jax.jit(jax.vmap(env.get_reward_from_goals))
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

    # Network setup
    # Actor
    actor = Actor(layer_sizes=args.policy_hidden_sizes + [action_size * 2])
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(key_actor, np.ones([1, obs_size]), np.ones([1, goal_size])),
        tx=optax.adam(learning_rate=args.actor_learning_rate)
    )
    # Critic
    critic = Critic(layer_sizes=args.value_hidden_sizes  + [1])
    tmp_init_critic_params = critic.init(key_critic, np.ones([1, obs_size]), np.ones([1, action_size]), np.ones([1, goal_size]))
    critic_state = CriticTrainState.create(
        apply_fn=critic.apply,
        params=tmp_init_critic_params,
        target_params=tmp_init_critic_params,
        tx=optax.adam(learning_rate=args.critic_learning_rate)
    )
    del tmp_init_critic_params

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)
    make_policy = make_inference_fn(actor)

    # Trainstate
    training_state = SACTrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
    )

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_goal = jnp.zeros((goal_size,))
    dummy_action = jnp.zeros((action_size,))

    dummy_transition = Transition(
        observation=dummy_obs,
        commanded_goal=dummy_goal,
        achieved_goal=dummy_goal,
        action=dummy_action,
        done=jnp.zeros(()),
        next_observation=dummy_obs,
        extras={
            "state_extras": {
                "truncation": 0.0,
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

    # Initialize evaluators
    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=episode_length,
        key=key_eval,
    )

    def actor_step(training_state, env, env_state, key, extra_fields, metrics_fields):
        means, log_stds = actor.apply(training_state.actor_state.params, env_state.obs, env_state.info['target_goal'])
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )
        
        next_env_state = env.step(env_state, actions)

        state_extras = {x: next_env_state.info[x] for x in extra_fields}
        metrics = {x: next_env_state.metrics[x] for x in metrics_fields}

        return training_state, next_env_state, Transition(
                                            observation=env_state.obs,
                                            commanded_goal=env_state.info['target_goal'],
                                            achieved_goal=env_state.info['achieved_goal'],
                                            action=actions,
                                            done=next_env_state.done,
                                            next_observation=next_env_state.obs,
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
                extra_fields=("truncation", "traj_id"),
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
        def actor_loss(actor_params, critic_params, transitions, key):
            state = transitions.observation
            commanded_goal = transitions.commanded_goal

            means, log_stds = actor.apply(actor_params, state, commanded_goal)
            stds = jnp.exp(log_stds)
            x_ts = means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            action = nn.tanh(x_ts)
            log_prob = jax.scipy.stats.norm.logpdf(x_ts, loc=means, scale=stds)
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)           # dimension = B

            qf_pi = critic.apply(
                jax.lax.stop_gradient(critic_params), 
                state, 
                action, 
                commanded_goal,
            )
            qf_pi = jnp.min(qf_pi, axis=-1)

            actor_loss = jnp.mean( args.entropy_cost * log_prob - (qf_pi) )

            return actor_loss, log_prob
        
        (actorloss, log_prob), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(training_state.actor_state.params, training_state.critic_state.params, transitions, key)
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        training_state = training_state.replace(actor_state=new_actor_state)

        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
        }

        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        def critic_loss(critic_params, critic_target_params, actor_params, transitions, key):
            
            # prepare goal and reward data
            achieved_goal = transitions.achieved_goal
            future_goal = transitions.extras['future_goals']
            commanded_goal = transitions.commanded_goal
            relabelling_mask = jax.random.bernoulli(key, args.relabelling_prob, shape=(future_goal.shape[0], 1))
            goal = jnp.where(relabelling_mask, future_goal, commanded_goal)
            reward = her_reward_fn(achieved_goal, goal)
            
            # get params
            actor_params = jax.lax.stop_gradient(actor_params)
            target_critic_params = jax.lax.stop_gradient(critic_target_params)
            
            next_means, next_log_stds = actor.apply(actor_params, transitions.next_observation, goal)
            next_stds = jnp.exp(next_log_stds)
            next_x_ts = next_means + next_stds * jax.random.normal(key, shape=next_means.shape, dtype=next_means.dtype)
            next_actions = nn.tanh(next_x_ts)
            next_log_prob = jax.scipy.stats.norm.logpdf(next_x_ts, loc=next_means, scale=next_stds)
            next_log_prob -= jnp.log((1 - jnp.square(next_actions)) + 1e-6)
            next_log_prob = next_log_prob.sum(-1)

            next_v = jnp.min( critic.apply(target_critic_params, transitions.next_observation, next_actions, goal), axis=-1 ) - args.entropy_cost * next_log_prob
            target_q = reward + args.discount * (1 - transitions.done) * next_v

            q = critic.apply(critic_params, transitions.observation, transitions.action, goal)

            q_error = q - jnp.expand_dims(target_q, -1)

            # Better bootstrapping for truncated episodes.
            truncation = transitions.extras['state_extras']['truncation']
            q_error *= jnp.expand_dims(1 - truncation, -1)

            critic_loss = 0.5 * jnp.mean(jnp.square(q_error))

            metric = {
                "q_error" : jnp.mean( q_error ),
                "buffer_reward" : jnp.mean( reward ),
                "target_q" : jnp.mean( target_q ),
                "target_q_min" : jnp.min( target_q ),
                "target_q_max" :  jnp.max( target_q ),
            }

            return critic_loss, metric
            
        (loss, metrics), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params,
            training_state.critic_state.target_params,
            training_state.actor_state.params,
            transitions, 
            key
        )

        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        new_critic_state = new_critic_state.replace(
            target_params = jax.tree_util.tree_map(
                lambda x, y: x * (1 - args.tau) + y * args.tau,
                new_critic_state.target_params,
                new_critic_state.params,
            )
        )
        training_state = training_state.replace(critic_state = new_critic_state)

        return training_state, metrics
    
    @jax.jit
    def sgd_step(carry, transitions):
        training_state, buffer_state, key = carry

        key, key_critic, key_actor, key_sampling1, key_sampling2 = jax.random.split(key, 5)

        buffer_state, transitions = replay_buffer.sample(buffer_state)
        batch_keys = jax.random.split(key_sampling1, transitions.observation.shape[0])
        transitions = jax.vmap(TrajectoryUniformSamplingQueue.flatten_sac_her_fn, in_axes=(None, 0, 0))(
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

        num_sgd_steps = (args.rollout_length * args.num_envs) // args.num_envs
        (training_state, buffer_state, _,), metrics = jax.lax.scan(sgd_step, (training_state, buffer_state, key), (), length=num_sgd_steps)

        return training_state, buffer_state, metrics

    training_walltime, data_collect_step_time, learn_step_time = 0, 0, 0
    xt = time.time()
    metrics = None

    print('prefilling replay buffer....')
    key, prefill_key = jax.random.split(key, 2)
    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )

    for ts in range(1, args.num_training_step + 1):
        
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
                policy_params=training_state.actor_state.params,
                training_metrics=metrics,
            )

            print(f'\nEvaluation step {es}:\n')
            pprint.pprint(metrics)
            if args.track:
                wandb.log(metrics, step=es)
                if args.wandb_mode == 'offline':
                    trigger_sync()
            metrics = None
            
            if args.save_checkpoint:
                save_params(
                    f"{save_path}/params_{es}.pkl", 
                    params = (
                        training_state.actor_state.params,
                        training_state.critic_state.params,
                    )
                )

            xt, data_collect_step_time, learn_step_time = time.time(), 0, 0

    if args.track:
        wandb.finish()
            
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)    