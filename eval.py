import os
import jax
import tyro
import functools
import numpy as np

from pathlib import Path
from pprint import pprint
from dataclasses import dataclass
from utils.networks import load_params

AGENT = "ppo"

if AGENT == "ppo":
    from ppo import Args as PPOArgs
    @dataclass
    class Args(PPOArgs):
        folder_path: str = "checkpoints/"
        num_envs: int = 1

else:
    raise NotImplementedError

def main(args: Args):

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    if args.agent in ["ppo"]:
        
        from builderbench.env_utils import make_env
        from utils.wrapper import wrap_env
        env_class, default_config = make_env(args)
        default_config.nconmax, default_config.njmax = default_config.nconmax * args.num_eval_envs, default_config.njmax
        eval_env = wrap_env( env_class(config=default_config), default_config.episode_length )  
        action_size = eval_env.action_size

        from utils.evaluation import Evaluator

        from ppo import PPONetworks, Actor, Value
        ppo_network = PPONetworks( 
            policy_network = Actor(layer_sizes=args.policy_hidden_sizes + [action_size * 2]),
        value_network = Value(layer_sizes=args.value_hidden_sizes  + [1]),
        )

        from ppo import make_inference_fn
        make_policy = make_inference_fn(ppo_network)

    else:
        raise NotImplementedError

    key = jax.random.PRNGKey(args.seed)
    key, key_eval = jax.random.split(key)

    evaluator = Evaluator(
        eval_env,
        functools.partial(make_policy, deterministic=True),
        num_eval_envs=args.num_eval_envs,
        episode_length=default_config.episode_length,
        key=key_eval,
    )

    folder_path = Path(args.folder_path)

    
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and args.env_id in subfolder.name:
            print(f"\nSubfolder: {subfolder}")

            video_path = f"{folder_path.parent}/videos/{subfolder.name}/"
            os.makedirs(video_path, exist_ok=True)

            for param_file in subfolder.iterdir():  

                if not Path( f"{video_path}/{param_file.stem}.mp4" ).exists(): 

                    params = load_params(f"{param_file}")
                    actor_params, _, normalize_params = params

                    metrics = evaluator.run_evaluation(
                        policy_params={'policy':actor_params, 'normalizer':normalize_params},
                        training_metrics={},
                    )
                    print(f"Evaluating {param_file}")
                    pprint(metrics)

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)