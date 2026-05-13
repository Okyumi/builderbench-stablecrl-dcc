# Experiment Commands
All runs are trained for 200,000,000 timesteps.

Baseline CRL (Short horizon/PD controller)
```
python stable_crl.py --env_id creative-4-task1 --use_pd --pd_duration 5 --architecture default --num_blocks 8
```
StableCRL (Short horizon/PD controller)
```
python stable_crl.py --env_id creative-4-task1 --use_pd --pd_duration 5 --architecture default --repetition_factor 12 --entropy_cost 0.01 
```
StableCRL Scaled (Short horizon/PD controller)
```
python stable_crl.py --env_id creative-4-task1 --use_pd --pd_duration 5 --architecture block --num_blocks 8 --repetition_factor 12 --entropy_cost 0.01 
```

StableCRL Scaled (Long horizon/Raw controller)
```
python stable_crl.py --env_id creative-4-task1 --no-use_pd --architecture block --num_blocks 8 --repetition_factor 12 --entropy_cost 0.01 
```

PPO (Short horizon/PD controller)
```
python ppo_pd.py --env_id creative-4-task1 --use_pd --pd_duration 5
```

PPO (Long horizon/Raw controller)
```
python ppo_pd.py --env_id creative-4-task1 --no-use_pd
```

## Demos

### Long Horizon (3 Stack)

![Long Horizon (3 Stack)](gifs/3_raw.gif)

### PD-5 (5 stack)

![PD-5 (5 stack)](gifs/5_pd.gif)

<!--1) For the creative cube mode, play around with action scales to make sure which values are best for RL.
2) Directly get mat from data.xmat or something instead of doing mjx.math.quat_to_mat.
3) Try reward normalization for ppo.-->
