# to-do

1) For the creative cube mode, play around with action scales to make sure which values are best for RL.
2) Directly get mat from data.xmat or something instead of doing mjx.math.quat_to_mat.
3) Try reward normalization for ppo.

    
    # def get_obs(self, data, info):

    #     obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
    #     achieved_goal = obj_pos[ self._target_cube_masks_data ].reshape(-1,)
    #     obj_pos = obj_pos.reshape(-1,)
    #     # obj_quat = data.qpos[(self._objs_qposadr + 3)[:, None] + np.arange(4)].reshape(-1,)
    #     obj_mat = data.xmat[self._objs_ids][:, : 2].reshape(-1,)

    #     obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)
    #     obj_angvel = data.qvel[(self._objs_qveladr + 3)[:, None] + np.arange(3)].reshape(-1,)

    #     select_action = info["select_action"]

    #     obs = jnp.concatenate([
    #         obj_pos,
    #         obj_mat,
    #         obj_linvel,
    #         obj_angvel,
    #         select_action[None],
    #     ])

    #     info.update({
    #         "achieved_goal": achieved_goal, 
    #     })

    #     return obs, info
    
    # def get_permutation_invariant_reward_from_obs(self, data, info):
    #     obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
    #     obj_mat = data.xmat[self._objs_ids].reshape(self._num_task_cubes, -1)[:, :6]
    #     achieved_goal = obj_pos[ self._target_cube_masks_data ]
    #     obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

    #     target_goal = info["target_goal"].reshape((self._num_task_cubes, -1))
    #     target_mat = jax.vmap(mjx_math.quat_to_mat)(data.mocap_quat[self._task_mocap_targets]).reshape(self._num_task_cubes, -1)[:, :6]

    #     obj_target_pos_squared_pairwise_err = jnp.sum( (achieved_goal[None, :, :] - target_goal[:, None, :]) ** 2, axis=-1)
    #     cube_ids, target_ids = optax.assignment.hungarian_algorithm( obj_target_pos_squared_pairwise_err )
    #     obj_target_pos_err = jnp.sqrt( obj_target_pos_squared_pairwise_err[cube_ids, target_ids] )
    #     obj_target_rot_err = jnp.sqrt( jnp.sum( ( (obj_mat[cube_ids] - target_mat[target_ids]) ) ** 2, axis=-1) ) / 6

    #     obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
    #     obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)
        
    #     reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * (0.9 * obj_target_pos_err + 0.1 * obj_target_rot_err)))

    #     success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
    #     easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

    #     reward_info = {
    #         "success": success,
    #         "easy_success":  easy_success,
    #         "obj_lifted": obj_lifted,
    #         "obj_moved": obj_moved,
    #         "obj_goal_dist": jnp.sum( obj_target_pos_err ),
    #     }

    #     return reward, reward_info

    # def get_permutation_variant_reward_from_obs(self, data, info):
    #     obj_pos = data.qpos[self._objs_qposadr[:, None] + np.arange(3)]
    #     obj_mat = data.xmat[self._objs_ids].reshape(self._num_task_cubes, -1)[:, :6]
    #     achieved_goal = obj_pos[ self._target_cube_masks_data ]
    #     obj_linvel = data.qvel[self._objs_qveladr[:, None] + np.arange(3)].reshape(-1,)

    #     target_goal = info["target_goal"].reshape((self._num_task_cubes, -1))
    #     target_mat = jax.vmap(mjx_math.quat_to_mat)(data.mocap_quat[self._task_mocap_targets]).reshape(self._num_task_cubes, -1)[:, :6]
            
    #     obj_target_pos_err = jnp.linalg.norm(target_goal - achieved_goal, axis=-1)
    #     obj_target_rot_err = jnp.sqrt( jnp.sum( ( (obj_mat - target_mat) ) ** 2, axis=-1) ) / 6

    #     obj_lifted = jnp.sum( obj_pos[:, 2] > 0.05 ).astype(float)
    #     obj_moved = jnp.any( obj_linvel > 0.001 ).astype(float)

    #     reward = jnp.sum(1 - jnp.tanh(self._config.reward_sensitivity * (0.9 * obj_target_pos_err + 0.1 * obj_target_rot_err)))

    #     success = jnp.all(obj_target_pos_err < self._config.success_threshold).astype(float)
    #     easy_success = jnp.all(obj_target_pos_err < self._config.easy_success_threshold).astype(float)

    #     reward_info = {
    #         "success": success,
    #         "easy_success":  easy_success,
    #         "obj_lifted": obj_lifted,
    #         "obj_moved": obj_moved,
    #         "obj_goal_dist": jnp.sum( obj_target_pos_err ),
    #     }

    #     return reward, reward_info
