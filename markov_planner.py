import mujoco
import mujoco.viewer
import numpy as np
import math
import time

def quat_to_yaw(quat):
    """Converts a quaternion (w, x, y, z) to yaw (z-axis rotation)."""
    w, x, y, z = quat
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def normalize_angle(angle):
    """Normalize an angle to be within [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))

class BlockPDController:
    """
    A stateless PD controller to move a specified block.
    The force applied is a pure function of the current state and the goal state.
    """
    def __init__(self, model, data, block_name, dt):
        self.model = model
        self.data = data
        self.dt = dt # Note: dt isn't strictly needed for PD, but good to have

        # --- Get block-specific IDs ---
        self.geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, block_name)
        self.body_id = model.geom_bodyid[self.geom_id]

        # Get qpos and qvel addresses
        self.qpos_addr = model.jnt_qposadr[model.body(f"{block_name}").jntadr[0]]
        self.qvel_addr = model.jnt_dofadr[model.body(f"{block_name}").jntadr[0]]

        # Get actuator IDs
        self.act_x_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"x_{block_name}")
        self.act_y_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"y_{block_name}")
        self.act_z_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"z_{block_name}")
        self.act_yaw_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"yaw_{block_name}")

        # --- Get block mass for gravity compensation ---
        self.mass = model.body_mass[self.body_id]
        self.gravity_comp = self.mass * 9.81

        # --- PD Gains (These require careful tuning!) ---
        # Position PD
        self.Kp_pos = 10.0
        self.Kd_pos = 2.0  # Kd is crucial for stability (damping)
        
        # Yaw (Rotation) PD
        self.Kp_rot = 1.0
        self.Kd_rot = 0.5

        # --- Actuator Ranges ---
        self.ctrl_range_pos = model.actuator_ctrlrange[[self.act_x_id, self.act_y_id, self.act_z_id]]
        self.ctrl_range_yaw = model.actuator_ctrlrange[self.act_yaw_id]

    def get_state(self):
        """Returns the current (position, yaw, velocity, yaw_velocity) of the block."""
        pos = self.data.qpos[self.qpos_addr : self.qpos_addr + 3]
        quat = self.data.qpos[self.qpos_addr + 3 : self.qpos_addr + 7]
        vel = self.data.qvel[self.qvel_addr : self.qvel_addr + 3]
        ang_vel = self.data.qvel[self.qvel_addr + 3 : self.qvel_addr + 6]

        yaw = quat_to_yaw(quat)
        yaw_vel = ang_vel[2] # Z-axis angular velocity

        return pos, yaw, vel, yaw_vel

    def get_dynamic_setpoint(self, current_pos, final_goal_pos, safe_height, horizontal_thresh=0.01):
        """
        This is the "reactive planner."
        It decides the *immediate* setpoint based only on the current state.
        """
        current_xy = current_pos[:2]
        goal_xy = final_goal_pos[:2]
        
        # Calculate horizontal distance to the final goal's XY coordinates
        horizontal_dist_to_goal = np.linalg.norm(current_xy - goal_xy)
        
        # Calculate horizontal distance to the start's XY coordinates (approx)
        # This is a bit of a heuristic; assumes we start near the block
        # A more robust way might be to pass start_pos, but this is more "Markovian"
        
        current_setpoint = np.copy(final_goal_pos)

        # 1. IF we are not at a safe height AND we are not yet above the goal
        if current_pos[2] < (safe_height - 0.005) and horizontal_dist_to_goal > horizontal_thresh:
            # Command: GO UP
            current_setpoint[0] = current_pos[0] # Stay at current X
            current_setpoint[1] = current_pos[1] # Stay at current Y
            current_setpoint[2] = safe_height    # Go to safe height
        
        # 2. IF we are at safe height BUT not yet above the goal
        elif current_pos[2] >= (safe_height - 0.005) and horizontal_dist_to_goal > horizontal_thresh:
            # Command: GO ACROSS
            current_setpoint[0] = final_goal_pos[0] # Go to goal X
            current_setpoint[1] = final_goal_pos[1] # Go to goal Y
            current_setpoint[2] = safe_height       # Stay at safe height

        # 3. IF we are (roughly) above the goal
        else:
            # Command: GO DOWN
            current_setpoint = final_goal_pos # Go to final goal position
            
        return current_setpoint

    def update_controls(self, setpoint_pos, setpoint_yaw):
        """Calculates and applies the PD control forces."""
        
        current_pos, current_yaw, current_vel, current_yaw_vel = self.get_state()

        # --- Position Control (X, Y, Z) ---
        error_pos = setpoint_pos - current_pos
        
        # Use velocity for derivative term (more stable than error derivative)
        # Goal velocity is 0
        derivative_pos = -current_vel 

        # PD output for position
        output_pos = (self.Kp_pos * error_pos) + (self.Kd_pos * derivative_pos)
        
        # Add gravity compensation for Z-axis
        output_pos[2] += self.gravity_comp

        # --- Yaw Control (Z-rotation) ---
        error_yaw = normalize_angle(setpoint_yaw - current_yaw)
        derivative_yaw = -current_yaw_vel

        # PD output for yaw
        output_yaw = (self.Kp_rot * error_yaw) + (self.Kd_rot * derivative_yaw)

        # --- Apply and Clamp Controls ---
        clamped_fx = np.clip(output_pos[0], self.ctrl_range_pos[0, 0], self.ctrl_range_pos[0, 1])
        clamped_fy = np.clip(output_pos[1], self.ctrl_range_pos[1, 0], self.ctrl_range_pos[1, 1])
        clamped_fz = np.clip(output_pos[2], self.ctrl_range_pos[2, 0], self.ctrl_range_pos[2, 1])
        clamped_tyaw = np.clip(output_yaw, self.ctrl_range_yaw[0], self.ctrl_range_yaw[1])

        # Set controls
        self.data.ctrl[self.act_x_id] = clamped_fx
        self.data.ctrl[self.act_y_id] = clamped_fy
        self.data.ctrl[self.act_z_id] = clamped_fz
        self.data.ctrl[self.act_yaw_id] = clamped_tyaw

def main():
    xml_path = "tmp/4-1.xml"
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"Error loading XML: {e}")
        print("Please make sure 'cube_fixed.xml' is in the same directory.")
        return
        
    data = mujoco.MjData(model)
    
    # --- Configuration ---
    BLOCK_TO_MOVE = "block_3" # Change this to "block_1", "block_2", etc.
    GOAL_POSITION = np.array([0.2, 0.1, 0.1]) # Target [x, y, z]
    GOAL_YAW = 0.0 # Target yaw (radians)
    SAFE_HEIGHT_OFFSET = 0.1 # How high to lift the block
    COMPLETION_THRESHOLD = 0.01 # How close to get to a setpoint (meters)
    # ---------------------
    
    ctrl_dt = 0.02
    sim_dt = model.opt.timestep
    nstep = int(ctrl_dt / sim_dt)

    # Initialize the PD controller
    controller = BlockPDController(model, data, BLOCK_TO_MOVE, ctrl_dt)
    
    # Get the block's starting z-position
    start_pos, _, _, _ = controller.get_state()
    # Safe height is relative to the *floor*, not the start pos.
    # Assuming start_pos[2] is the block's height (0.02)
    safe_height = start_pos[2] + SAFE_HEIGHT_OFFSET
    
    print(f"Moving {BLOCK_TO_MOVE} to {GOAL_POSITION} using a reactive PD controller.")
    print(f"Safe height set to: {safe_height}")
    
    goal_reached = False

    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # --- Get Current State ---
            current_pos, _, _, _ = controller.get_state()

            # --- Reactive Planner Step ---
            # At every step, calculate the *immediate* setpoint based on the current state
            current_setpoint = controller.get_dynamic_setpoint(
                current_pos, 
                GOAL_POSITION, 
                safe_height
            )
            
            # --- Controller Step ---
            # Apply forces based *only* on current state and immediate setpoint
            controller.update_controls(current_setpoint, GOAL_YAW)
            
            # --- MuJoCo Step ---
            mujoco.mj_step(model, data, nstep=nstep)
            
            # --- Check if plan is complete ---
            dist_to_final_goal = np.linalg.norm(current_pos - GOAL_POSITION)
            
            if not goal_reached and dist_to_final_goal < COMPLETION_THRESHOLD:
                print("Goal reached!")
                goal_reached = True
                # The controller will now just hold the block at the goal position
            
            # Sync viewer
            viewer.sync()

            # Rudimentary sleep to aim for real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()