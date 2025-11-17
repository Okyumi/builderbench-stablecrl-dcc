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

class BlockPlanner:
    """
    A PID controller to move a specified block along a trajectory.
    """
    def __init__(self, model, data, block_name, dt):
        self.model = model
        self.data = data
        self.dt = dt

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

        # --- PID Gains (These require careful tuning!) ---
        # Position PID
        self.Kp_pos = 10.0
        self.Ki_pos = 0.5
        self.Kd_pos = 2.0
        
        # Yaw (Rotation) PID
        self.Kp_rot = 1.0
        self.Ki_rot = 0.1
        self.Kd_rot = 0.5

        # --- PID State Variables ---
        self.integral_pos = np.zeros(3)
        self.integral_yaw = 0.0

        # --- Actuator Ranges ---
        self.ctrl_range_pos = model.actuator_ctrlrange[[self.act_x_id, self.act_y_id, self.act_z_id]]
        self.ctrl_range_yaw = model.actuator_ctrlrange[self.act_yaw_id]

    def get_state(self):
        """Returns the current (position, yaw, velocity, yaw_velocity) of the block."""
        pos = self.data.qpos[self.qpos_addr : self.qpos_addr + 3]
        # Note: MuJoCo free joint qpos is [x, y, z, w, x, y, z]
        quat = self.data.qpos[self.qpos_addr + 3 : self.qpos_addr + 7]
        
        vel = self.data.qvel[self.qvel_addr : self.qvel_addr + 3]
        # Note: MuJoCo free joint qvel is [vx, vy, vz, wx, wy, wz]
        ang_vel = self.data.qvel[self.qvel_addr + 3 : self.qvel_addr + 6]

        yaw = quat_to_yaw(quat)
        yaw_vel = ang_vel[2] # Z-axis angular velocity

        return pos, yaw, vel, yaw_vel

    def update_controls(self, setpoint_pos, setpoint_yaw):
        """Calculates and applies the PID control forces."""
        
        current_pos, current_yaw, current_vel, current_yaw_vel = self.get_state()

        # --- Position Control (X, Y, Z) ---
        error_pos = setpoint_pos - current_pos
        self.integral_pos += error_pos * self.dt
        
        # Use velocity for derivative term (more stable than error derivative)
        derivative_pos = -current_vel 

        # PID output for position
        output_pos = (self.Kp_pos * error_pos) + \
                     (self.Ki_pos * self.integral_pos) + \
                     (self.Kd_pos * derivative_pos)
        
        # Add gravity compensation for Z-axis
        output_pos[2] += self.gravity_comp

        # --- Yaw Control (Z-rotation) ---
        error_yaw = normalize_angle(setpoint_yaw - current_yaw)
        self.integral_yaw += error_yaw * self.dt
        derivative_yaw = -current_yaw_vel

        # PID output for yaw
        output_yaw = (self.Kp_rot * error_yaw) + \
                     (self.Ki_rot * self.integral_yaw) + \
                     (self.Kd_rot * derivative_yaw)

        # --- Apply and Clamp Controls ---
        # Clamp position forces
        clamped_fx = np.clip(output_pos[0], self.ctrl_range_pos[0, 0], self.ctrl_range_pos[0, 1])
        clamped_fy = np.clip(output_pos[1], self.ctrl_range_pos[1, 0], self.ctrl_range_pos[1, 1])
        clamped_fz = np.clip(output_pos[2], self.ctrl_range_pos[2, 0], self.ctrl_range_pos[2, 1])
        
        # Clamp yaw torque
        clamped_tyaw = np.clip(output_yaw, self.ctrl_range_yaw[0], self.ctrl_range_yaw[1])

        # Set controls
        self.data.ctrl[self.act_x_id] = clamped_fx
        self.data.ctrl[self.act_y_id] = clamped_fy
        self.data.ctrl[self.act_z_id] = clamped_fz
        self.data.ctrl[self.act_yaw_id] = clamped_tyaw

    def generate_trajectory(self, start_pos, goal_pos, safe_height_offset=0.1):
        """Creates a simple 3-stage (Lift, Move, Place) trajectory."""
        
        # 1. Lift position: Straight up from start
        lift_pos = start_pos.copy()
        lift_pos[2] += safe_height_offset
        
        # 2. Move position: Horizontally to a point above the goal
        move_pos = goal_pos.copy()
        move_pos[2] = lift_pos[2] # Maintain safe height
        
        # 3. Place position: Straight down to the goal
        place_pos = goal_pos.copy()
        
        return [lift_pos, move_pos, place_pos]

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
    GOAL_POSITION = np.array([0.2, 0.1, 0.02]) # Target [x, y, z]
    GOAL_YAW = 0.0 # Target yaw (radians)
    SAFE_HEIGHT_OFFSET = 0.1 # How high to lift the block
    COMPLETION_THRESHOLD = 0.01 # How close to get to a setpoint (meters)
    # ---------------------

    ctrl_dt = 0.02
    sim_dt = model.opt.timestep
    nstep = int(ctrl_dt / sim_dt)

    # Initialize the planner
    planner = BlockPlanner(model, data, BLOCK_TO_MOVE, ctrl_dt)
    
    # Get the block's starting position
    start_pos, _, _, _ = planner.get_state()
    print(f"Moving {BLOCK_TO_MOVE} from {start_pos} to {GOAL_POSITION}")
    
    # Generate the 3-stage plan
    trajectory = planner.generate_trajectory(start_pos, GOAL_POSITION, SAFE_HEIGHT_OFFSET)
    
    current_stage = 0
    current_setpoint = trajectory[current_stage]
    print(f"Stage 0 (Lift): -> {current_setpoint}")
    
    # Launch the viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- Planner/Controller Step ---
            planner.update_controls(current_setpoint, GOAL_YAW)
            
            # --- MuJoCo Step ---
            mujoco.mj_step(model, data, nstep=nstep)
            
            # --- Check if stage is complete ---
            current_pos, _, _, _ = planner.get_state()
            dist_to_setpoint = np.linalg.norm(current_pos - current_setpoint)
            
            if dist_to_setpoint < COMPLETION_THRESHOLD:
                if current_stage < len(trajectory) - 1:
                    # Move to next stage
                    current_stage += 1
                    current_setpoint = trajectory[current_stage]
                    print(f"Stage {current_stage} ({['Lift','Move','Place'][current_stage]}): -> {current_setpoint}")
                    # Reset integral terms to prevent windup
                    planner.integral_pos.fill(0)
                    planner.integral_yaw = 0.0
                else:
                    # Plan is complete
                    if time.time() - start_time > 1.0: # Show complete for 1 sec
                        print("Plan complete!")
                        # You could break here, or just hold the final position
                        # For this demo, we'll just hold the position
                        pass 

            # Sync viewer
            viewer.sync()

            # Rudimentary sleep to aim for real-time
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()