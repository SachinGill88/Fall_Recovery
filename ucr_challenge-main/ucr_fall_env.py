"""
UCR v0H Fall Recovery Environment - Fixed for actual robot dimensions
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
from typing import Dict, Any, Tuple, Optional

class UCRFallRecoveryEnv(gym.Env):
    """
    I based this on HumanUP methodology with two-stage training approach.

    """
    
    def __init__(
        self,
        model_path: str = "model/mjcf/xml/v0H_pos.xml",
        max_episode_steps: int = 500,
        training_stage: str = "stage1",  # "stage1" or "stage2" 
        randomize_terrain: bool = False,
        render_mode: str = None
    ):
        super().__init__()
        
        # Load MuJoCo model
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Environment parameters
        self.max_episode_steps = max_episode_steps
        self.training_stage = training_stage
        self.randomize_terrain = randomize_terrain
        self.render_mode = render_mode
        
        # Initialize viewer if needed
        self.viewer = None
        
        # Episode tracking
        self.step_count = 0
        self.initial_qpos = None
        self.target_height = 0.76  # Target standing height (80% of 0.95m) - may need adjustment
        
        # Get robot dimensions
        self.robot_height = self._get_robot_height()
        self.n_joints = self.model.nq  # 30
        self.n_dof = self.model.nv     # 29
        self.n_actuators = self.model.nu # 23
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Reward weights (different for each stage)
        self._setup_reward_weights()
        
        print(f"UCR v0H Fall Recovery Environment initialized:")
        print(f"  - Total Joints: {self.n_joints}")
        print(f"  - DOF: {self.n_dof}")
        print(f"  - Actuators: {self.n_actuators}")
        print(f"  - Training Stage: {self.training_stage}")
        print(f"  - Max Episode Steps: {self.max_episode_steps}")
        print(f"  - Target Height: {self.target_height:.2f}m")
    
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        
        # Action space: normalized joint torques [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(self.n_actuators,),  # 23
            dtype=np.float32
        )
        
        
        obs_dim = (
            self.n_joints +      
            self.n_dof +          
            16 +                
            2 +                  
            3 +                  
            self.n_actuators   
        )   # Total: 103 dimensions
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        print(f"Action space: {self.action_space.shape} (expected: 23)")
        print(f"Observation space: {self.observation_space.shape} (calculated: {obs_dim})")
    
    def _setup_reward_weights(self):
        """Setup reward weights based on training stage"""
        if self.training_stage == "stage1":
            # First stage idea is to get upright quickly
            self.reward_weights = {
                "upright": 15.0,          # Primary goal: get upright
                "height": 10.0,           # Get off the ground
                "stability": 3.0,         # Basic stability
                "progress": 5.0,          # Progress toward standing
                "energy": -0.05,          # Light energy penalty
                "contact": -0.02,         # Light contact penalty
                "joint_limits": -0.1,     # Avoid extreme poses
            }
            ## current results have robot getting upright by jumping -- look into modifying height reward to be more gradual
        
        else:  # stage2
            # Stage 2: Focus on realistic, deployable motion
            self.reward_weights = {
                "upright": 20.0,          # Strong upright preference
                "height": 15.0,           # Strong height preference
                "stability": 8.0,         # Higher stability requirement
                "progress": 5.0,          # Maintain progress reward
                "energy": -0.3,           # Stronger energy penalty
                "contact": -0.1,          # Reduce ground contact
                "joint_limits": -0.5,     # Stronger joint limit penalty
                "smoothness": 3.0,        # Smooth motion reward
                "torque_rate": -0.2,      # Penalize rapid torque changes
            }
    
    def _get_robot_height(self):
        """Get robot's standing height from pelvis position"""
        # Reset to default pose and measure height
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'pelvis')
        if pelvis_id == -1:

            return 0.95  # Default UCR v0H standing height
        
        return self.data.xpos[pelvis_id, 2] 
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment to random fallen pose"""
        super().reset(seed=seed)
        
        # Reset MuJoCo
        mujoco.mj_resetData(self.model, self.data)
        
        # Generate random fallen pose
        self._generate_random_fallen_pose()
        
        # Store initial pose for reference
        self.initial_qpos = self.data.qpos.copy()
        
        # Reset counters
        self.step_count = 0
        self.previous_action = np.zeros(self.n_actuators)
        
        # Forward simulation to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _generate_random_fallen_pose(self):
        """Generate random fallen poses - 6 realistic poses """
        from ucr_v0h_config import UCRv0HConfig
        
        # Get fallen pose templates
        templates = UCRv0HConfig.get_fallen_pose_templates()
        
        # Choose random template
        template = np.random.choice(templates)
        
        # Set base position with small randomization
        self.data.qpos[0] = np.random.uniform(-0.1, 0.1)  # x
        self.data.qpos[1] = np.random.uniform(-0.1, 0.1)  # y  
        self.data.qpos[2] = template["base_height"] + np.random.uniform(-0.05, 0.05)
        
        # Set base orientation from template with small noise
        base_quat = np.array(template["base_orientation"])
        noise_angle = np.random.uniform(-0.2, 0.2)  # Small rotation noise
        noise_axis = np.random.uniform(-1, 1, 3)
        noise_axis = noise_axis / (np.linalg.norm(noise_axis) + 1e-8)
        
        # Use template orientation directly for simplicity
        self.data.qpos[3:7] = base_quat
        
        # Set joint positions from template
        joint_order = UCRv0HConfig.get_joint_order()
        template_joints = template["joint_positions"]
        
        # Start from joint index 7 (after floating base: 3 pos + 4 quat)
        joint_start_idx = 7
        
        for i, joint_name in enumerate(joint_order):
            joint_idx = joint_start_idx + i
            
            if joint_idx < len(self.data.qpos):  # Safety check
                if joint_name in template_joints:
                    # Use template position with small randomization
                    template_pos = template_joints[joint_name]
                    noise = np.random.uniform(-0.1, 0.1)  # Small joint noise
                    self.data.qpos[joint_idx] = template_pos + noise
                else:
                    # Small random position for unspecified joints
                    self.data.qpos[joint_idx] = np.random.uniform(-0.1, 0.1)
        
        # Ensure joint limits are respected
        for i in range(joint_start_idx, len(self.data.qpos)):
            if i - joint_start_idx < len(self.model.jnt_range):
                joint_range_idx = i - joint_start_idx
                if joint_range_idx < len(self.model.jnt_range):
                    low, high = self.model.jnt_range[joint_range_idx]
                    self.data.qpos[i] = np.clip(self.data.qpos[i], low, high)
        
        # Zero initial velocities
        self.data.qvel[:] = 0
        
        if hasattr(self, '_debug') and self._debug:
            print(f"Generated fallen pose: {template['name']} - {template['description']}")
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        # Clip and scale action
        action = np.clip(action, -1.0, 1.0)
        
        # Scale action to actuator limits
        scaled_action = self._scale_action(action)
        
        # Apply action
        self.data.ctrl[:] = scaled_action
        
        # Step simulation (multiple substeps for stability)
        substeps = 5
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation and reward
        observation = self._get_observation()
        reward = self._compute_reward(action)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        # Update counters
        self.step_count += 1
        self.previous_action = action.copy()
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized action [-1,1] to actuator torque limits"""
        from ucr_v0h_config import UCRv0HConfig
        
        # Get UCR v0H specific actuator limits
        torque_limits = UCRv0HConfig.get_actuator_limits()
        
        # Ensure we have the right number of limits
        if len(torque_limits) != len(action):
            print(f"Warning: torque_limits length ({len(torque_limits)}) != action length ({len(action)})")
            torque_limits = np.full(len(action), 100.0)  # Default fallback
        
        return action * torque_limits
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation using UCR v0H sensors"""
        
        # Joint positions and velocities (actual dimensions from your robot)
        qpos = self.data.qpos.copy()  # 30 values 
        qvel = self.data.qvel.copy()  # 29 values
        
        # Get IMU data from pelvis sensor
        imu_data = self._get_imu_data()  # 16 values
        
        # Foot contact forces (from touch sensors)
        contact_forces = self._get_contact_info()  # 2 values
        
        # Center of mass position
        com_pos = self._get_com_position()  # 3 values
        
        # Combine all observations
        obs = np.concatenate([
            qpos,                   # 30
            qvel,                   # 29
            imu_data,               # 16
            contact_forces,         # 2
            com_pos,                # 3
            self.previous_action    # 23
        ])  # Total: 103 dimensions
        
        return obs.astype(np.float32)
    
    def _get_imu_data(self) -> np.ndarray:
        """Get IMU data from pelvis sensor"""
        imu_data = np.zeros(16)
        
        try:
            # Get sensor IDs for pelvis IMU
            pos_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'pos_pelvis')
            ori_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'ori_pelvis')
            linvel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'linvel_pelvis')
            angvel_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'angvel_pelvis')
            linacc_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'linacc_pelvis')
            
            idx = 0
            
            # Position (3)
            if pos_id != -1:
                imu_data[idx:idx+3] = self.data.sensordata[pos_id:pos_id+3]
            else:
                imu_data[idx:idx+3] = self.data.qpos[0:3]  # Fallback to qpos
            idx += 3
            
            # Orientation quaternion (4)
            if ori_id != -1:
                imu_data[idx:idx+4] = self.data.sensordata[ori_id:ori_id+4]
            else:
                imu_data[idx:idx+4] = self.data.qpos[3:7]  # Fallback to qpos
            idx += 4
            
            # Linear velocity (3)
            if linvel_id != -1:
                imu_data[idx:idx+3] = self.data.sensordata[linvel_id:linvel_id+3]
            else:
                imu_data[idx:idx+3] = self.data.qvel[0:3]  # Fallback to qvel
            idx += 3
            
            # Angular velocity (3)
            if angvel_id != -1:
                imu_data[idx:idx+3] = self.data.sensordata[angvel_id:angvel_id+3]
            else:
                imu_data[idx:idx+3] = self.data.qvel[3:6]  # Fallback to qvel
            idx += 3
            
            # Linear acceleration (3)
            if linacc_id != -1:
                imu_data[idx:idx+3] = self.data.sensordata[linacc_id:linacc_id+3]
            # else: leave as zeros (no fallback for acceleration)
                
        except Exception as e:
            # Fallback to basic data from joint positions/velocities
            imu_data[0:3] = self.data.qpos[0:3]  # Position
            imu_data[3:7] = self.data.qpos[3:7]  # Quaternion - this is complex # that represents rotation 
            imu_data[7:10] = self.data.qvel[0:3]  # Linear velocity
            imu_data[10:13] = self.data.qvel[3:6]  # Angular velocity
            # Leave acceleration as zeros
        
        return imu_data
    
    def _get_contact_info(self) -> np.ndarray:
        """Get UCR v0H foot contact information using built-in touch sensors"""
        contact_info = np.zeros(2)  # [left_foot, right_foot]
        
        try:
            # Find foot contact sensor IDs
            left_contact_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'left_foot_contact'
            )
            right_contact_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'right_foot_contact'
            )
            
            if left_contact_id != -1:
                contact_info[0] = self.data.sensordata[left_contact_id]
            if right_contact_id != -1:
                contact_info[1] = self.data.sensordata[right_contact_id]
                
        except:
            # Fallback to general contact detection
            for i in range(self.data.ncon):
                contact = self.data.contact[i]
                if contact.dist < 0.01:  # In contact
                    # Simplified contact detection
                    contact_info[0] = 1.0
                    contact_info[1] = 1.0
                    break
        
        return contact_info
    
    def _get_com_position(self) -> np.ndarray:
        """Get center of mass position"""
        total_mass = 0
        com = np.zeros(3)
        
        for i in range(self.model.nbody):
            body_mass = self.model.body_mass[i]
            body_pos = self.data.xpos[i]
            com += body_mass * body_pos
            total_mass += body_mass
        
        return com / total_mass if total_mass > 0 else np.zeros(3)
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """Compute reward based on current state and action"""
        
        reward = 0.0
        info = {}
        
        # Get robot's current orientation and height
        root_quat = self.data.qpos[3:7]
        root_pos = self.data.qpos[:3]
        current_height = root_pos[2]
        
        # 1. Upright reward (based on quaternion)
        # Upright = quaternion close to [1, 0, 0, 0] (no rotation) - this is key for standing! 
        upright_quat = np.array([1, 0, 0, 0])
        upright_similarity = np.dot(root_quat, upright_quat) ** 2
        upright_reward = self.reward_weights["upright"] * upright_similarity
        reward += upright_reward
        info["upright_reward"] = upright_reward
        
        # 2. Height reward
        height_progress = current_height / self.target_height
        height_reward = self.reward_weights["height"] * min(height_progress, 1.0)
        reward += height_reward  
        info["height_reward"] = height_reward
        
        # 3. Stability reward (low angular velocity)
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        stability_reward = self.reward_weights["stability"] * np.exp(-angular_vel)
        reward += stability_reward
        info["stability_reward"] = stability_reward
        
        # 4. Progress reward (getting closer to upright + high)
        progress_score = upright_similarity * height_progress
        progress_reward = self.reward_weights["progress"] * progress_score
        reward += progress_reward
        info["progress_reward"] = progress_reward
        
        # 5. Energy penalty
        energy_penalty = self.reward_weights["energy"] * np.sum(action ** 2)
        reward += energy_penalty
        info["energy_penalty"] = energy_penalty
        
        # 6. Contact penalty (encourage getting off ground)
        contact_forces = self._get_contact_info()
        contact_penalty = self.reward_weights["contact"] * np.sum(contact_forces)
        reward += contact_penalty
        info["contact_penalty"] = contact_penalty
        
        # 7. Joint limits penalty
        joint_limits_penalty = 0
        for i in range(7, len(self.data.qpos)):  # Skip floating base
            joint_idx = i - 7
            if joint_idx < len(self.model.jnt_range):
                low, high = self.model.jnt_range[joint_idx]
                q = self.data.qpos[i]
                if q < low or q > high:
                    joint_limits_penalty += abs(q - np.clip(q, low, high))
        
        joint_limits_penalty *= self.reward_weights["joint_limits"]
        reward += joint_limits_penalty
        info["joint_limits_penalty"] = joint_limits_penalty
        
        # ---------------- need too look into stage 2 more ----------------
        # Stage 2 specific rewards
        if self.training_stage == "stage2":
            # Smoothness reward (penalize action changes)
            if hasattr(self, 'previous_action'):
                action_diff = np.linalg.norm(action - self.previous_action)
                smoothness_reward = self.reward_weights["smoothness"] * np.exp(-action_diff)
                reward += smoothness_reward
                info["smoothness_reward"] = smoothness_reward
            
            # Torque rate penalty (penalize rapid torque changes)
            if hasattr(self, 'previous_action'):
                torque_rate = np.sum((action - self.previous_action) ** 2)
                torque_rate_penalty = self.reward_weights["torque_rate"] * torque_rate
                reward += torque_rate_penalty
                info["torque_rate_penalty"] = torque_rate_penalty
        
        return reward
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate (success condition)"""
        
        # Success: robot is upright and stable
        root_quat = self.data.qpos[3:7] 
        current_height = self.data.qpos[2]
        angular_vel = np.linalg.norm(self.data.qvel[3:6])
        
        # Check if upright (quaternion close to identity)
        upright_quat = np.array([1, 0, 0, 0])
        upright_similarity = np.dot(root_quat, upright_quat) ** 2
        
        is_upright = upright_similarity > 0.9
        is_high_enough = current_height > self.target_height
        is_stable = angular_vel < 0.5
        
        success = is_upright and is_high_enough and is_stable
        
        return success
    
    def _get_info(self) -> Dict:
        """Get additional info about current state"""
        root_quat = self.data.qpos[3:7]
        current_height = self.data.qpos[2]
        
        upright_quat = np.array([1, 0, 0, 0]) 
        upright_similarity = np.dot(root_quat, upright_quat) ** 2
        
        return {
            "current_height": current_height,
            "target_height": self.target_height,
            "upright_similarity": upright_similarity,
            "is_upright": upright_similarity > 0.9,
            "step_count": self.step_count,
            "training_stage": self.training_stage
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model, self.data
                )
            self.viewer.sync()
    
    def close(self):
        """Clean up resources"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


# Example usage and testing - thanks to Claude for testing structure
if __name__ == "__main__":
    # Test the environment
    print("🧪 Testing UCR v0H Fall Recovery Environment...")
    
    try:
        env = UCRFallRecoveryEnv(
            model_path="model/mjcf/xml/v0H_pos.xml",
            training_stage="stage1",
            render_mode=None  # Set to "human" to see visualization
        )
        
        print("✅ Environment created successfully!")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful - Observation shape: {obs.shape}")
        print(f"   Initial height: {info['current_height']:.3f}m")
        print(f"   Target height: {info['target_height']:.3f}m")
        
        # Test a few steps
        print(f"\n🔄 Testing environment steps...")
        for step in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: reward={reward:.3f}, height={info['current_height']:.3f}m, upright={info['upright_similarity']:.3f}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {step+1}")
                break
        
        env.close()
        print("✅ Environment test completed successfully!")
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()

        """Get robot's standing height from pelvis position"""
        #