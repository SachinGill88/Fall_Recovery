"""
UCR v0H Humanoid Robot Specific Configuration
Get specific positions for robot joints and fallen poses
based on analysis of the UCR v0H MJCF XML model.
"""

import numpy as np
from typing import Dict, List, Tuple

class UCRv0HConfig:
    """Configuration class for UCR v0H humanoid robot training"""
    
    # Robot specifications from XML analysis
    ROBOT_SPECS = {
        "name": "UCR v0H",
        "total_mass": 45.0,  # Approximate total mass (kg)
        "standing_height": 0.95,  # Pelvis height in standing pose (m)
        "total_dof": 23,  # Number of actuated joints
        "control_freq": 50,  # Control frequency (Hz)
        "sim_timestep": 0.001,  # MuJoCo timestep from XML
    }
    
    # Joint specifications based on actuator classes
    JOINT_INFO = {
        # Format: joint_name: (actuator_class, force_limit, joint_range)
        "leftHipYaw": ("A900", 120, (-1.58, 1.58)),
        "rightHipYaw": ("A900", 120, (-1.58, 1.58)),
        "torsoYaw": ("A900", 120, (-1.58, 1.58)),
        
        "leftHipRoll": ("D110a", 190, (-0.5236, 0.4695)),
        "rightHipRoll": ("D110a", 190, (-0.4695, 0.5236)),
        "torsoPitch": ("A900", 140, (-0.0873, 0.5263)),  # Parallel drive
        
        "leftHipPitch": ("D110a", 190, (-1.58, 1.58)),
        "rightHipPitch": ("D110a", 190, (-1.58, 1.58)),
        "torsoRoll": ("A900", 140, (-0.1396, 0.1396)),  # Parallel drive
        
        "leftKneePitch": ("D110a", 190, (-0.1, 2.7925)),
        "rightKneePitch": ("D110a", 190, (-0.1, 2.7925)),
        
        "leftShoulderPitch": ("B903", 116, (-3.14, 3.14)),
        "rightShoulderPitch": ("B903", 116, (-3.14, 3.14)),
        
        "leftAnklePitch": ("D110a", 190, (-1.2217, 0.4363)),
        "rightAnklePitch": ("D110a", 190, (-1.2217, 0.4363)),
        
        "leftShoulderRoll": ("B903", 116, (-3.14, 3.14)),
        "rightShoulderRoll": ("B903", 116, (-3.14, 3.14)),
        
        "leftAnkleRoll": ("A71", 51, (-0.7854, 0.7854)),
        "rightAnkleRoll": ("A71", 51, (-0.7854, 0.7854)),
        
        "leftShoulderYaw": ("C806", 91, (-3.14, 3.14)),
        "rightShoulderYaw": ("C806", 91, (-3.14, 3.14)),
        "leftElbow": ("C806", 91, (-0.3822, 3.1)),
        "rightElbow": ("C806", 91, (-3.1, 0.3822)),
    }
    
    @classmethod
    def get_training_config(cls, stage: str = "stage1") -> Dict:
        """Get training configuration optimized for UCR v0H"""
        
        base_config = {
            # Environment settings
            "model_path": "model/mjcf/xml/v0H_pos.xml",
            "max_episode_steps": 500,  # 10 seconds at 50Hz
            "randomize_terrain": False,
            "control_frequency": cls.ROBOT_SPECS["control_freq"],
            
            # Robot-specific settings
            "standing_height": cls.ROBOT_SPECS["standing_height"],
            "target_height_threshold": 0.8 * cls.ROBOT_SPECS["standing_height"],  # 0.76m
            "stability_threshold": 0.5,  # rad/s angular velocity
            "upright_threshold": 0.9,  # quaternion similarity
            
            # Training settings
            "algorithm": "PPO",  # PPO works well for humanoids
            "n_envs": 8,  # Parallel environments
            "device": "auto",  # Will detect CUDA automatically
            "seed": 42,
            
            # General RL hyperparameters
            "gamma": 0.99,
            "use_wandb": True,
            "log_freq": 1000,
            "eval_freq": 50000,
            "save_freq": 100000,
        }
        
        # Stage-specific configurations
        if stage == "stage1":
            stage_config = {
                "stage1_timesteps": 2000000,  # 2M steps for complex robot
                "stage1_lr": 3e-4,
                "stage1_reward_threshold": 200,  # Higher threshold for complex robot
                
                # Stage 1 specific reward weights
                "reward_weights": {
                    "upright": 15.0,        # Primary goal
                    "height": 10.0,         # Get off the ground
                    "stability": 3.0,       # Basic stability
                    "progress": 5.0,        # Progress toward standing
                    "energy": -0.05,        # Light energy penalty
                    "contact": -0.02,       # Light contact penalty
                    "joint_limits": -0.1,   # Avoid extreme poses
                },
                
                # Exploration settings for stage 1
                "exploration": {
                    "entropy_coef": 0.01,   # Higher exploration
                    "clip_range": 0.2,
                    "n_epochs": 10,
                }
            }
        else:  # stage2
            stage_config = {
                "stage2_timesteps": 3000000,  # 3M steps for refinement
                "stage2_lr": 1e-4,  # Lower learning rate
                
                # Stage 2 specific reward weights  
                "reward_weights": {
                    "upright": 20.0,        # Strong upright preference
                    "height": 15.0,         # Strong height preference
                    "stability": 8.0,       # Higher stability requirement
                    "progress": 5.0,        # Maintain progress reward
                    "energy": -0.3,         # Stronger energy penalty
                    "contact": -0.1,        # Reduce ground contact
                    "joint_limits": -0.5,   # Stronger joint limit penalty
                    "smoothness": 3.0,      # Smooth motion reward
                    "torque_rate": -0.2,    # Penalize rapid torque changes
                },
                
                # Exploitation settings for stage 2
                "exploration": {
                    "entropy_coef": 0.005,  # Lower exploration
                    "clip_range": 0.15,     # Tighter clipping
                    "n_epochs": 15,         # More training epochs
                }
            }
        
        # Merge configurations
        config = {**base_config, **stage_config}
        return config
    
    @classmethod 
    def get_fallen_pose_templates(cls) -> List[Dict]:
        """Get realistic fallen pose templates for UCR v0H"""
        
        templates = [
            {
                "name": "back_fall",
                "description": "Robot fallen on back",
                "base_orientation": [1, 0, 0, 0],  # Upright
                "base_height": 0.2,
                "joint_positions": {
                    # Legs bent to avoid self-collision
                    "leftHipPitch": -0.5,
                    "rightHipPitch": -0.5,
                    "leftKneePitch": 1.0,
                    "rightKneePitch": 1.0,
                    # Arms spread for stability
                    "leftShoulderRoll": 1.5,
                    "rightShoulderRoll": -1.5,
                }
            },
            {
                "name": "side_fall_left",
                "description": "Robot fallen on left side",
                "base_orientation": [0.707, 0, 0, 0.707],  # 90° roll
                "base_height": 0.25,
                "joint_positions": {
                    # Asymmetric leg positions
                    "leftHipRoll": 0.3,
                    "rightHipRoll": -0.3,
                    "leftKneePitch": 0.8,
                    "rightKneePitch": 1.2,
                    # Arms positioned naturally
                    "leftShoulderPitch": -0.5,
                    "rightShoulderPitch": 0.5,
                }
            },
            {
                "name": "side_fall_right", 
                "description": "Robot fallen on right side",
                "base_orientation": [0.707, 0, 0, -0.707],  # -90° roll
                "base_height": 0.25,
                "joint_positions": {
                    # Mirror of left side fall
                    "leftHipRoll": -0.3,
                    "rightHipRoll": 0.3,
                    "leftKneePitch": 1.2,
                    "rightKneePitch": 0.8,
                    "leftShoulderPitch": 0.5,
                    "rightShoulderPitch": -0.5,
                }
            },
            {
                "name": "front_fall",
                "description": "Robot fallen on front/stomach",
                "base_orientation": [0, 1, 0, 0],  # 180° pitch
                "base_height": 0.15,
                "joint_positions": {
                    # Protective arm position
                    "leftShoulderPitch": 1.5,
                    "rightShoulderPitch": 1.5,
                    "leftElbow": -1.5,
                    "rightElbow": -1.5,
                    # Legs slightly bent
                    "leftKneePitch": 0.3,
                    "rightKneePitch": 0.3,
                }
            },
            {
                "name": "sitting_fall",
                "description": "Robot in sitting position",
                "base_orientation": [1, 0, 0, 0],
                "base_height": 0.4,
                "joint_positions": {
                    # Sitting pose with legs extended
                    "leftHipPitch": 1.2,
                    "rightHipPitch": 1.2,
                    "leftKneePitch": -0.1,
                    "rightKneePitch": -0.1,
                    # Torso slightly forward
                    "torsoPitch": 0.3,
                }
            },
            {
                "name": "twisted_fall",
                "description": "Complex twisted fall pose",
                "base_orientation": [0.5, 0.5, 0.5, 0.5],  # Complex rotation
                "base_height": 0.2,
                "joint_positions": {
                    # Asymmetric, challenging pose
                    "torsoYaw": 0.8,
                    "leftHipYaw": -0.5,
                    "rightHipYaw": 0.5,
                    "leftKneePitch": 1.5,
                    "rightKneePitch": 0.5,
                    "leftShoulderPitch": -1.0,
                    "rightShoulderPitch": 1.5,
                }
            }
        ]
        
        return templates
    
    @classmethod
    def get_joint_order(cls) -> List[str]:
        """Get the joint order as defined in the XML actuator section"""
        return [
            "leftHipYaw", "rightHipYaw", "torsoYaw",
            "leftHipRoll", "rightHipRoll", "torsoPitch", 
            "leftHipPitch", "rightHipPitch", "torsoRoll",
            "leftKneePitch", "rightKneePitch",
            "leftShoulderPitch", "rightShoulderPitch",
            "leftAnklePitch", "rightAnklePitch",
            "leftShoulderRoll", "rightShoulderRoll",
            "leftAnkleRoll", "rightAnkleRoll",
            "leftShoulderYaw", "rightShoulderYaw",
            "leftElbow", "rightElbow"
        ]
    
    @classmethod
    def get_actuator_limits(cls) -> np.ndarray:
        """Get actuator force limits in joint order"""
        joint_order = cls.get_joint_order()
        limits = []
        
        for joint_name in joint_order:
            if joint_name in cls.JOINT_INFO:
                _, force_limit, _ = cls.JOINT_INFO[joint_name]
                limits.append(force_limit)
            else:
                limits.append(100.0)  # Default limit
        
        return np.array(limits)
    
    @classmethod
    def validate_pose(cls, joint_positions: Dict[str, float]) -> bool:
        """Validate that joint positions are within limits"""
        for joint_name, position in joint_positions.items():
            if joint_name in cls.JOINT_INFO:
                _, _, (min_pos, max_pos) = cls.JOINT_INFO[joint_name]
                if not (min_pos <= position <= max_pos):
                    print(f"Warning: {joint_name} position {position} outside range [{min_pos}, {max_pos}]")
                    return False
        return True

# Example usage and testing
if __name__ == "__main__":
    # Get stage 1 configuration
    config = UCRv0HConfig.get_training_config("stage1")
    print("Stage 1 Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nUCR v0H Specifications:")
    for key, value in UCRv0HConfig.ROBOT_SPECS.items():
        print(f"  {key}: {value}")
    
    print(f"\nJoint Order ({len(UCRv0HConfig.get_joint_order())} joints):")
    for i, joint in enumerate(UCRv0HConfig.get_joint_order()):
        print(f"  {i}: {joint}")
    
    print(f"\nActuator Force Limits:")
    limits = UCRv0HConfig.get_actuator_limits()
    for i, (joint, limit) in enumerate(zip(UCRv0HConfig.get_joint_order(), limits)):
        print(f"  {joint}: {limit} N⋅m")
    
    print(f"\nFallen Pose Templates:")
    templates = UCRv0HConfig.get_fallen_pose_templates()
    for template in templates:
        print(f"  {template['name']}: {template['description']}")