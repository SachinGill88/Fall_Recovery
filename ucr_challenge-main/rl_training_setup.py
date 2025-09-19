"""
RL Training Setup for UCR v0H Humanoid Fall Recovery
Final fix for environment factory function structure
"""

import os
import numpy as np
import torch
import wandb
from datetime import datetime
from typing import Dict, Any, Callable

# Stable-Baselines3 imports
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, 
    CallbackList, StopTrainingOnRewardThreshold
)
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# Import the custom environment and config
from ucr_fall_env import UCRFallRecoveryEnv
from ucr_v0h_config import UCRv0HConfig

class WandbCallback(BaseCallback):
    """Custom callback to log metrics to Weights & Biases"""
    
    def __init__(self, log_freq: int = 100):
        super().__init__()
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Log training metrics
            if len(self.model.ep_info_buffer) > 0:
                ep_info = self.model.ep_info_buffer[-1]
                wandb.log({
                    "train/episode_reward": ep_info.get("r", 0),
                    "train/episode_length": ep_info.get("l", 0),
                    "train/timesteps": self.num_timesteps
                })
        return True

class ProgressiveTrainingCallback(BaseCallback):
    """Callback to handle progressive curriculum learning"""
    
    def __init__(self, stage_transition_steps: int = 100000):
        super().__init__()
        self.stage_transition_steps = stage_transition_steps
        self.stage_transitioned = False
        
    def _on_step(self) -> bool:
        if (self.num_timesteps > self.stage_transition_steps and 
            not self.stage_transitioned):
            
            print(f"\n Transitioning to Stage 2 training at step {self.num_timesteps}")
            
            if wandb.run:
                wandb.log({
                    "curriculum/stage_transition": self.num_timesteps,
                    "curriculum/current_stage": 2
                })
            
            self.stage_transitioned = True
            
        return True

class UCRTrainer:
    """Main training class for UCR v0H humanoid fall recovery"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ucr_config = UCRv0HConfig()
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = f"experiments/ucr_v0h_fall_recovery_{timestamp}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        print(f" UCR v0H Training Setup Initialized")
        print(f" Experiment Directory: {self.experiment_dir}")
        print(f"Robot: {UCRv0HConfig.ROBOT_SPECS['name']}")
        print(f" Mass: {UCRv0HConfig.ROBOT_SPECS['total_mass']}kg")
        print(f" Height: {UCRv0HConfig.ROBOT_SPECS['standing_height']}m")
        print(f"DOF: {UCRv0HConfig.ROBOT_SPECS['total_dof']}")
        
        # Setup environments after initialization
        self.setup_environment()
        
        # Save configuration
        self._save_config()
    
    def _save_config(self):
        """Save configuration to file"""
        import json
        config_save_path = f"{self.experiment_dir}/config.json"
        with open(config_save_path, 'w') as f:
            # Convert numpy arrays and other non-serializable objects to lists/strings
            serializable_config = {}
            for k, v in self.config.items():
                if isinstance(v, np.ndarray):
                    serializable_config[k] = v.tolist()
                elif isinstance(v, dict):
                    serializable_config[k] = v
                else:
                    serializable_config[k] = str(v)
            json.dump(serializable_config, f, indent=2)
        print(f"💾 Configuration saved to: {config_save_path}")
        
    def setup_environment(self):
        """Setup training and evaluation environments (Windows compatible)"""
        
        # Fixed environment factory function
        def make_env(rank: int = 0, stage: str = "stage1"):
            """
            Create a single environment instance.
            This function directly returns the environment, not another function.
            """
            set_random_seed(self.config.get("seed", 42) + rank)
            
            env = UCRFallRecoveryEnv(
                model_path=self.config["model_path"],
                max_episode_steps=self.config["max_episode_steps"],
                training_stage=stage,
                randomize_terrain=self.config.get("randomize_terrain", False)
            )
            env = Monitor(env)
            env.reset(seed=self.config.get("seed", 42) + rank)
            return env
        
        # Create vectorized environments using DummyVecEnv (Windows compatible)
        print(f" Creating {self.config['n_envs']} parallel environments...")
        print(f" Using DummyVecEnv for Windows compatibility")
        
        # Create list of environment factories
        env_fns = [lambda i=i: make_env(i, "stage1") for i in range(self.config["n_envs"])]
        
        # Use DummyVecEnv instead of SubprocVecEnv for Windows compatibility
        self.train_env = DummyVecEnv(env_fns)
        
        # Normalize observations and rewards
        self.train_env = VecNormalize(
            self.train_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            gamma=self.config.get("gamma", 0.99)
        )
        
        # Evaluation environment (single environment)
        eval_env_fn = [lambda: make_env(0, "stage1")]
        self.eval_env = DummyVecEnv(eval_env_fn)
        self.eval_env = VecNormalize(
            self.eval_env,
            norm_obs=True,
            norm_reward=False,  # Don't normalize rewards for evaluation
            training=False
        )
    
    def create_model(self, stage: str = "stage1"):
        """Create RL model (PPO or SAC)"""
        
        algorithm = self.config.get("algorithm", "PPO").upper()
        
        # Stage-specific hyperparameters
        if stage == "stage1":
            # Stage 1: Optimized for GPU's power
            hyperparams = {
                "learning_rate": self.config.get("stage1_lr", 3e-4),
                "n_steps": 4096,      # Increased from 2048 - more data per update
                "batch_size": 512,    # Much larger - H100 can handle this easily
                "n_epochs": 8,        # Reduced since larger batches are more stable
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "ent_coef": 0.01,
                "vf_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": 0.02
            }
        else:  # stage2
            # Stage 2: Even more aggressive for GPU
            hyperparams = {
                "learning_rate": self.config.get("stage2_lr", 1e-4),
                "n_steps": 4096,      # Increased from 2048
                "batch_size": 1024,   # Very large - GPU exclusive territory
                "n_epochs": 15,       # Reduced from 20 due to larger batches
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "clip_range": 0.15,
                "ent_coef": 0.005,
                "vf_coef": 0.5,
                "max_grad_norm": 0.3,
                "target_kl": 0.015
            }
                
        device = self.config.get("device", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu" ## GPU training 
        
        if algorithm == "PPO":
            model = PPO(
                "MlpPolicy",
                self.train_env,
                verbose=1,
                device=device,
                tensorboard_log=f"{self.experiment_dir}/tensorboard",
                **hyperparams
            )
        elif algorithm == "SAC":
            # SAC hyperparameters
            sac_params = {
                "learning_rate": hyperparams["learning_rate"],
                "buffer_size": 300000,
                "batch_size": 256,
                "gamma": hyperparams["gamma"],
                "tau": 0.005,
                "ent_coef": "auto",
                "target_entropy": "auto",
                "train_freq": 1
            }
            
            model = SAC(
                "MlpPolicy",
                self.train_env,
                verbose=1,
                device=device,
                tensorboard_log=f"{self.experiment_dir}/tensorboard",
                **sac_params
            )
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        print(f"🤖 Created {algorithm} model for {stage}")
        return model
    
    def setup_callbacks(self, stage: str):
        """Setup training callbacks"""
        callbacks = []
        
        # Wandb logging
        if self.config.get("use_wandb", False):
            callbacks.append(WandbCallback(log_freq=self.config.get("log_freq", 1000)))
        
        # Model checkpointing
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.get("save_freq", 100000),
            save_path=f"{self.experiment_dir}/checkpoints/",
            name_prefix=f"ucr_fall_recovery_{stage}"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path=f"{self.experiment_dir}/best_model/",
            log_path=f"{self.experiment_dir}/eval_logs/",
            eval_freq=self.config.get("eval_freq", 50000),
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        callbacks.append(eval_callback)
        
        # # Stop training if reward threshold reached
        # if stage == "stage1" and "stage1_reward_threshold" in self.config:
        #     stop_callback = StopTrainingOnRewardThreshold(
        #         reward_threshold=self.config["stage1_reward_threshold"],
        #         verbose=1
        #     )
        #     callbacks.append(stop_callback)
        
        # Progressive training callback
        if stage == "stage1":
            progressive_callback = ProgressiveTrainingCallback(
                stage_transition_steps=self.config.get("stage1_timesteps", 1000000)
            )
            callbacks.append(progressive_callback)
        
        return CallbackList(callbacks) if callbacks else None
    
    def train_stage1(self):
        """Train Stage 1: Get-up capability with sparse rewards"""
        print("\n🚀 Starting Stage 1 Training: Basic Get-Up Capability")
        print("=" * 60)
        
        # Initialize Wandb for stage 1
        if self.config.get("use_wandb", False):
            wandb.init(
                project="ucr-fall-recovery",
                name=f"stage1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={**self.config, "stage": 1},
                group="stage1"
            )
        
        # Create model and callbacks
        model = self.create_model("stage1")
        callbacks = self.setup_callbacks("stage1")
        
        # Train the model
        model.learn(
            total_timesteps=self.config.get("stage1_timesteps", 1000000),
            callback=callbacks,
            tb_log_name="stage1",
            reset_num_timesteps=True
        )
        
        # Save final model
        model.save(f"{self.experiment_dir}/stage1_final_model")
        self.train_env.save(f"{self.experiment_dir}/stage1_vec_normalize.pkl")
        
        print("Stage 1 training completed!")
        
        if self.config.get("use_wandb", False):
            wandb.finish()
        
        return model
    
    def train_stage2(self, pretrained_model_path: str = None):
        """Train Stage 2: Refined, deployable motion"""
        print("\n Starting Stage 2 Training: Refined Deployable Motion")
        print("=" * 60)
        
        # Update environment to stage 2
        self.train_env.close()
        self.eval_env.close()
        
        # Recreate environments for stage 2
        def make_env_stage2(rank: int = 0):
            """Create stage 2 environment"""
            set_random_seed(self.config.get("seed", 42) + rank)
            
            env = UCRFallRecoveryEnv(
                model_path=self.config["model_path"],
                max_episode_steps=self.config["max_episode_steps"],
                training_stage="stage2",  # Stage 2 environment
                randomize_terrain=self.config.get("randomize_terrain", False)
            )
            env = Monitor(env)
            env.reset(seed=self.config.get("seed", 42) + rank)
            return env
        
        # Create new environments
        env_fns = [lambda i=i: make_env_stage2(i) for i in range(self.config["n_envs"])]
        self.train_env = DummyVecEnv(env_fns)
        self.train_env = VecNormalize(
            self.train_env, norm_obs=True, norm_reward=True, 
            clip_obs=10.0, gamma=self.config.get("gamma", 0.99)
        )
        
        eval_env_fn = [lambda: make_env_stage2(0)]
        self.eval_env = DummyVecEnv(eval_env_fn)
        self.eval_env = VecNormalize(
            self.eval_env, norm_obs=True, norm_reward=False, training=False
        )
        
        # Initialize Wandb for stage 2
        if self.config.get("use_wandb", False):
            wandb.init(
                project="ucr-fall-recovery",
                name=f"stage2_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={**self.config, "stage": 2},
                group="stage2"
            )
        
        # Create stage 2 model
        model = self.create_model("stage2")
        
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            print(f" Loading pretrained model from: {pretrained_model_path}")
            # Note: Loading pretrained models requires careful handling
            # For now, we start fresh but you could implement model loading here
        
        callbacks = self.setup_callbacks("stage2")
        
        # Train stage 2
        model.learn(
            total_timesteps=self.config.get("stage2_timesteps", 2000000),
            callback=callbacks,
            tb_log_name="stage2",
            reset_num_timesteps=True
        )
        
        # Save final model
        model.save(f"{self.experiment_dir}/stage2_final_model")
        self.train_env.save(f"{self.experiment_dir}/stage2_vec_normalize.pkl")
        
        print("Stage 2 training completed!")
        
        if self.config.get("use_wandb", False):
            wandb.finish()
        
        return model
    
    def evaluate_model(self, model_path: str, n_episodes: int = 50, render: bool = False):
        """Evaluate trained model"""
        print(f"\n Evaluating model: {model_path}")
        
        # Load model
        algorithm = self.config.get("algorithm", "PPO").upper()
        try:
            if algorithm == "PPO":
                model = PPO.load(model_path)
            elif algorithm == "SAC":
                model = SAC.load(model_path)
        except Exception as e:
            print(f" Error loading model: {e}")
            return {"success_rate": 0.0, "average_reward": 0.0, "average_recovery_time": float('inf'), "total_episodes": 0}
        
        # Create evaluation environment
        eval_env = UCRFallRecoveryEnv(
            model_path=self.config["model_path"],
            max_episode_steps=self.config["max_episode_steps"],
            training_stage="stage2",
            render_mode="human" if render else None
        )
        
        # Evaluation metrics
        success_count = 0
        total_rewards = []
        episode_lengths = []
        recovery_times = []
        
        for episode in range(n_episodes):
            obs, info = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            recovered = False
            recovery_time = None
            
            for step in range(self.config["max_episode_steps"]):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check if robot successfully recovered
                if info.get("is_upright", False) and not recovered:
                    recovery_time = step * 0.02  # Assuming 50Hz control
                    recovered = True
                    success_count += 1
                
                if render:
                    eval_env.render()
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            if recovery_time:
                recovery_times.append(recovery_time)
            
            success_str = "✅" if recovered else "❌"
            time_str = f"{recovery_time:.2f}s" if recovery_time else "Failed"
            print(f"Episode {episode + 1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Success={success_str}, "
                  f"Time={time_str}")
        
        # Calculate metrics
        success_rate = success_count / n_episodes
        avg_reward = np.mean(total_rewards)
        avg_recovery_time = np.mean(recovery_times) if recovery_times else float('inf')
        
        print(f"\n Evaluation Results:")
        print(f"Success Rate: {success_rate:.2%} ({success_count}/{n_episodes})")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Recovery Time: {avg_recovery_time:.2f}s")
        print(f"Average Episode Length: {np.mean(episode_lengths):.1f} steps")
        
        eval_env.close()
        
        return {
            "success_rate": success_rate,
            "average_reward": avg_reward,
            "average_recovery_time": avg_recovery_time,
            "total_episodes": n_episodes
        }
    
    def cleanup(self):
        """Clean up environments"""
        if hasattr(self, 'train_env'):
            self.train_env.close()
        if hasattr(self, 'eval_env'):
            self.eval_env.close()

def main():
    """Main training function optimized for UCR v0H"""
    
    print(" UCR v0H Humanoid Fall Recovery Training")
    print("=" * 60)
    
    # Get UCR v0H optimized configuration
    stage1_config = UCRv0HConfig.get_training_config("stage1")
    
    # Override with any custom settings
    custom_overrides = {
        "n_envs": 12,  # Changed for runpod run - usually set to 4 for CPU
        "use_wandb": True,  # Enable for experiment tracking
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    
    config = {**stage1_config, **custom_overrides}
    
    print(f"  Device: {config['device']}")
    print(f" Parallel Environments: {config['n_envs']}")
    print(f" Wandb Logging: {config['use_wandb']}")
    
    # Validate robot model exists
    if not os.path.exists(config['model_path']):
        print(f" Error: UCR v0H model not found at {config['model_path']}")
        print("   Make sure you have the robot model files:")
        print("   - model/mjcf/xml/v0H_pos.xml")
        print("   - model/mjcf/xml/v0H_base.xml")
        return
    
    # Initialize trainer
    trainer = UCRTrainer(config)
    
    try:
        # Stage 1: Basic get-up capability
        print(f"\n Starting Stage 1: Basic Get-Up Capability")
        print(f" Target: {config.get('stage1_reward_threshold', 200)} reward threshold")
        print(f"  Duration: {config['stage1_timesteps']:,} timesteps")
        
        stage1_model = trainer.train_stage1()
        
        # Evaluate stage 1
        print(f"\n🧪 Evaluating Stage 1 Performance...")
        stage1_results = trainer.evaluate_model(
            f"{trainer.experiment_dir}/stage1_final_model.zip",
            n_episodes=10,  # Reduced for faster testing
            render=False
        )
        
        print(f"\n Stage 1 Results:")
        print(f"   Success Rate: {stage1_results['success_rate']:.1%}")
        print(f"   Average Reward: {stage1_results['average_reward']:.1f}")
        print(f"   Recovery Time: {stage1_results['average_recovery_time']:.2f}s")
        
        # Training completed successfully
        print(f"\n Training completed successfully!")
        print(f" Results saved in: {trainer.experiment_dir}")
            
    except KeyboardInterrupt:
        print(f"\n Training interrupted by user")
        
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print(f"\n Cleaning up resources...")
        trainer.cleanup()
        print(f"Cleanup completed")

if __name__ == "__main__":
    main()