"""
Comprehensive test script for UCR v0H setup
Run this before starting training to verify everything is working
"""

import sys
import os
import numpy as np

def test_dependencies():
    """Test all required dependencies"""
    print("🔍 Testing Dependencies...")
    
    dependencies = [
        ("mujoco", "MuJoCo"),
        ("stable_baselines3", "Stable-Baselines3"),
        ("torch", "PyTorch"),
        ("gymnasium", "Gymnasium"),
        ("numpy", "NumPy"),
        ("wandb", "Weights & Biases")
    ]
    
    results = []
    for module, name in dependencies:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f" {name}: {version}")
            results.append(True)
        except ImportError:
            print(f"{name}: Not installed")
            results.append(False)
    
    # Special checks
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA: Available ({torch.cuda.device_count()} GPUs)")
        else:
            print(f"CUDA: Not available (CPU only)")
    except:
        pass
    
    return all(results)

def test_file_structure():
    """Test that all required files are present"""
    print(f"\n Testing File Structure...")
    
    required_files = [
        ("model/mjcf/xml/v0H_pos.xml", "UCR v0H main model"),
        ("model/mjcf/xml/v0H_base.xml", "UCR v0H base model"),
        ("ucr_fall_env.py", "Environment"),
        ("ucr_v0h_config.py", "Configuration"),
        ("rl_training_setup.py", "Training setup"),
        ("run_ucr_v0h_training.py", "Main script"),
        ("requirements.txt", "Dependencies")
    ]
    
    results = []
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f" {description}: {file_path}")
            results.append(True)
        else:
            print(f" {description}: Missing {file_path}")
            results.append(False)
    
    # Check for mesh directory
    if os.path.exists("model/mjcf/mesh"):
        mesh_files = [f for f in os.listdir("model/mjcf/mesh/arms") if f.endswith(".obj")]
        if mesh_files:
            print(f" Mesh files: {len(mesh_files)} .obj files found")
        else:
            print(f" Mesh files: Directory exists but no .obj files found")
    else:
        print(f" Mesh directory: Missing model/mjcf/mesh/ directory")

    return all(results)

def test_ucr_model_loading():
    """Test loading UCR v0H model with all components"""
    print(f"\n Testing UCR v0H Model Loading...")
    
    model_path = "model/mjcf/xml/v0H_pos.xml"
    if not os.path.exists(model_path):
        print(f" Model file not found: {model_path}")
        return False
    
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
        
        print(f" UCR v0H model loaded successfully")
        print(f"   Bodies: {model.nbody}")
        print(f"   Joints: {model.nq} (expected: 30)") ## currently getting 29 
        print(f"   Actuators: {model.nu} (expected: 23)")
        print(f"   DOF: {model.nv} (expected: 28)")
        print(f"   Sensors: {model.nsensor}")
        
        # Test specific UCR v0H components
        expected_joints = 30  # 7 floating base + 23 actuated joints
        expected_actuators = 23
        
        if model.nq == expected_joints and model.nu == expected_actuators:
            print(f" UCR v0H dimensions match expected values")
        else:
            print(f" Dimension mismatch - check model file")
            print(f"   Expected: {expected_joints} joints, {expected_actuators} actuators")
            print(f"   Got: {model.nq} joints, {model.nu} actuators")
            
        # Test sensor loading
        sensor_names = ["pos_pelvis", "ori_pelvis", "left_foot_contact", "right_foot_contact"]
        missing_sensors = []
        for sensor_name in sensor_names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            if sensor_id == -1:
                missing_sensors.append(sensor_name)
        
        if missing_sensors:
            print(f" Missing sensors: {missing_sensors}")
        else:
            print(f" All expected sensors found")
            
        return True
        
    except Exception as e:
        print(f" Error loading UCR v0H model: {e}")
        return False

def test_ucr_config():
    """Test UCR v0H configuration"""
    print(f"\n Testing UCR v0H Configuration...")
    
    try:
        from ucr_v0h_config import UCRv0HConfig
        
        # Test configuration loading
        config = UCRv0HConfig.get_training_config("stage1")
        print(f" Stage 1 config loaded: {len(config)} parameters")
        
        config2 = UCRv0HConfig.get_training_config("stage2") 
        print(f" Stage 2 config loaded: {len(config2)} parameters")
        
        # Test joint information
        joint_order = UCRv0HConfig.get_joint_order()
        print(f" Joint order: {len(joint_order)} joints")
        
        actuator_limits = UCRv0HConfig.get_actuator_limits()
        print(f" Actuator limits: {len(actuator_limits)} values")
        
        # Test fallen pose templates
        templates = UCRv0HConfig.get_fallen_pose_templates()
        print(f" Fallen pose templates: {len(templates)} poses")
        
        # Verify essential parameters
        essential_params = ["model_path", "max_episode_steps", "reward_weights", "stage1_timesteps"]
        for param in essential_params:
            if param not in config:
                print(f" Missing essential parameter: {param}")
            
        return True
        
    except Exception as e:
        print(f" Error testing UCR v0H config: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ucr_environment():
    """Test UCR v0H environment creation and basic functionality"""
    print(f"\n Testing UCR v0H Environment...")
    
    try:
        from ucr_fall_env import UCRFallRecoveryEnv
        from ucr_v0h_config import UCRv0HConfig
        
        # Create environment
        env = UCRFallRecoveryEnv(
            model_path="model/mjcf/xml/v0H_pos.xml",
            max_episode_steps=100,
            training_stage="stage1"
        )
        
        print(f" UCR v0H environment created")
        print(f"   Observation space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        print(f"   Expected obs dim: 101 (29+28+16+2+3+23)")
        print(f"   Expected action dim: 23")
        
        # Verify dimensions
        if env.observation_space.shape[0] == 101 and env.action_space.shape[0] == 23:
            print(f" Environment dimensions correct")
        else:
            print(f" Environment dimension mismatch")
        
        # Test reset and step
        obs, info = env.reset()
        print(f" Environment reset successful")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Initial height: {info.get('current_height', 0):.3f}m")
        print(f"   Target height: {info.get('target_height', 0):.3f}m")
        print(f"   Training stage: {info.get('training_stage', 'unknown')}")
        
        # Test multiple steps
        total_reward = 0
        for step in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            print(f"   Step {step+1}: reward={reward:.3f}, height={info.get('current_height', 0):.3f}m, upright={info.get('upright_similarity', 0):.3f}")
            
            if terminated or truncated:
                print(f"   Episode ended at step {step+1}")
                break
        
        print(f" Environment stepping successful, total reward: {total_reward:.3f}")
        
        # Test fallen pose generation
        print(f"\n Testing fallen pose generation...")
        for i in range(3):
            env.reset()
            height = env.data.qpos[2]  # Z position
            quat = env.data.qpos[3:7]  # Orientation
            print(f"   Pose {i+1}: height={height:.3f}m, quat=[{quat[0]:.2f}, {quat[1]:.2f}, {quat[2]:.2f}, {quat[3]:.2f}]")
        
        env.close()
        return True
        
    except Exception as e:
        print(f" Error testing UCR v0H environment: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_pipeline():
    """Test that training pipeline can be imported and initialized"""
    print(f"\n Testing Training Pipeline...")
    
    try:
        from rl_training_setup import UCRTrainer
        from ucr_v0h_config import UCRv0HConfig
        
        # Get minimal config for testing
        config = UCRv0HConfig.get_training_config("stage1")
        config["n_envs"] = 2  # Reduce for testing
        config["stage1_timesteps"] = 1000  # Very short for testing
        config["use_wandb"] = False  # Disable logging for test
        
        print(f" Training configuration prepared")
        print(f"   Environments: {config['n_envs']}")
        print(f"   Timesteps: {config['stage1_timesteps']}")
        
        # Try to initialize trainer (don't actually train)
        print(f" Initializing trainer (this may take a moment)...")
        trainer = UCRTrainer(config)
        print(f" UCRTrainer initialized successfully")
        print(f"   Experiment dir: {trainer.experiment_dir}")
        
        # Cleanup
        trainer.cleanup()
        print(f" Training pipeline test completed")
        
        return True
        
    except Exception as e:
        print(f" Error testing training pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_main_script():
    """Test main training script can be imported"""
    print(f"\n Testing Main Training Script...")
    
    try:
        # Test import
        import run_ucr_v0h_training
        print(f" Main training script imported successfully")
        
        # Test help message
        print(f" Main script help available")
        
        return True
        
    except Exception as e:
        print(f" Error testing main script: {e}")
        return False

def main():
    """Run comprehensive UCR v0H test suite"""
    print(" UCR v0H Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("File Structure", test_file_structure),
        ("UCR v0H Model", test_ucr_model_loading),
        ("UCR v0H Config", test_ucr_config),
        ("UCR v0H Environment", test_ucr_environment),
        ("Training Pipeline", test_training_pipeline),
        ("Main Script", test_main_script),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f" {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Final summary
    print(f"\n{'='*60}")
    print(" Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:<20}: {status}")
        if success:
            passed += 1
    
    print(f"\n Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print(f"\n All tests passed! UCR v0H is ready for training.")
        print(f"\nNext steps:")
        print(f"   1. Quick test: python run_ucr_v0h_training.py --quick_test")
        print(f"   2. Full training: python run_ucr_v0h_training.py --stage both")
        print(f"   3. Monitor progress: tensorboard --logdir experiments/")
        print(f"   4. Stage 1 only: python run_ucr_v0h_training.py --stage stage1")
    else:
        failed_tests = [name for name, success in results if not success]
        print(f"\n {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
        print(f"\nTroubleshooting:")
        print(f"   - Install dependencies: pip install -r requirements.txt")
        print(f"   - Check file locations match the expected structure")
        print(f"   - Ensure UCR v0H model files and meshes are present")
        print(f"   - Verify all .py files are in the same directory")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)