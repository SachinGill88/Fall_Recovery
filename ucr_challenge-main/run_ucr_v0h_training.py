"""
UCR v0H specific training script with optimized parameters
Main entry point for training the fall recovery policy - FIXED VERSION
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Train UCR v0H Fall Recovery Policy")
    parser.add_argument("--stage", type=str, default="both",
                       choices=["stage1", "stage2", "both"],
                       help="Which training stage to run")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Override default training timesteps")
    parser.add_argument("--n_envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--no_wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--render", action="store_true",
                       help="Render during evaluation")
    parser.add_argument("--eval_only", type=str, default=None,
                       help="Path to model for evaluation only")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode with verbose output")
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test run with reduced timesteps")
    parser.add_argument("--algorithm", type=str, default="PPO",
                       choices=["PPO", "SAC"],
                       help="RL algorithm to use")
    
    args = parser.parse_args()
    
    print(" UCR v0H Fall Recovery Training")
    print("=" * 50)
    
    # Verify model files exist
    model_files = [
        "model/mjcf/xml/v0H_pos.xml",
        "model/mjcf/xml/v0H_base.xml"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Error: Missing UCR v0H model files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        print("\nMake sure you have the complete robot model files in the correct location.")
        print("Also check that mesh files are in the mesh/ directory.")
        sys.exit(1)
    
    print("✅ UCR v0H model files found")
    
    # Import training modules (after file verification)
    try:
        from rl_training_setup import UCRTrainer
        from ucr_v0h_config import UCRv0HConfig
        import torch
    except ImportError as e:
        print(f" Import Error: {e}")
        print("Make sure you've installed all requirements:")
        print("pip install -r requirements.txt")
        print("\nAlso ensure all Python files are in the same directory:")
        print("- ucr_fall_env.py")
        print("- ucr_v0h_config.py") 
        print("- rl_training_setup.py")
        sys.exit(1)
    
    # Get UCR v0H specific configuration
    if args.stage == "stage2":
        config = UCRv0HConfig.get_training_config("stage2")
    else:
        config = UCRv0HConfig.get_training_config("stage1")
    
    # Apply command line overrides
    config["n_envs"] = args.n_envs
    config["use_wandb"] = not args.no_wandb
    config["algorithm"] = args.algorithm
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Override timesteps if specified
    if args.timesteps:
        if args.stage == "stage1" or args.stage == "both":
            config["stage1_timesteps"] = args.timesteps
        if args.stage == "stage2" or args.stage == "both":
            config["stage2_timesteps"] = args.timesteps
    
    # Apply quick test settings AFTER config is fully loaded
    if args.quick_test:
        print("⚡ Quick test mode enabled")
        config["stage1_timesteps"] = min(config.get("stage1_timesteps", 1000000), 100000)
        config["stage2_timesteps"] = min(config.get("stage2_timesteps", 2000000), 100000)
        config["max_episode_steps"] = 200  # Shorter episodes
        config["eval_freq"] = 10000  # More frequent evaluation
        config["save_freq"] = 25000  # More frequent saving
        config["n_envs"] = min(config["n_envs"], 4)  # Fewer environments
    
    print(f"  Device: {config['device']}")
    print(f"Parallel Environments: {config['n_envs']}")
    print(f"Wandb Logging: {'Enabled' if config['use_wandb'] else 'Disabled'}")
    print(f"Training Stage: {args.stage}")
    print(f" Algorithm: {config['algorithm']}")
    
    if args.debug:
        print(f" Debug mode enabled")
        print(f" Full Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"     {k}: {v}")
            else:
                print(f"   {key}: {value}")
    
    # Initialize trainer
    try:
        trainer = UCRTrainer(config)
    except Exception as e:
        print(f" Failed to initialize trainer: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    try:
        if args.eval_only:
            # Evaluation only mode
            print(f"\n Evaluating UCR v0H model: {args.eval_only}")
            
            if not os.path.exists(args.eval_only):
                print(f" Error: Model file not found: {args.eval_only}")
                return
            
            results = trainer.evaluate_model(
                args.eval_only, 
                n_episodes=30,  # More episodes for UCR v0H
                render=args.render
            )
            
            print(f"\n UCR v0H Evaluation Results:")
            print(f"   Success Rate: {results['success_rate']:.1%}")
            print(f"   Average Recovery Time: {results['average_recovery_time']:.2f}s")
            print(f"   Average Reward: {results['average_reward']:.1f}")
            
            # Performance rating
            if results['success_rate'] >= 0.8:
                print(f"🏆 Rating: EXCELLENT")
            elif results['success_rate'] >= 0.6:
                print(f" Rating: GOOD")
            elif results['success_rate'] >= 0.4:
                print(f" Rating: FAIR")
            else:
                print(f" Rating: POOR")
            
        elif args.stage == "stage1":
            # Train stage 1 only
            print(f"\n Training Stage 1 Only")
            print(f"  Duration: {config['stage1_timesteps']:,} timesteps")
            
            model = trainer.train_stage1()
            results = trainer.evaluate_model(
                f"{trainer.experiment_dir}/stage1_final_model.zip",
                n_episodes=20,
                render=args.render
            )
            
            print(f"\n Stage 1 Final Results:")
            print(f"   Success Rate: {results['success_rate']:.1%}")
            print(f"   Model saved: {trainer.experiment_dir}/stage1_final_model.zip")
            
        elif args.stage == "stage2":
            # Train stage 2 only (requires pretrained stage 1)
            print(f"\n Training Stage 2 Only")
            print(f" Note: This requires a pretrained Stage 1 model")
            print(f"  Duration: {config['stage2_timesteps']:,} timesteps")
            
            model = trainer.train_stage2()
            results = trainer.evaluate_model(
                f"{trainer.experiment_dir}/stage2_final_model.zip",
                n_episodes=30,
                render=args.render
            )
            
            print(f"\n📊 Stage 2 Final Results:")
            print(f"   Success Rate: {results['success_rate']:.1%}")
            print(f"   Model saved: {trainer.experiment_dir}/stage2_final_model.zip")
            
        else:  # both stages
            # Full two-stage training
            print(f"\n Full Two-Stage Training")
            print(f" Stage 1: {config['stage1_timesteps']:,} timesteps")
            print(f" Stage 2: {config['stage2_timesteps']:,} timesteps")
            total_time_hours = (config['stage1_timesteps'] + config['stage2_timesteps']) / 1000000 * 6
            print(f"⏱️  Estimated Total Time: ~{total_time_hours:.1f} hours")
            
            # Run the main training function
            from rl_training_setup import main as training_main
            training_main()
            
    except KeyboardInterrupt:
        print("\n Training interrupted by user")
        print(" Partial results may be saved in the experiment directory")
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        print(f" Try running with --debug for more detailed error information")
    finally:
        if 'trainer' in locals():
            trainer.cleanup()

if __name__ == "__main__":
    main()