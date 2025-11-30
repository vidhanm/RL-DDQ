"""
Training Script for Debt Collection Agent
Supports both DQN (baseline) and DDQ (with world model)
With Weights & Biases (wandb) integration for experiment tracking
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from environment.debtor_env import DebtCollectionEnv
from agent.dqn_agent import DQNAgent
from agent.ddq_agent import DDQAgent
from llm.nvidia_client import NVIDIAClient
from config import (
    TrainingConfig,
    RLConfig,
    EnvironmentConfig,
    DeviceConfig,
    DDQConfig,
    print_config
)

# Neptune.ai for experiment tracking
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("[WARN] Neptune not installed. Run: pip install neptune")

# Load environment variables for Neptune API token
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[WARN] python-dotenv not installed. Run: pip install python-dotenv")


def train_agent(
    algorithm: str = "dqn",
    num_episodes: int = TrainingConfig.NUM_EPISODES_DQN,
    use_llm: bool = True,
    render: bool = False,
    save_dir: str = TrainingConfig.CHECKPOINT_DIR,
    use_neptune: bool = False,
    neptune_project: str = "ddq-debt-collection"
):
    """
    Train agent (DQN or DDQ)

    Args:
        algorithm: "dqn" or "ddq"
        num_episodes: Number of episodes to train
        use_llm: Whether to use LLM for conversations
        render: Whether to render conversations
        save_dir: Directory to save checkpoints
        use_neptune: Whether to use Neptune.ai for tracking
        neptune_project: Neptune project name (format: workspace/project-name)
    """
    print("="*70)
    print(f"{algorithm.upper()} TRAINING - Debt Collection Agent")
    print("="*70)
    print_config()

    # Initialize Neptune.ai
    neptune_run = None
    if use_neptune and NEPTUNE_AVAILABLE:
        try:
            neptune_run = neptune.init_run(
                project=neptune_project,
                name=f"{algorithm}-{num_episodes}ep",
                tags=[algorithm, f"{num_episodes}ep", "debt-collection"],
            )

            # Log hyperparameters
            neptune_run["parameters"] = {
                "algorithm": algorithm,
                "num_episodes": num_episodes,
                "learning_rate": RLConfig.LEARNING_RATE,
                "gamma": RLConfig.GAMMA,
                "epsilon_start": RLConfig.EPSILON_START,
                "epsilon_end": RLConfig.EPSILON_END,
                "epsilon_decay": RLConfig.EPSILON_DECAY,
                "batch_size": RLConfig.BATCH_SIZE,
                "buffer_size": RLConfig.REPLAY_BUFFER_SIZE,
                "K": DDQConfig.K if algorithm == "ddq" else 0,
                "real_ratio": DDQConfig.REAL_RATIO if algorithm == "ddq" else 1.0,
                "state_dim": EnvironmentConfig.STATE_DIM,
                "action_dim": EnvironmentConfig.NUM_ACTIONS,
                "max_turns": EnvironmentConfig.MAX_TURNS,
                "use_llm": use_llm
            }

            print(f"\n[OK] Neptune.ai initialized")
            print(f"     View at: {neptune_run.get_url()}")
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize Neptune: {e}")
            print("  Make sure NEPTUNE_API_TOKEN is set in .env file")
            use_neptune = False
    elif use_neptune and not NEPTUNE_AVAILABLE:
        print("\n[WARN] Neptune requested but not installed. Install: pip install neptune")
        use_neptune = False

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize LLM client (NVIDIA by default)
    llm_client = None
    if use_llm:
        try:
            llm_client = NVIDIAClient()
            print(f"\n[OK] LLM client initialized (model: {llm_client.model})")
        except Exception as e:
            print(f"\n[ERROR] Failed to initialize LLM: {e}")
            print("  Continuing without LLM (using placeholders)")
            use_llm = False

    # Create environment
    render_mode = "human" if render else None
    env = DebtCollectionEnv(llm_client=llm_client, render_mode=render_mode)
    print(f"[OK] Environment created")

    # Create agent based on algorithm
    if algorithm.lower() == "ddq":
        agent = DDQAgent(
            state_dim=EnvironmentConfig.STATE_DIM,
            action_dim=EnvironmentConfig.NUM_ACTIONS,
            K=DDQConfig.K,
            device=DeviceConfig.DEVICE
        )
        print(f"[OK] DDQ agent created (device: {DeviceConfig.DEVICE}, K={DDQConfig.K})")
    else:
        agent = DQNAgent(
            state_dim=EnvironmentConfig.STATE_DIM,
            action_dim=EnvironmentConfig.NUM_ACTIONS,
            device=DeviceConfig.DEVICE
        )
        print(f"[OK] DQN agent created (device: {DeviceConfig.DEVICE})")

    # Training statistics
    episode_rewards = []
    episode_lengths = []
    success_history = []
    loss_history = []

    print(f"\n{'='*70}")
    print(f"Starting training for {num_episodes} episodes...")
    print(f"{'='*70}\n")

    # Training loop
    for episode in tqdm(range(1, num_episodes + 1), desc="Training"):

        # Reset environment
        state, info = env.reset()
        episode_reward = 0.0
        episode_loss = []
        done = False
        turn = 0

        # Episode loop
        while not done:
            # Select action
            action = agent.select_action(state, explore=True)

            # Take step
            next_state, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train
            if agent.get_buffer_size() >= RLConfig.MIN_BUFFER_SIZE:
                loss = agent.train_step()
                episode_loss.append(loss)

            # Update state
            state = next_state
            episode_reward += reward
            turn += 1

        # Episode done
        agent.episode_done(episode)

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(turn)
        success_history.append(1 if step_info['has_committed'] else 0)
        if episode_loss:
            loss_history.append(np.mean(episode_loss))

        # Log to Neptune.ai
        if use_neptune and neptune_run:
            # Log main metrics
            neptune_run["episode/reward"].append(episode_reward)
            neptune_run["episode/length"].append(turn)
            neptune_run["episode/success"].append(1 if step_info['has_committed'] else 0)
            neptune_run["agent/epsilon"].append(agent.epsilon)
            neptune_run["agent/buffer_size"].append(agent.get_buffer_size())
            neptune_run["debtor/final_sentiment"].append(step_info.get('final_sentiment', 0))
            neptune_run["debtor/final_cooperation"].append(step_info.get('final_cooperation', 0))

            if episode_loss:
                neptune_run["training/loss"].append(np.mean(episode_loss))

            # Rolling averages
            if len(episode_rewards) >= 10:
                neptune_run["episode/reward_ma10"].append(np.mean(episode_rewards[-10:]))
                neptune_run["episode/success_rate_ma10"].append(np.mean(success_history[-10:]))
            if len(episode_rewards) >= 50:
                neptune_run["episode/reward_ma50"].append(np.mean(episode_rewards[-50:]))
                neptune_run["episode/success_rate_ma50"].append(np.mean(success_history[-50:]))

        # Log progress
        if episode % TrainingConfig.LOG_FREQ == 0:
            recent_rewards = episode_rewards[-10:]
            recent_success = success_history[-10:]
            recent_loss = loss_history[-10:] if loss_history else [0]

            tqdm.write(f"\nEpisode {episode}/{num_episodes}")
            tqdm.write(f"  Avg Reward (last 10): {np.mean(recent_rewards):.2f}")
            tqdm.write(f"  Success Rate (last 10): {np.mean(recent_success):.1%}")
            tqdm.write(f"  Avg Loss: {np.mean(recent_loss):.4f}")
            tqdm.write(f"  Epsilon: {agent.epsilon:.4f}")
            tqdm.write(f"  Buffer: {agent.get_buffer_size()}/{RLConfig.REPLAY_BUFFER_SIZE}")

        # Save checkpoint
        if episode % TrainingConfig.SAVE_FREQ == 0:
            checkpoint_path = os.path.join(save_dir, f"{algorithm}_episode_{episode}.pt")
            agent.save(checkpoint_path)
            tqdm.write(f"  -> Checkpoint saved: {checkpoint_path}")

        # Evaluate
        if episode % TrainingConfig.EVAL_FREQ == 0:
            eval_success_rate = evaluate_agent(agent, env, num_episodes=5)
            tqdm.write(f"  -> Evaluation Success Rate: {eval_success_rate:.1%}")

            if use_neptune and neptune_run:
                neptune_run["evaluation/success_rate"].append(eval_success_rate)

    # Training complete
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")

    # Final statistics
    print(f"\nFinal Statistics:")
    print(f"  Total Episodes: {num_episodes}")
    print(f"  Avg Reward: {np.mean(episode_rewards):.2f}")
    print(f"  Overall Success Rate: {np.mean(success_history):.1%}")
    print(f"  Final Epsilon: {agent.epsilon:.4f}")

    # Save final model
    final_path = os.path.join(save_dir, f"{algorithm}_final.pt")
    agent.save(final_path)

    # Print agent stats
    agent.print_statistics()

    # Print LLM stats if used
    if use_llm and llm_client:
        llm_client.print_statistics()

    # Build hyperparameters dict for saving
    hyperparams = {
        "learning_rate": RLConfig.LEARNING_RATE,
        "gamma": RLConfig.GAMMA,
        "epsilon_start": RLConfig.EPSILON_START,
        "epsilon_end": RLConfig.EPSILON_END,
        "epsilon_decay": RLConfig.EPSILON_DECAY,
        "batch_size": RLConfig.BATCH_SIZE,
        "buffer_size": RLConfig.REPLAY_BUFFER_SIZE,
        "state_dim": EnvironmentConfig.STATE_DIM,
        "action_dim": EnvironmentConfig.NUM_ACTIONS,
        "max_turns": EnvironmentConfig.MAX_TURNS,
        "use_llm": use_llm
    }
    
    # Add DDQ-specific params if applicable
    if algorithm.lower() == "ddq":
        hyperparams["K"] = DDQConfig.K
        hyperparams["real_ratio"] = DDQConfig.REAL_RATIO
        hyperparams["world_model_lr"] = DDQConfig.WORLD_MODEL_LR

    # Save training history with metadata
    history_path = save_training_history(
        episode_rewards,
        episode_lengths,
        success_history,
        loss_history,
        save_dir,
        algorithm=algorithm,
        num_episodes=num_episodes,
        hyperparams=hyperparams
    )

    # Save to Neptune.ai and close
    if use_neptune and neptune_run:
        # Log final summary
        neptune_run["summary/final_avg_reward"] = np.mean(episode_rewards)
        neptune_run["summary/final_success_rate"] = np.mean(success_history)
        neptune_run["summary/final_epsilon"] = agent.epsilon

        # Upload model file
        neptune_run["model/final"].upload(final_path)

        # Stop Neptune run
        neptune_run.stop()
        print(f"\n[OK] Neptune.ai logs saved")
        print(f"     View your experiment at: {neptune_run.get_url()}")

    return agent, episode_rewards, success_history


def evaluate_agent(
    agent: DQNAgent,
    env: DebtCollectionEnv,
    num_episodes: int = 10
) -> float:
    """
    Evaluate agent performance

    Args:
        agent: DQN agent
        env: Environment
        num_episodes: Number of evaluation episodes

    Returns:
        Success rate
    """
    successes = []

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.select_action(state, explore=False)  # No exploration
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state

        successes.append(1 if info['has_committed'] else 0)

    return np.mean(successes)


def save_training_history(
    rewards, lengths, successes, losses, save_dir,
    algorithm: str = "unknown", num_episodes: int = 0, hyperparams: dict = None
):
    """Save training history to file with unique naming"""
    import json
    from datetime import datetime

    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build history with metadata
    history = {
        'metadata': {
            'algorithm': algorithm,
            'num_episodes': num_episodes,
            'timestamp': timestamp,
            'hyperparameters': hyperparams or {}
        },
        'episode_rewards': rewards,
        'episode_lengths': lengths,
        'success_history': successes,
        'loss_history': losses
    }

    # Save with unique filename: {algorithm}_{episodes}ep_{timestamp}.json
    filename = f"{algorithm}_{num_episodes}ep_{timestamp}.json"
    history_path = os.path.join(save_dir, filename)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n[OK] Training history saved: {history_path}")
    
    return history_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Debt Collection Agent (DQN or DDQ)")
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'ddq'],
                        help='Algorithm to use: dqn (baseline) or ddq (with world model)')
    parser.add_argument('--episodes', type=int, default=TrainingConfig.NUM_EPISODES_DQN,
                        help='Number of training episodes')
    parser.add_argument('--no-llm', action='store_true',
                        help='Train without LLM (faster, for testing)')
    parser.add_argument('--render', action='store_true',
                        help='Render conversations during training')
    parser.add_argument('--save-dir', type=str, default=TrainingConfig.CHECKPOINT_DIR,
                        help='Directory to save checkpoints')
    parser.add_argument('--neptune', action='store_true',
                        help='Enable Neptune.ai experiment tracking')
    parser.add_argument('--neptune-project', type=str, default='ddq-debt-collection',
                        help='Neptune project name (format: workspace/project-name)')

    args = parser.parse_args()

    # Train
    train_agent(
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        use_llm=not args.no_llm,
        render=args.render,
        save_dir=args.save_dir,
        use_neptune=args.neptune,
        neptune_project=args.neptune_project
    )


if __name__ == "__main__":
    main()
