"""
Evaluation Script
Evaluate DDQ/DQN checkpoints with NLU environment
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.nlu_env import NLUDebtCollectionEnv
from src.agent.dqn_agent import DQNAgent
from src.agent.ddq_agent import DDQAgent
from src.llm.nvidia_client import NVIDIAClient
from src.config import EnvironmentConfig, DeviceConfig



def load_training_history(filepath: str) -> Dict:
    """Load training history from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def evaluate_checkpoint(
    checkpoint_path: str,
    algorithm: str,
    env: NLUDebtCollectionEnv,
    num_episodes: int = 20
) -> Dict:
    """
    Evaluate a trained agent

    Args:
        checkpoint_path: Path to checkpoint file
        algorithm: "dqn" or "ddq"
        env: NLU Environment
        num_episodes: Number of evaluation episodes

    Returns:
        Evaluation metrics
    """
    # Create agent with NLU state dimension
    state_dim = EnvironmentConfig.NLU_STATE_DIM  # 19 dimensions
    
    if algorithm.lower() == "ddq":
        agent = DDQAgent(
            state_dim=state_dim,
            action_dim=EnvironmentConfig.NUM_ACTIONS,
            device=DeviceConfig.DEVICE
        )
    else:
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=EnvironmentConfig.NUM_ACTIONS,
            device=DeviceConfig.DEVICE
        )

    # Load checkpoint
    agent.load(checkpoint_path)

    # Evaluate
    successes = []
    rewards = []
    lengths = []
    conversations = []

    for ep in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        done = False
        turn = 0
        conversation = []

        while not done:
            action = agent.select_action(state, explore=False)  # No exploration
            next_state, reward, terminated, truncated, step_info = env.step(action)

            episode_reward += reward
            turn += 1
            done = terminated or truncated
            state = next_state

            # Record conversation (if available)
            if hasattr(env, 'conversation_history') and len(env.conversation_history) > 0:
                conversation = env.conversation_history

        successes.append(1 if step_info['has_committed'] else 0)
        rewards.append(episode_reward)
        lengths.append(turn)
        if conversation:
            conversations.append(conversation)

    return {
        'success_rate': np.mean(successes),
        'avg_reward': np.mean(rewards),
        'avg_length': np.mean(lengths),
        'successes': successes,
        'rewards': rewards,
        'lengths': lengths,
        'conversations': conversations[:3]  # Save first 3 conversations
    }


def plot_comparison(dqn_history: Dict, ddq_history: Dict, save_path: str = None):
    """
    Plot DQN vs DDQ comparison

    Args:
        dqn_history: DQN training history
        ddq_history: DDQ training history
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Smooth data
    def smooth(data, window=10):
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed

    # Plot 1: Rewards
    ax = axes[0, 0]
    dqn_rewards = smooth(dqn_history['episode_rewards'])
    ddq_rewards = smooth(ddq_history['episode_rewards'])
    ax.plot(dqn_rewards, label='DQN', alpha=0.7)
    ax.plot(ddq_rewards, label='DDQ', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Average Reward per Episode (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Success Rate
    ax = axes[0, 1]
    dqn_success = smooth(dqn_history['success_history'])
    ddq_success = smooth(ddq_history['success_history'])
    ax.plot(dqn_success, label='DQN', alpha=0.7)
    ax.plot(ddq_success, label='DDQ', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Episode Length
    ax = axes[1, 0]
    dqn_lengths = smooth(dqn_history['episode_lengths'])
    ddq_lengths = smooth(ddq_history['episode_lengths'])
    ax.plot(dqn_lengths, label='DQN', alpha=0.7)
    ax.plot(ddq_lengths, label='DDQ', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Length (turns)')
    ax.set_title('Average Episode Length (smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Sample Efficiency (Success rate vs episodes)
    ax = axes[1, 1]

    # Calculate cumulative success rate
    dqn_cum_success = [np.mean(dqn_history['success_history'][:i+1])
                       for i in range(len(dqn_history['success_history']))]
    ddq_cum_success = [np.mean(ddq_history['success_history'][:i+1])
                       for i in range(len(ddq_history['success_history']))]

    ax.plot(dqn_cum_success, label='DQN', alpha=0.7)
    ax.plot(ddq_cum_success, label='DDQ', alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Success Rate')
    ax.set_title('Sample Efficiency (cumulative success)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Plot saved to: {save_path}")
    else:
        plt.show()


def print_evaluation_results(dqn_results: Dict, ddq_results: Dict):
    """Print comparison results"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    print("\n" + "-"*70)
    print("DQN Baseline:")
    print("-"*70)
    print(f"  Success Rate: {dqn_results['success_rate']:.1%}")
    print(f"  Avg Reward: {dqn_results['avg_reward']:.2f}")
    print(f"  Avg Length: {dqn_results['avg_length']:.1f} turns")

    print("\n" + "-"*70)
    print("DDQ (with World Model):")
    print("-"*70)
    print(f"  Success Rate: {ddq_results['success_rate']:.1%}")
    print(f"  Avg Reward: {ddq_results['avg_reward']:.2f}")
    print(f"  Avg Length: {ddq_results['avg_length']:.1f} turns")

    print("\n" + "-"*70)
    print("Improvement (DDQ vs DQN):")
    print("-"*70)
    success_improvement = (ddq_results['success_rate'] - dqn_results['success_rate']) / max(0.01, dqn_results['success_rate'])
    reward_improvement = (ddq_results['avg_reward'] - dqn_results['avg_reward']) / max(0.01, abs(dqn_results['avg_reward']))

    print(f"  Success Rate: {success_improvement:+.1%}")
    print(f"  Avg Reward: {reward_improvement:+.1%}")

    print("\n" + "="*70)


def main():
    """Main evaluation entry point"""
    parser = argparse.ArgumentParser(description="Evaluate DDQ/DQN Checkpoints with NLU Environment")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/ddq_final.pt',
                        help='Path to checkpoint file')
    parser.add_argument('--algorithm', type=str, default='ddq', choices=['dqn', 'ddq'],
                        help='Algorithm type (dqn or ddq)')
    parser.add_argument('--num-episodes', type=int, default=20,
                        help='Number of evaluation episodes')
    parser.add_argument('--no-llm', action='store_true',
                        help='Evaluate without LLM')
    parser.add_argument('--verbose', action='store_true',
                        help='Print sample conversations')
    
    # Legacy comparison mode
    parser.add_argument('--compare', action='store_true',
                        help='Compare DQN vs DDQ (requires both checkpoints)')
    parser.add_argument('--dqn-checkpoint', type=str, default='checkpoints/dqn_final.pt',
                        help='Path to DQN checkpoint (for comparison mode)')
    parser.add_argument('--ddq-checkpoint', type=str, default='checkpoints/ddq_final.pt',
                        help='Path to DDQ checkpoint (for comparison mode)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate comparison plots (requires --compare)')

    args = parser.parse_args()

    print("="*70)
    print("EVALUATION - NLU Environment (19-dim state)")
    print("="*70)

    # Initialize LLM
    llm_client = None
    if not args.no_llm:
        try:
            llm_client = NVIDIAClient()
            print(f"\n[OK] LLM client initialized")
        except Exception as e:
            print(f"\n[ERROR] LLM initialization failed: {e}")
            print("  Continuing without LLM")

    # Create NLU environment
    env = NLUDebtCollectionEnv(llm_client=llm_client, render_mode=None)
    print(f"[OK] NLU Environment created (state_dim={EnvironmentConfig.NLU_STATE_DIM})")

    if args.compare:
        # Comparison mode: DQN vs DDQ
        print("\n--- Comparison Mode: DQN vs DDQ ---")
        
        # Evaluate DQN
        print(f"\nEvaluating DQN...")
        if os.path.exists(args.dqn_checkpoint):
            dqn_results = evaluate_checkpoint(
                args.dqn_checkpoint,
                "dqn",
                env,
                args.num_episodes
            )
            print(f"[OK] DQN evaluation complete")
        else:
            print(f"[ERROR] DQN checkpoint not found: {args.dqn_checkpoint}")
            return

        # Evaluate DDQ
        print(f"\nEvaluating DDQ...")
        if os.path.exists(args.ddq_checkpoint):
            ddq_results = evaluate_checkpoint(
                args.ddq_checkpoint,
                "ddq",
                env,
                args.num_episodes
            )
            print(f"[OK] DDQ evaluation complete")
        else:
            print(f"[ERROR] DDQ checkpoint not found: {args.ddq_checkpoint}")
            return

        # Print results
        print_evaluation_results(dqn_results, ddq_results)

        # Generate plots if requested
        if args.plot:
            print(f"\nGenerating comparison plots...")

            dqn_history_path = os.path.join(os.path.dirname(args.dqn_checkpoint), 'training_history.json')
            ddq_history_path = os.path.join(os.path.dirname(args.ddq_checkpoint), 'training_history.json')

            if os.path.exists(dqn_history_path) and os.path.exists(ddq_history_path):
                dqn_history = load_training_history(dqn_history_path)
                ddq_history = load_training_history(ddq_history_path)

                plot_path = 'plots/dqn_vs_ddq_comparison.png'
                os.makedirs('plots', exist_ok=True)
                plot_comparison(dqn_history, ddq_history, plot_path)
            else:
                print(f"[ERROR] Training history files not found")
    else:
        # Single checkpoint evaluation mode
        print(f"\n--- Single Checkpoint Evaluation ---")
        print(f"Algorithm: {args.algorithm.upper()}")
        print(f"Checkpoint: {args.checkpoint}")
        
        if not os.path.exists(args.checkpoint):
            print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
            return
        
        results = evaluate_checkpoint(
            args.checkpoint,
            args.algorithm,
            env,
            args.num_episodes
        )
        
        # Print results
        print("\n" + "="*70)
        print(f"EVALUATION RESULTS - {args.algorithm.upper()}")
        print("="*70)
        print(f"  Success Rate: {results['success_rate']:.1%}")
        print(f"  Avg Reward: {results['avg_reward']:.2f}")
        print(f"  Avg Length: {results['avg_length']:.1f} turns")
        print("="*70)
        
        # Print sample conversations if verbose
        if args.verbose and results.get('sample_conversations'):
            print("\n--- Sample Conversations ---")
            for i, conv in enumerate(results['sample_conversations'][:3]):
                print(f"\nConversation {i+1}:")
                for turn in conv:
                    print(f"  {turn}")


if __name__ == "__main__":
    main()
