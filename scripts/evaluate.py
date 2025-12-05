"""
Evaluation Script
Evaluate DDQ/DQN checkpoints with NLU environment
Supports adversarial evaluation modes for self-play trained agents
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.nlu_env import NLUDebtCollectionEnv
from src.environment.selfplay_env import SelfPlayEnv
from src.agent.dqn_agent import DQNAgent
from src.agent.ddq_agent import DDQAgent
from src.agent.adversarial_agent import AdversarialDebtorAgent, create_adversarial_agent
from src.llm.nvidia_client import NVIDIAClient
from src.config import EnvironmentConfig, SelfPlayConfig, DeviceConfig



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


# =============================================================================
# ADVERSARIAL EVALUATION FUNCTIONS
# =============================================================================

def evaluate_against_adversary(
    collector_path: str,
    adversary_path: str,
    num_episodes: int = 50,
    use_llm: bool = False,
    render: bool = False
) -> Dict:
    """
    Evaluate collector against a trained adversarial debtor.
    
    Args:
        collector_path: Path to collector checkpoint
        adversary_path: Path to adversary checkpoint
        num_episodes: Number of evaluation episodes
        use_llm: Whether to use LLM for conversations
        render: Whether to render conversations
        
    Returns:
        Evaluation metrics
    """
    print(f"\n--- Adversarial Evaluation ---")
    print(f"Collector: {collector_path}")
    print(f"Adversary: {adversary_path}")
    
    # Create agents
    collector = DDQAgent(
        state_dim=EnvironmentConfig.NLU_STATE_DIM,
        action_dim=EnvironmentConfig.NUM_ACTIONS,
    )
    collector.load(collector_path)
    
    adversary = create_adversarial_agent()
    adversary.load(adversary_path)
    
    # Create environment
    llm_client = None
    if use_llm:
        try:
            llm_client = NVIDIAClient()
        except Exception as e:
            print(f"[WARN] LLM init failed: {e}")
    
    env = SelfPlayEnv(
        llm_client=llm_client,
        render_mode="human" if render else None
    )
    
    # Run evaluation
    collector_wins = 0
    adversary_wins = 0
    draws = 0
    collector_rewards = []
    adversary_rewards = []
    turns_list = []
    outcomes = []
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_c_reward = 0
        episode_a_reward = 0
        
        while not done:
            c_action = collector.select_action(obs, explore=False)
            a_action = adversary.select_strategy(obs)
            
            obs, c_reward, a_reward, terminated, truncated, step_info = env.step(
                c_action, a_action
            )
            
            done = terminated or truncated
            episode_c_reward += c_reward
            episode_a_reward += a_reward
        
        # Record outcome
        outcome = step_info.get("outcome", "unknown")
        outcomes.append(outcome)
        
        if outcome == "collector_win":
            collector_wins += 1
        elif outcome == "adversary_win":
            adversary_wins += 1
        else:
            draws += 1
        
        collector_rewards.append(episode_c_reward)
        adversary_rewards.append(episode_a_reward)
        turns_list.append(step_info.get("turn", 0))
    
    results = {
        "collector_win_rate": collector_wins / num_episodes,
        "adversary_win_rate": adversary_wins / num_episodes,
        "draw_rate": draws / num_episodes,
        "avg_collector_reward": np.mean(collector_rewards),
        "avg_adversary_reward": np.mean(adversary_rewards),
        "avg_turns": np.mean(turns_list),
        "outcomes": outcomes,
    }
    
    print(f"\n📊 Adversarial Evaluation Results:")
    print(f"   Collector Win Rate: {results['collector_win_rate']:.1%}")
    print(f"   Adversary Win Rate: {results['adversary_win_rate']:.1%}")
    print(f"   Draw Rate: {results['draw_rate']:.1%}")
    print(f"   Avg Turns: {results['avg_turns']:.1f}")
    
    return results


def robustness_benchmark(
    collector_path: str,
    num_episodes_per_level: int = 20,
    use_llm: bool = False
) -> Dict:
    """
    Evaluate collector against different difficulty levels.
    
    Tests robustness by running against:
    - Easy debtors (high agreeableness)
    - Medium debtors (balanced)
    - Hard debtors (low agreeableness)
    - Adversarial (if checkpoint exists)
    
    Args:
        collector_path: Path to collector checkpoint
        num_episodes_per_level: Episodes per difficulty level
        use_llm: Whether to use LLM
        
    Returns:
        Benchmark results for each difficulty
    """
    print(f"\n--- Robustness Benchmark ---")
    print(f"Collector: {collector_path}")
    
    # Create collector
    collector = DDQAgent(
        state_dim=EnvironmentConfig.NLU_STATE_DIM,
        action_dim=EnvironmentConfig.NUM_ACTIONS,
    )
    collector.load(collector_path)
    
    # Initialize LLM if needed
    llm_client = None
    if use_llm:
        try:
            llm_client = NVIDIAClient()
        except Exception as e:
            print(f"[WARN] LLM init failed: {e}")
    
    # Create environment
    env = NLUDebtCollectionEnv(llm_client=llm_client)
    
    results = {}
    difficulties = ["easy", "medium", "hard"]
    
    for difficulty in difficulties:
        print(f"\n  Testing {difficulty.upper()} debtors...")
        
        successes = 0
        rewards = []
        turns = []
        
        for ep in range(num_episodes_per_level):
            state, info = env.reset(options={"difficulty": difficulty})
            episode_reward = 0
            done = False
            turn = 0
            
            while not done:
                action = collector.select_action(state, explore=False)
                state, reward, terminated, truncated, step_info = env.step(action)
                episode_reward += reward
                turn += 1
                done = terminated or truncated
            
            if step_info.get("has_committed", False):
                successes += 1
            rewards.append(episode_reward)
            turns.append(turn)
        
        results[difficulty] = {
            "success_rate": successes / num_episodes_per_level,
            "avg_reward": np.mean(rewards),
            "avg_turns": np.mean(turns),
        }
        
        print(f"    Success: {results[difficulty]['success_rate']:.1%}, "
              f"Reward: {results[difficulty]['avg_reward']:.2f}")
    
    # Summary
    print(f"\n📊 Robustness Benchmark Summary:")
    for diff, res in results.items():
        print(f"   {diff.upper():8s}: {res['success_rate']:.1%} success, "
              f"{res['avg_reward']:.2f} reward, {res['avg_turns']:.1f} turns")
    
    return results


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
    
    # NEW: Adversarial evaluation modes
    parser.add_argument('--adversarial', action='store_true',
                        help='Evaluate collector against trained adversary')
    parser.add_argument('--adversary-checkpoint', type=str, 
                        default='checkpoints/selfplay/adversary_final.pt',
                        help='Path to adversary checkpoint')
    parser.add_argument('--collector-checkpoint', type=str,
                        default='checkpoints/selfplay/collector_final.pt',
                        help='Path to collector checkpoint (for adversarial mode)')
    parser.add_argument('--robustness', action='store_true',
                        help='Run robustness benchmark across difficulty levels')
    parser.add_argument('--render', action='store_true',
                        help='Render conversations to console')

    args = parser.parse_args()

    print("="*70)
    print("EVALUATION - NLU Environment")
    print("="*70)

    # === ADVERSARIAL EVALUATION MODE ===
    if args.adversarial:
        if not os.path.exists(args.collector_checkpoint):
            print(f"[ERROR] Collector not found: {args.collector_checkpoint}")
            return
        if not os.path.exists(args.adversary_checkpoint):
            print(f"[ERROR] Adversary not found: {args.adversary_checkpoint}")
            return
        
        results = evaluate_against_adversary(
            collector_path=args.collector_checkpoint,
            adversary_path=args.adversary_checkpoint,
            num_episodes=args.num_episodes,
            use_llm=not args.no_llm,
            render=args.render
        )
        return
    
    # === ROBUSTNESS BENCHMARK MODE ===
    if args.robustness:
        checkpoint = args.collector_checkpoint if args.adversarial else args.checkpoint
        if not os.path.exists(checkpoint):
            print(f"[ERROR] Checkpoint not found: {checkpoint}")
            return
        
        results = robustness_benchmark(
            collector_path=checkpoint,
            num_episodes_per_level=args.num_episodes,
            use_llm=not args.no_llm
        )
        return

    # Initialize LLM for standard modes
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
