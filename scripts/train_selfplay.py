"""
Self-Play Training Script for Adversarial Agent Training

Implements Fictitious Self-Play (FSP) training loop:
1. Train collector against sampled adversaries
2. Train adversary against sampled collectors
3. Add both to opponent pools
4. Repeat for N generations
"""

import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    SelfPlayConfig, EnvironmentConfig, TrainingConfig, 
    RLConfig, DDQConfig, DeviceConfig
)
from src.agent.ddq_agent import DDQAgent
from src.agent.adversarial_agent import AdversarialDebtorAgent, create_adversarial_agent
from src.agent.opponent_pool import DualPoolManager
from src.environment.selfplay_env import SelfPlayEnv


def create_collector_agent() -> DDQAgent:
    """Create a new collector agent."""
    return DDQAgent(
        state_dim=EnvironmentConfig.NLU_STATE_DIM,
        action_dim=EnvironmentConfig.NUM_ACTIONS,
    )


def load_agent_from_checkpoint(agent, checkpoint_path: str):
    """Load agent weights from checkpoint."""
    if checkpoint_path and os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
    return agent


def train_one_episode(
    env: SelfPlayEnv,
    collector: DDQAgent,
    adversary: AdversarialDebtorAgent,
    train_collector: bool = True,
    train_adversary: bool = True
) -> Dict[str, Any]:
    """
    Run one episode of self-play.
    
    Args:
        env: SelfPlayEnv instance
        collector: Collector agent
        adversary: Adversary agent
        train_collector: Whether to train collector
        train_adversary: Whether to train adversary
        
    Returns:
        Episode statistics
    """
    obs, info = env.reset()
    
    done = False
    episode_stats = {
        "turns": 0,
        "collector_reward": 0.0,
        "adversary_reward": 0.0,
        "outcome": "ongoing"
    }
    
    while not done:
        # Both agents select actions
        collector_action = collector.select_action(obs)
        adversary_action = adversary.select_strategy(obs)
        
        # Environment step
        next_obs, c_reward, a_reward, terminated, truncated, info = env.step(
            collector_action, adversary_action
        )
        
        done = terminated or truncated
        
        # Store experiences
        if train_collector:
            collector.store_experience(obs, collector_action, c_reward, next_obs, done)
        
        if train_adversary:
            adversary.store_experience(obs, adversary_action, a_reward, next_obs, done)
        
        # Update stats
        episode_stats["turns"] += 1
        episode_stats["collector_reward"] += c_reward
        episode_stats["adversary_reward"] += a_reward
        
        obs = next_obs
    
    episode_stats["outcome"] = info.get("outcome", "unknown")
    
    # End episode for adversary stats tracking
    adversary.episode_end(collector_succeeded=(episode_stats["outcome"] == "collector_win"))
    
    return episode_stats


def train_generation(
    env: SelfPlayEnv,
    collector: DDQAgent,
    adversary: AdversarialDebtorAgent,
    pool_manager: DualPoolManager,
    generation: int,
    episodes_per_gen: int = SelfPlayConfig.EPISODES_PER_GENERATION,
    render: bool = False
) -> Dict[str, Any]:
    """
    Train one generation of self-play.
    
    Args:
        env: SelfPlayEnv instance
        collector: Current collector agent
        adversary: Current adversary agent
        pool_manager: Dual pool manager
        generation: Current generation number
        episodes_per_gen: Episodes to run this generation
        render: Whether to render episodes
        
    Returns:
        Generation statistics
    """
    gen_stats = {
        "generation": generation,
        "collector_wins": 0,
        "adversary_wins": 0,
        "draws": 0,
        "total_collector_reward": 0.0,
        "total_adversary_reward": 0.0,
        "collector_losses": [],
        "adversary_losses": [],
    }
    
    # Set render mode
    if render:
        env.render_mode = "human"
    else:
        env.render_mode = None
    
    # Training loop for this generation
    pbar = tqdm(range(episodes_per_gen), desc=f"Gen {generation}", leave=False)
    
    for episode in pbar:
        # Run episode
        ep_stats = train_one_episode(env, collector, adversary)
        
        # Update gen stats
        gen_stats["total_collector_reward"] += ep_stats["collector_reward"]
        gen_stats["total_adversary_reward"] += ep_stats["adversary_reward"]
        
        if ep_stats["outcome"] == "collector_win":
            gen_stats["collector_wins"] += 1
        elif ep_stats["outcome"] == "adversary_win":
            gen_stats["adversary_wins"] += 1
        else:
            gen_stats["draws"] += 1
        
        # Train agents
        if collector.replay_buffer.is_ready(RLConfig.BATCH_SIZE):
            c_loss = collector.train_step()
            if c_loss is not None:
                gen_stats["collector_losses"].append(c_loss)
        
        if adversary.replay_buffer.is_ready(RLConfig.BATCH_SIZE):
            a_loss = adversary.train_step()
            if a_loss is not None:
                gen_stats["adversary_losses"].append(a_loss)
        
        # Update progress bar
        c_win_rate = gen_stats["collector_wins"] / (episode + 1)
        pbar.set_postfix({
            "C_win": f"{c_win_rate:.1%}",
            "A_win": f"{gen_stats['adversary_wins']/(episode+1):.1%}"
        })
    
    # Calculate final stats
    total_episodes = episodes_per_gen
    gen_stats["collector_win_rate"] = gen_stats["collector_wins"] / total_episodes
    gen_stats["adversary_win_rate"] = gen_stats["adversary_wins"] / total_episodes
    gen_stats["avg_collector_reward"] = gen_stats["total_collector_reward"] / total_episodes
    gen_stats["avg_adversary_reward"] = gen_stats["total_adversary_reward"] / total_episodes
    
    if gen_stats["collector_losses"]:
        gen_stats["avg_collector_loss"] = np.mean(gen_stats["collector_losses"])
    if gen_stats["adversary_losses"]:
        gen_stats["avg_adversary_loss"] = np.mean(gen_stats["adversary_losses"])
    
    return gen_stats


def run_selfplay_training(
    generations: int = SelfPlayConfig.GENERATIONS,
    episodes_per_gen: int = SelfPlayConfig.EPISODES_PER_GENERATION,
    use_llm: bool = False,
    render: bool = False,
    save_dir: str = "checkpoints/selfplay",
    collector_checkpoint: Optional[str] = None,
    adversary_checkpoint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run full self-play training loop.
    
    Args:
        generations: Number of generations to train
        episodes_per_gen: Episodes per generation
        use_llm: Whether to use LLM for conversations
        render: Whether to render conversations
        save_dir: Directory to save checkpoints
        collector_checkpoint: Optional checkpoint to resume collector from
        adversary_checkpoint: Optional checkpoint to resume adversary from
        
    Returns:
        Training history
    """
    print("=" * 70)
    print("ADVERSARIAL SELF-PLAY TRAINING")
    print("=" * 70)
    print(f"Generations: {generations}")
    print(f"Episodes/Gen: {episodes_per_gen}")
    print(f"Use LLM: {use_llm}")
    print(f"Save Dir: {save_dir}")
    print("=" * 70)
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize LLM client if needed
    llm_client = None
    if use_llm:
        try:
            from src.llm.nvidia_client import NvidiaLLMClient
            llm_client = NvidiaLLMClient()
            print("✓ LLM client initialized")
        except Exception as e:
            print(f"⚠ LLM init failed: {e}, running without LLM")
    
    # Create environment
    env = SelfPlayEnv(
        llm_client=llm_client,
        render_mode="human" if render else None,
        use_llm_for_collector=use_llm,
        use_llm_for_adversary=use_llm
    )
    
    # Create agents
    collector = create_collector_agent()
    adversary = create_adversarial_agent()
    
    # Load from checkpoints if provided
    if collector_checkpoint:
        load_agent_from_checkpoint(collector, collector_checkpoint)
        print(f"✓ Loaded collector from {collector_checkpoint}")
    
    if adversary_checkpoint:
        load_agent_from_checkpoint(adversary, adversary_checkpoint)
        print(f"✓ Loaded adversary from {adversary_checkpoint}")
    
    # Initialize opponent pools
    pool_manager = DualPoolManager(
        pool_dir=os.path.join(save_dir, "opponent_pool"),
        max_size=SelfPlayConfig.OPPONENT_POOL_SIZE
    )
    
    # Training history
    history = {
        "generations": [],
        "collector_win_rates": [],
        "adversary_win_rates": [],
        "collector_rewards": [],
        "adversary_rewards": [],
    }
    
    # Main training loop
    print("\n🎮 Starting Self-Play Training...\n")
    
    for gen in range(generations):
        print(f"\n--- Generation {gen + 1}/{generations} ---")
        
        # Train this generation
        gen_stats = train_generation(
            env=env,
            collector=collector,
            adversary=adversary,
            pool_manager=pool_manager,
            generation=gen,
            episodes_per_gen=episodes_per_gen,
            render=render
        )
        
        # Update history
        history["generations"].append(gen)
        history["collector_win_rates"].append(gen_stats["collector_win_rate"])
        history["adversary_win_rates"].append(gen_stats["adversary_win_rate"])
        history["collector_rewards"].append(gen_stats["avg_collector_reward"])
        history["adversary_rewards"].append(gen_stats["avg_adversary_reward"])
        
        # Print generation summary
        print(f"\n📊 Generation {gen + 1} Results:")
        print(f"   Collector Win Rate: {gen_stats['collector_win_rate']:.1%}")
        print(f"   Adversary Win Rate: {gen_stats['adversary_win_rate']:.1%}")
        print(f"   Avg Collector Reward: {gen_stats['avg_collector_reward']:.2f}")
        print(f"   Avg Adversary Reward: {gen_stats['avg_adversary_reward']:.2f}")
        
        # Save to opponent pools
        pool_manager.add_collector(collector, gen, gen_stats["collector_win_rate"])
        pool_manager.add_adversary(adversary, gen, gen_stats["adversary_win_rate"])
        
        # Save checkpoints periodically
        if (gen + 1) % 5 == 0 or gen == generations - 1:
            collector.save(os.path.join(save_dir, f"collector_gen{gen+1:04d}.pt"))
            adversary.save(os.path.join(save_dir, f"adversary_gen{gen+1:04d}.pt"))
            print(f"   💾 Checkpoints saved")
        
        # Check convergence
        if gen_stats["collector_win_rate"] >= SelfPlayConfig.WIN_RATE_THRESHOLD:
            if gen >= SelfPlayConfig.MIN_GENERATIONS:
                print(f"\n🎯 Convergence reached! Collector win rate: {gen_stats['collector_win_rate']:.1%}")
                break
    
    # Final save
    collector.save(os.path.join(save_dir, "collector_final.pt"))
    adversary.save(os.path.join(save_dir, "adversary_final.pt"))
    
    # Print final statistics
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total Generations: {len(history['generations'])}")
    print(f"Final Collector Win Rate: {history['collector_win_rates'][-1]:.1%}")
    print(f"Final Adversary Win Rate: {history['adversary_win_rates'][-1]:.1%}")
    print(f"Pool Stats: {pool_manager.get_statistics()}")
    print("=" * 70)
    
    return history


def main():
    parser = argparse.ArgumentParser(description="Self-Play Training for Adversarial Agent")
    
    parser.add_argument("--generations", type=int, default=10,
                        help="Number of generations to train")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per generation")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM for conversations")
    parser.add_argument("--render", action="store_true",
                        help="Render conversations to console")
    parser.add_argument("--save-dir", type=str, default="checkpoints/selfplay",
                        help="Directory to save checkpoints")
    parser.add_argument("--collector-checkpoint", type=str, default=None,
                        help="Resume collector from checkpoint")
    parser.add_argument("--adversary-checkpoint", type=str, default=None,
                        help="Resume adversary from checkpoint")
    parser.add_argument("--test-mode", action="store_true",
                        help="Quick test with minimal episodes")
    
    args = parser.parse_args()
    
    # Test mode overrides
    if args.test_mode:
        args.generations = 2
        args.episodes = 5
        print("🧪 TEST MODE: Running with minimal episodes")
    
    # Run training
    history = run_selfplay_training(
        generations=args.generations,
        episodes_per_gen=args.episodes,
        use_llm=args.use_llm,
        render=args.render,
        save_dir=args.save_dir,
        collector_checkpoint=args.collector_checkpoint,
        adversary_checkpoint=args.adversary_checkpoint,
    )
    
    print("\n✓ Training complete!")
    return history


if __name__ == "__main__":
    main()
