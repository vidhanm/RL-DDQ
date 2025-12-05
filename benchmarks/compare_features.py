"""
Benchmark Script: Compare Baseline vs Enhanced

Run training episodes with and without new features,
then compare performance metrics.

Usage:
    python benchmarks/compare_features.py --episodes 50
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
from typing import Dict, Tuple

from src.evaluation import ExperimentTracker, EpisodeMetrics
from src.environment.nlu_env import NLUDebtCollectionEnv
from src.agent.ddq_agent import DDQAgent


class FeatureToggle:
    """Toggle features on/off for A/B testing"""
    
    def __init__(self):
        # Default: all new features OFF (baseline)
        self.use_conversation_phase = False
        self.use_negotiation_stage = False
        self.use_opponent_modeling = False
        self.use_curriculum = False
        self.use_event_priorities = False
        self.use_success_memory = False
        self.use_lookahead = False
    
    def enable_all(self):
        """Enable all new features"""
        self.use_conversation_phase = True
        self.use_negotiation_stage = True
        self.use_opponent_modeling = True
        self.use_curriculum = True
        self.use_event_priorities = True
        self.use_success_memory = True
        self.use_lookahead = True
    
    def disable_all(self):
        """Disable all new features (baseline mode)"""
        self.use_conversation_phase = False
        self.use_negotiation_stage = False
        self.use_opponent_modeling = False
        self.use_curriculum = False
        self.use_event_priorities = False
        self.use_success_memory = False
        self.use_lookahead = False
    
    def list_enabled(self):
        """Get list of enabled features"""
        enabled = []
        if self.use_conversation_phase:
            enabled.append("conversation_phase")
        if self.use_negotiation_stage:
            enabled.append("negotiation_stage")
        if self.use_opponent_modeling:
            enabled.append("opponent_modeling")
        if self.use_curriculum:
            enabled.append("curriculum")
        if self.use_event_priorities:
            enabled.append("event_priorities")
        if self.use_success_memory:
            enabled.append("success_memory")
        if self.use_lookahead:
            enabled.append("lookahead")
        return enabled if enabled else ["baseline"]


def run_episode(env, agent, features: FeatureToggle, train: bool = True) -> Tuple[float, bool, int, Dict]:
    """
    Run single episode with feature toggles.
    
    Returns:
        (total_reward, success, num_turns, info_dict)
    """
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    num_turns = 0
    
    while not done and num_turns < 15:
        # Select action
        if features.use_lookahead and hasattr(agent, 'select_action_with_lookahead'):
            action = agent.select_action_with_lookahead(obs, lookahead_depth=2)
        else:
            action = agent.select_action(obs, explore=train)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        num_turns += 1
        
        # Store experience
        if train:
            agent.replay_buffer.add(obs, action, reward, next_obs, done)
            
            # Train step
            if len(agent.replay_buffer) >= agent.batch_size:
                agent.train_step()
        
        obs = next_obs
        done = done or truncated
    
    # Determine success
    success = info.get('got_commitment', False)
    
    # Final NLU features
    final_sentiment = info.get('sentiment', 0.0)
    final_cooperation = info.get('cooperation', 0.5)
    
    return total_reward, success, num_turns, {
        'final_sentiment': final_sentiment,
        'final_cooperation': final_cooperation,
        'phase': env.conversation_history[-1].get('phase', 'unknown') if env.conversation_history else 'unknown',
        'negotiation_stage': env.conversation_history[-1].get('negotiation_stage', 'none') if env.conversation_history else 'none'
    }


def run_benchmark(num_episodes: int = 50, verbose: bool = True):
    """
    Run benchmark comparing baseline vs enhanced.
    
    Args:
        num_episodes: Episodes per experiment
        verbose: Print progress
    """
    print("\n" + "="*70)
    print("BENCHMARK: Baseline vs Enhanced Features")
    print("="*70)
    
    tracker = ExperimentTracker("baseline_vs_enhanced", save_dir="experiments")
    
    # =========================================================================
    # EXPERIMENT 1: BASELINE (no new features)
    # =========================================================================
    print("\n" + "-"*50)
    print("Running BASELINE (no new features)...")
    print("-"*50)
    
    features_baseline = FeatureToggle()
    features_baseline.disable_all()
    
    env = NLUDebtCollectionEnv(llm_client=None)
    agent = DDQAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        use_ensemble=True
    )
    
    tracker.start_experiment("baseline", features_baseline.list_enabled())
    
    for ep in range(num_episodes):
        total_reward, success, num_turns, extra = run_episode(
            env, agent, features_baseline, train=True
        )
        
        metrics = EpisodeMetrics(
            episode_id=ep,
            success=success,
            total_reward=total_reward,
            num_turns=num_turns,
            final_sentiment=extra['final_sentiment'],
            final_cooperation=extra['final_cooperation'],
            conversation_phase=extra['phase'],
            negotiation_stage=extra['negotiation_stage']
        )
        tracker.record_episode(metrics)
        
        if verbose and (ep + 1) % 10 == 0:
            recent = tracker.current_episodes[-10:]
            recent_success = sum(1 for e in recent if e.success) / len(recent)
            recent_reward = np.mean([e.total_reward for e in recent])
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Success={recent_success:.0%}, Reward={recent_reward:.1f}")
    
    tracker.end_experiment()
    
    # =========================================================================
    # EXPERIMENT 2: ENHANCED (all new features)
    # =========================================================================
    print("\n" + "-"*50)
    print("Running ENHANCED (all new features)...")
    print("-"*50)
    
    features_enhanced = FeatureToggle()
    features_enhanced.enable_all()
    
    # Fresh environment and agent
    env = NLUDebtCollectionEnv(llm_client=None)
    agent = DDQAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        use_ensemble=True
    )
    
    # Initialize curriculum if enabled
    curriculum = None
    if features_enhanced.use_curriculum:
        from src.curriculum_learning import DifficultyAutoCurriculum
        curriculum = DifficultyAutoCurriculum()
    
    tracker.start_experiment("enhanced", features_enhanced.list_enabled())
    
    for ep in range(num_episodes):
        total_reward, success, num_turns, extra = run_episode(
            env, agent, features_enhanced, train=True
        )
        
        # Update curriculum if enabled
        if curriculum:
            curriculum.record_episode(success)
        
        metrics = EpisodeMetrics(
            episode_id=ep,
            success=success,
            total_reward=total_reward,
            num_turns=num_turns,
            final_sentiment=extra['final_sentiment'],
            final_cooperation=extra['final_cooperation'],
            conversation_phase=extra['phase'],
            negotiation_stage=extra['negotiation_stage']
        )
        tracker.record_episode(metrics)
        
        if verbose and (ep + 1) % 10 == 0:
            recent = tracker.current_episodes[-10:]
            recent_success = sum(1 for e in recent if e.success) / len(recent)
            recent_reward = np.mean([e.total_reward for e in recent])
            print(f"  Episode {ep+1}/{num_episodes}: "
                  f"Success={recent_success:.0%}, Reward={recent_reward:.1f}")
    
    tracker.end_experiment()
    
    # =========================================================================
    # COMPARISON REPORT
    # =========================================================================
    tracker.generate_comparison_report()
    tracker.save()
    
    return tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark baseline vs enhanced features")
    parser.add_argument("--episodes", type=int, default=30, help="Episodes per experiment")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()
    
    run_benchmark(num_episodes=args.episodes, verbose=not args.quiet)
