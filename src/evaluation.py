"""
Evaluation Framework for A/B Testing

Compare baseline (main branch) vs enhanced (new features) performance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode"""
    episode_id: int
    success: bool  # Got commitment
    total_reward: float
    num_turns: int
    final_sentiment: float
    final_cooperation: float
    debtor_type: str = "UNKNOWN"
    conversation_phase: str = "unknown"
    negotiation_stage: str = "none"


@dataclass 
class ExperimentMetrics:
    """Aggregated metrics for an experiment run"""
    experiment_name: str
    features_enabled: List[str]
    start_time: str
    end_time: str = ""
    
    # Core metrics
    total_episodes: int = 0
    successful_episodes: int = 0
    success_rate: float = 0.0
    
    # Reward metrics
    avg_reward: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    reward_std: float = 0.0
    
    # Efficiency metrics
    avg_turns_to_success: float = 0.0
    avg_turns_to_failure: float = 0.0
    
    # Sentiment/Cooperation metrics
    avg_final_sentiment: float = 0.0
    avg_final_cooperation: float = 0.0
    
    # Convergence
    episodes_to_50pct_success: int = -1  # -1 = not reached
    episodes_to_70pct_success: int = -1
    
    # Raw episode data
    episode_rewards: List[float] = field(default_factory=list)


class ExperimentTracker:
    """
    Track and compare experiments.
    
    Usage:
        tracker = ExperimentTracker("baseline_vs_enhanced")
        
        # Run baseline
        tracker.start_experiment("baseline", features_enabled=[])
        for ep in range(100):
            result = run_episode(...)
            tracker.record_episode(result)
        tracker.end_experiment()
        
        # Run enhanced
        tracker.start_experiment("enhanced", features_enabled=["all"])
        for ep in range(100):
            result = run_episode(...)
            tracker.record_episode(result)
        tracker.end_experiment()
        
        # Compare
        tracker.generate_comparison_report()
    """
    
    def __init__(self, name: str, save_dir: str = "experiments"):
        self.name = name
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.experiments: Dict[str, ExperimentMetrics] = {}
        self.current_experiment: Optional[str] = None
        self.current_episodes: List[EpisodeMetrics] = []
        
    def start_experiment(self, experiment_name: str, features_enabled: List[str]):
        """Start a new experiment run"""
        self.current_experiment = experiment_name
        self.current_episodes = []
        
        self.experiments[experiment_name] = ExperimentMetrics(
            experiment_name=experiment_name,
            features_enabled=features_enabled,
            start_time=datetime.now().isoformat()
        )
        print(f"\n{'='*60}")
        print(f"Starting experiment: {experiment_name}")
        print(f"Features: {features_enabled}")
        print(f"{'='*60}")
    
    def record_episode(self, metrics: EpisodeMetrics):
        """Record episode results"""
        self.current_episodes.append(metrics)
        
        # Update running stats
        exp = self.experiments[self.current_experiment]
        exp.total_episodes += 1
        if metrics.success:
            exp.successful_episodes += 1
        exp.episode_rewards.append(metrics.total_reward)
        
        # Check convergence thresholds
        if exp.total_episodes >= 20:
            recent_success = sum(1 for e in self.current_episodes[-20:] if e.success) / 20
            if recent_success >= 0.5 and exp.episodes_to_50pct_success < 0:
                exp.episodes_to_50pct_success = exp.total_episodes
            if recent_success >= 0.7 and exp.episodes_to_70pct_success < 0:
                exp.episodes_to_70pct_success = exp.total_episodes
    
    def end_experiment(self):
        """Finalize experiment and compute aggregated metrics"""
        if not self.current_experiment:
            return
            
        exp = self.experiments[self.current_experiment]
        exp.end_time = datetime.now().isoformat()
        
        if not self.current_episodes:
            return
        
        # Core metrics
        exp.success_rate = exp.successful_episodes / exp.total_episodes
        
        # Reward metrics
        rewards = [e.total_reward for e in self.current_episodes]
        exp.avg_reward = np.mean(rewards)
        exp.max_reward = np.max(rewards)
        exp.min_reward = np.min(rewards)
        exp.reward_std = np.std(rewards)
        
        # Efficiency
        successes = [e for e in self.current_episodes if e.success]
        failures = [e for e in self.current_episodes if not e.success]
        if successes:
            exp.avg_turns_to_success = np.mean([e.num_turns for e in successes])
        if failures:
            exp.avg_turns_to_failure = np.mean([e.num_turns for e in failures])
        
        # Sentiment/Cooperation
        exp.avg_final_sentiment = np.mean([e.final_sentiment for e in self.current_episodes])
        exp.avg_final_cooperation = np.mean([e.final_cooperation for e in self.current_episodes])
        
        print(f"\n✓ Experiment '{self.current_experiment}' complete")
        print(f"  Success rate: {exp.success_rate:.1%}")
        print(f"  Avg reward: {exp.avg_reward:.2f}")
        
        self.current_experiment = None
    
    def generate_comparison_report(self) -> str:
        """Generate comparison report between experiments"""
        if len(self.experiments) < 2:
            return "Need at least 2 experiments to compare"
        
        report_lines = [
            "=" * 70,
            f"EXPERIMENT COMPARISON: {self.name}",
            "=" * 70,
            "",
        ]
        
        # Header
        exp_names = list(self.experiments.keys())
        header = f"{'Metric':<30}" + "".join(f"{n:<20}" for n in exp_names)
        report_lines.append(header)
        report_lines.append("-" * 70)
        
        # Key metrics
        metrics_to_show = [
            ("Success Rate", lambda e: f"{e.success_rate:.1%}"),
            ("Avg Reward", lambda e: f"{e.avg_reward:.2f}"),
            ("Reward Std", lambda e: f"{e.reward_std:.2f}"),
            ("Avg Turns (Success)", lambda e: f"{e.avg_turns_to_success:.1f}"),
            ("Avg Final Sentiment", lambda e: f"{e.avg_final_sentiment:.2f}"),
            ("Avg Final Cooperation", lambda e: f"{e.avg_final_cooperation:.2f}"),
            ("Episodes to 50% success", lambda e: str(e.episodes_to_50pct_success)),
            ("Episodes to 70% success", lambda e: str(e.episodes_to_70pct_success)),
        ]
        
        for metric_name, metric_fn in metrics_to_show:
            values = [metric_fn(self.experiments[n]) for n in exp_names]
            row = f"{metric_name:<30}" + "".join(f"{v:<20}" for v in values)
            report_lines.append(row)
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        # Summary
        if len(exp_names) >= 2:
            baseline = self.experiments[exp_names[0]]
            enhanced = self.experiments[exp_names[1]]
            
            success_diff = enhanced.success_rate - baseline.success_rate
            reward_diff = enhanced.avg_reward - baseline.avg_reward
            
            report_lines.append("")
            report_lines.append("SUMMARY:")
            if success_diff > 0:
                report_lines.append(f"  ✅ Enhanced is {success_diff:.1%} BETTER in success rate")
            else:
                report_lines.append(f"  ❌ Enhanced is {abs(success_diff):.1%} WORSE in success rate")
            
            if reward_diff > 0:
                report_lines.append(f"  ✅ Enhanced has {reward_diff:.2f} HIGHER avg reward")
            else:
                report_lines.append(f"  ❌ Enhanced has {abs(reward_diff):.2f} LOWER avg reward")
        
        report = "\n".join(report_lines)
        print(report)
        
        # Save report
        report_path = os.path.join(self.save_dir, f"{self.name}_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nReport saved to: {report_path}")
        
        return report
    
    def save(self):
        """Save experiment data"""
        save_path = os.path.join(self.save_dir, f"{self.name}_data.json")
        data = {
            name: asdict(exp) for name, exp in self.experiments.items()
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Data saved to: {save_path}")


# ============================================================================
# QUICK TEST FUNCTIONS
# ============================================================================

def quick_feature_test():
    """Quick test that all new features work together"""
    print("\n" + "="*60)
    print("QUICK FEATURE INTEGRATION TEST")
    print("="*60)
    
    results = {}
    
    # 1. Test NLU environment with phases and stages
    print("\n1. Testing NLU Environment...")
    try:
        from src.environment.nlu_env import NLUDebtCollectionEnv
        env = NLUDebtCollectionEnv(llm_client=None)
        obs, info = env.reset()
        
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        last = env.conversation_history[-1]
        results['nlu_env'] = {
            'status': '✅ PASS',
            'phase': last.get('phase', 'N/A'),
            'negotiation_stage': last.get('negotiation_stage', 'N/A')
        }
        print(f"   ✅ Phase: {last.get('phase')}, Stage: {last.get('negotiation_stage')}")
    except Exception as e:
        results['nlu_env'] = {'status': '❌ FAIL', 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # 2. Test Opponent Pool
    print("\n2. Testing Opponent Pool...")
    try:
        from src.environment.opponent_pool import OpponentPool
        pool = OpponentPool()
        adv = pool.sample(style='hostile')
        results['opponent_pool'] = {
            'status': '✅ PASS',
            'count': len(pool),
            'sampled': adv.name
        }
        print(f"   ✅ Pool size: {len(pool)}, Sampled: {adv.name}")
    except Exception as e:
        results['opponent_pool'] = {'status': '❌ FAIL', 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # 3. Test Success Memory
    print("\n3. Testing Success Memory...")
    try:
        from src.utils.success_memory import SuccessMemory
        memory = SuccessMemory()
        memory.record_success("HOSTILE", "empathy", -0.5, 0.0, 0.3)
        context = memory.get_context_for_prompt("HOSTILE")
        results['success_memory'] = {
            'status': '✅ PASS',
            'has_context': len(context) > 0
        }
        print(f"   ✅ Context generated: {len(context)} chars")
    except Exception as e:
        results['success_memory'] = {'status': '❌ FAIL', 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # 4. Test Curriculum
    print("\n4. Testing Auto-Curriculum...")
    try:
        from src.curriculum_learning import DifficultyAutoCurriculum, QARLStrengthScheduler
        
        curriculum = DifficultyAutoCurriculum()
        qarl = QARLStrengthScheduler()
        
        # Simulate episodes
        for _ in range(10):
            curriculum.record_episode(success=True)
            qarl.record_episode(success=True)
        
        results['curriculum'] = {
            'status': '✅ PASS',
            'difficulty': curriculum.current_difficulty,
            'qarl_strength': qarl.get_strength()
        }
        print(f"   ✅ Difficulty: {curriculum.current_difficulty}, QARL: {qarl.get_strength():.2f}")
    except Exception as e:
        results['curriculum'] = {'status': '❌ FAIL', 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # 5. Test Prioritized Replay with event bonuses
    print("\n5. Testing Prioritized Replay...")
    try:
        from src.utils.replay_buffer import PrioritizedReplayBuffer
        buffer = PrioritizedReplayBuffer(100)
        
        # Add with info
        state = np.zeros(10)
        info = {'has_committed': True, 'sentiment': 0.5}
        buffer.add_with_info(state, 0, 5.0, state, False, info)
        
        stats = buffer.get_event_statistics()
        results['replay_buffer'] = {
            'status': '✅ PASS',
            'event_stats': stats
        }
        print(f"   ✅ Event stats: {stats}")
    except Exception as e:
        results['replay_buffer'] = {'status': '❌ FAIL', 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # 6. Test World Model with commitment predictor
    print("\n6. Testing World Model...")
    try:
        import torch
        from src.agent.world_model import WorldModel
        
        model = WorldModel(state_dim=10, action_dim=9)
        state = torch.randn(1, 10)
        action = torch.zeros(1, 9)
        action[0, 0] = 1.0
        
        commit_prob = model.predict_commitment(state, action)
        results['world_model'] = {
            'status': '✅ PASS',
            'commit_prob': commit_prob.item()
        }
        print(f"   ✅ Commitment prob: {commit_prob.item():.3f}")
    except Exception as e:
        results['world_model'] = {'status': '❌ FAIL', 'error': str(e)}
        print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    passed = sum(1 for r in results.values() if '✅' in r['status'])
    total = len(results)
    print(f"RESULTS: {passed}/{total} components passed")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # Run quick feature test
    quick_feature_test()
