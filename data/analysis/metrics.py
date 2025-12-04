"""
Metrics Calculator
Compute derived metrics and statistics from training runs
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import will be relative when used as package
try:
    from analysis.history_loader import TrainingRun
except ImportError:
    from history_loader import TrainingRun


@dataclass
class ComparisonMetrics:
    """Metrics comparing two training runs"""
    
    # Basic comparisons
    success_rate_diff: float  # run2 - run1
    success_rate_improvement: float  # percentage improvement
    reward_diff: float
    reward_improvement: float
    
    # Sample efficiency
    episodes_to_50_success_run1: Optional[int]
    episodes_to_50_success_run2: Optional[int]
    sample_efficiency_gain: Optional[float]  # How many fewer episodes needed
    
    # Convergence
    final_10_success_run1: float
    final_10_success_run2: float
    
    # Statistical significance (if enough data)
    t_statistic: Optional[float] = None
    p_value: Optional[float] = None


class MetricsCalculator:
    """Calculate various metrics from training runs"""
    
    @staticmethod
    def episodes_to_threshold(
        success_history: List[int], 
        threshold: float = 0.5,
        window: int = 10
    ) -> Optional[int]:
        """
        Find first episode where rolling success rate exceeds threshold
        
        Args:
            success_history: List of 0/1 success values
            threshold: Success rate threshold (default 50%)
            window: Rolling window size
            
        Returns:
            Episode number or None if never reached
        """
        if len(success_history) < window:
            return None
            
        for i in range(window, len(success_history) + 1):
            rolling_success = sum(success_history[i-window:i]) / window
            if rolling_success >= threshold:
                return i
        
        return None
    
    @staticmethod
    def calculate_learning_speed(
        success_history: List[int],
        window: int = 10
    ) -> float:
        """
        Calculate learning speed as area under the success curve
        Higher is better (faster learning)
        
        Args:
            success_history: List of 0/1 success values
            window: Smoothing window
            
        Returns:
            Learning speed score (0-1, higher is better)
        """
        if not success_history:
            return 0.0
        
        # Calculate cumulative success
        cumulative = []
        for i in range(len(success_history)):
            cumulative.append(sum(success_history[:i+1]) / (i+1))
        
        # Area under curve normalized by max possible area
        auc = sum(cumulative) / len(cumulative)
        return auc
    
    @staticmethod
    def calculate_stability(
        rewards: List[float],
        window: int = 20
    ) -> float:
        """
        Calculate training stability as inverse of variance in later episodes
        Higher is better (more stable)
        
        Args:
            rewards: Episode rewards
            window: Window for stability calculation
            
        Returns:
            Stability score (higher is better)
        """
        if len(rewards) < window:
            return 0.0
        
        # Use last portion of training
        later_rewards = rewards[-window:]
        variance = np.var(later_rewards)
        
        # Convert to stability score (inverse, normalized)
        if variance == 0:
            return 1.0
        stability = 1.0 / (1.0 + variance)
        return stability
    
    @staticmethod
    def compare_runs(run1: TrainingRun, run2: TrainingRun) -> ComparisonMetrics:
        """
        Compare two training runs
        
        Args:
            run1: First run (typically baseline, e.g., DQN)
            run2: Second run (typically experimental, e.g., DDQ)
            
        Returns:
            ComparisonMetrics object
        """
        # Basic differences
        success_diff = run2.success_rate - run1.success_rate
        if run1.success_rate > 0:
            success_improvement = success_diff / run1.success_rate
        else:
            success_improvement = float('inf') if success_diff > 0 else 0.0
        
        reward_diff = run2.avg_reward - run1.avg_reward
        if abs(run1.avg_reward) > 0.01:
            reward_improvement = reward_diff / abs(run1.avg_reward)
        else:
            reward_improvement = float('inf') if reward_diff > 0 else 0.0
        
        # Sample efficiency (episodes to 50% success)
        ep_50_run1 = MetricsCalculator.episodes_to_threshold(run1.success_history, 0.5)
        ep_50_run2 = MetricsCalculator.episodes_to_threshold(run2.success_history, 0.5)
        
        if ep_50_run1 and ep_50_run2:
            efficiency_gain = (ep_50_run1 - ep_50_run2) / ep_50_run1
        else:
            efficiency_gain = None
        
        # Final performance (last 10 episodes)
        final_window = min(10, len(run1.success_history), len(run2.success_history))
        final_10_run1 = np.mean(run1.success_history[-final_window:]) if run1.success_history else 0
        final_10_run2 = np.mean(run2.success_history[-final_window:]) if run2.success_history else 0
        
        # Statistical test (if enough data)
        t_stat, p_val = None, None
        if len(run1.episode_rewards) >= 20 and len(run2.episode_rewards) >= 20:
            try:
                from scipy import stats
                # Compare last 20 episodes
                r1 = run1.episode_rewards[-20:]
                r2 = run2.episode_rewards[-20:]
                t_stat, p_val = stats.ttest_ind(r1, r2)
            except ImportError:
                pass  # scipy not available
        
        return ComparisonMetrics(
            success_rate_diff=success_diff,
            success_rate_improvement=success_improvement,
            reward_diff=reward_diff,
            reward_improvement=reward_improvement,
            episodes_to_50_success_run1=ep_50_run1,
            episodes_to_50_success_run2=ep_50_run2,
            sample_efficiency_gain=efficiency_gain,
            final_10_success_run1=final_10_run1,
            final_10_success_run2=final_10_run2,
            t_statistic=t_stat,
            p_value=p_val
        )
    
    @staticmethod
    def print_comparison(
        run1: TrainingRun, 
        run2: TrainingRun,
        metrics: ComparisonMetrics = None
    ):
        """Pretty print comparison between two runs"""
        if metrics is None:
            metrics = MetricsCalculator.compare_runs(run1, run2)
        
        print("\n" + "="*70)
        print(f"COMPARISON: {run1.display_name} vs {run2.display_name}")
        print("="*70)
        
        print(f"\n{'Metric':<30} {run1.display_name:>15} {run2.display_name:>15} {'Diff':>10}")
        print("-"*70)
        
        print(f"{'Success Rate':<30} {run1.success_rate:>14.1%} {run2.success_rate:>14.1%} {metrics.success_rate_diff:>+9.1%}")
        print(f"{'Avg Reward':<30} {run1.avg_reward:>15.2f} {run2.avg_reward:>15.2f} {metrics.reward_diff:>+10.2f}")
        print(f"{'Avg Episode Length':<30} {run1.avg_length:>15.1f} {run2.avg_length:>15.1f}")
        
        print(f"\n{'Episodes to 50% Success':<30}", end="")
        ep1 = metrics.episodes_to_50_success_run1
        ep2 = metrics.episodes_to_50_success_run2
        print(f" {str(ep1) if ep1 else 'N/A':>15} {str(ep2) if ep2 else 'N/A':>15}", end="")
        if metrics.sample_efficiency_gain:
            print(f" {metrics.sample_efficiency_gain:>+9.1%}")
        else:
            print()
        
        print(f"{'Final 10-ep Success':<30} {metrics.final_10_success_run1:>14.1%} {metrics.final_10_success_run2:>14.1%}")
        
        if metrics.p_value is not None:
            sig = "**" if metrics.p_value < 0.01 else "*" if metrics.p_value < 0.05 else ""
            print(f"\n{'Statistical Significance':<30} p={metrics.p_value:.4f} {sig}")
        
        print("\n" + "="*70)
        
        # Summary
        if metrics.success_rate_diff > 0:
            print(f"\n✓ {run2.display_name} outperforms {run1.display_name} by {metrics.success_rate_improvement:.1%}")
        elif metrics.success_rate_diff < 0:
            print(f"\n✗ {run1.display_name} outperforms {run2.display_name} by {-metrics.success_rate_improvement:.1%}")
        else:
            print(f"\n= Both algorithms perform similarly")
    
    @staticmethod
    def generate_summary_table(runs: List[TrainingRun]) -> str:
        """Generate a markdown summary table of all runs"""
        if not runs:
            return "No runs to summarize."
        
        lines = []
        lines.append("| Algorithm | Episodes | Success Rate | Avg Reward | Avg Length | K | Timestamp |")
        lines.append("|-----------|----------|--------------|------------|------------|---|-----------|")
        
        for run in runs:
            k = run.hyperparameters.get('K', 'N/A')
            lines.append(
                f"| {run.algorithm.upper()} | {run.num_episodes} | "
                f"{run.success_rate:.1%} | {run.avg_reward:.2f} | "
                f"{run.avg_length:.1f} | {k} | {run.timestamp} |"
            )
        
        return "\n".join(lines)
