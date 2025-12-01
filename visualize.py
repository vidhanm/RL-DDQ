"""
Visualization Script
Generate comprehensive visualizations from training runs

Usage:
    python visualize.py                    # Auto-discover and plot all runs
    python visualize.py --compare          # Compare DQN vs DDQ
    python visualize.py --ablation         # Compare different K values
    python visualize.py --run FILE         # Visualize specific run
    python visualize.py --summary          # Print summary table only
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.history_loader import HistoryLoader, TrainingRun
from analysis.metrics import MetricsCalculator, ComparisonMetrics
from analysis.plot_utils import (
    STYLE, create_figure, save_figure, smooth_data,
    add_confidence_band, format_percentage_axis,
    ACTION_NAMES, PERSONA_NAMES, get_action_color
)


class Visualizer:
    """Main visualization class"""
    
    def __init__(self, checkpoint_dir: str = "checkpoints", output_dir: str = "figures"):
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        self.loader = HistoryLoader(checkpoint_dir)
        self.runs: List[TrainingRun] = []
        
    def load_runs(self) -> List[TrainingRun]:
        """Load all training runs"""
        self.runs = self.loader.discover_and_load()
        return self.runs
    
    def plot_single_run(self, run: TrainingRun, save: bool = True):
        """
        Create comprehensive visualization for a single training run
        
        Generates:
        - Learning curve (reward over episodes)
        - Success rate curve
        - Loss curve
        - Episode length distribution
        """
        fig, axes = create_figure(2, 2)
        fig.suptitle(f"Training Analysis: {run.display_name} ({run.num_episodes} episodes)", 
                     fontsize=14, fontweight='bold')
        
        episodes = np.arange(1, len(run.episode_rewards) + 1)
        color = STYLE.get_color(run.algorithm)
        
        # Plot 1: Reward curve
        ax = axes[0, 0]
        ax.plot(episodes, run.episode_rewards, alpha=0.3, color=color, label='Raw')
        smoothed = smooth_data(run.episode_rewards, 10)
        ax.plot(episodes, smoothed, color=color, linewidth=2, label='Smoothed (10-ep)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend(loc='lower right')
        
        # Plot 2: Success rate
        ax = axes[0, 1]
        if run.success_history:
            ax.plot(episodes, run.success_history, alpha=0.3, color=color, label='Raw')
            smoothed_success = smooth_data(run.success_history, 10)
            ax.plot(episodes, smoothed_success, color=color, linewidth=2, label='Smoothed')
            cumulative = run.get_cumulative_success()
            ax.plot(episodes, cumulative, color='gray', linestyle='--', label='Cumulative')
            ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='50% target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='lower right')
        
        # Plot 3: Loss curve
        ax = axes[1, 0]
        if run.loss_history:
            loss_episodes = np.arange(1, len(run.loss_history) + 1)
            ax.plot(loss_episodes, run.loss_history, alpha=0.5, color=color)
            smoothed_loss = smooth_data(run.loss_history, 10)
            ax.plot(loss_episodes[:len(smoothed_loss)], smoothed_loss, 
                   color=color, linewidth=2, label='Smoothed')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend(loc='upper right')
        else:
            ax.text(0.5, 0.5, 'No loss data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            ax.set_title('Training Loss')
        
        # Plot 4: Episode length distribution
        ax = axes[1, 1]
        if run.episode_lengths:
            ax.hist(run.episode_lengths, bins=15, color=color, alpha=0.7, edgecolor='white')
            ax.axvline(x=np.mean(run.episode_lengths), color='red', linestyle='--',
                      label=f'Mean: {np.mean(run.episode_lengths):.1f}')
            ax.set_xlabel('Episode Length (turns)')
            ax.set_ylabel('Frequency')
            ax.set_title('Episode Length Distribution')
            ax.legend()
        
        plt.tight_layout()
        
        if save:
            filename = f"{run.algorithm}_{run.num_episodes}ep_{run.timestamp}"
            save_figure(fig, filename, self.output_dir)
        
        return fig
    
    def plot_comparison(
        self, 
        run1: TrainingRun, 
        run2: TrainingRun,
        save: bool = True
    ):
        """
        Create side-by-side comparison of two runs
        
        Args:
            run1: First run (typically baseline DQN)
            run2: Second run (typically DDQ)
        """
        fig, axes = create_figure(2, 2)
        fig.suptitle(f"Comparison: {run1.display_name} vs {run2.display_name}", 
                     fontsize=14, fontweight='bold')
        
        episodes1 = np.arange(1, len(run1.episode_rewards) + 1)
        episodes2 = np.arange(1, len(run2.episode_rewards) + 1)
        
        color1 = STYLE.get_color(run1.algorithm)
        color2 = STYLE.get_color(run2.algorithm)
        
        # Plot 1: Reward comparison
        ax = axes[0, 0]
        smooth1 = smooth_data(run1.episode_rewards, 10)
        smooth2 = smooth_data(run2.episode_rewards, 10)
        ax.plot(episodes1, smooth1, color=color1, linewidth=2, label=run1.display_name)
        ax.plot(episodes2, smooth2, color=color2, linewidth=2, label=run2.display_name)
        add_confidence_band(ax, episodes1, np.array(run1.episode_rewards), color1)
        add_confidence_band(ax, episodes2, np.array(run2.episode_rewards), color2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Comparison (smoothed)')
        ax.legend()
        
        # Plot 2: Success rate comparison
        ax = axes[0, 1]
        if run1.success_history and run2.success_history:
            smooth_s1 = smooth_data(run1.success_history, 10)
            smooth_s2 = smooth_data(run2.success_history, 10)
            ax.plot(episodes1[:len(smooth_s1)], smooth_s1, color=color1, 
                   linewidth=2, label=run1.display_name)
            ax.plot(episodes2[:len(smooth_s2)], smooth_s2, color=color2, 
                   linewidth=2, label=run2.display_name)
            ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='50% target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        
        # Plot 3: Cumulative success (sample efficiency)
        ax = axes[1, 0]
        cum1 = run1.get_cumulative_success()
        cum2 = run2.get_cumulative_success()
        if cum1 and cum2:
            ax.plot(episodes1[:len(cum1)], cum1, color=color1, 
                   linewidth=2, label=run1.display_name)
            ax.plot(episodes2[:len(cum2)], cum2, color=color2, 
                   linewidth=2, label=run2.display_name)
            ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.7)
            
            # Mark episodes to reach 50%
            metrics = MetricsCalculator.compare_runs(run1, run2)
            if metrics.episodes_to_50_success_run1:
                ax.axvline(x=metrics.episodes_to_50_success_run1, color=color1, 
                          linestyle='--', alpha=0.7)
            if metrics.episodes_to_50_success_run2:
                ax.axvline(x=metrics.episodes_to_50_success_run2, color=color2,
                          linestyle='--', alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative Success Rate')
        ax.set_title('Sample Efficiency')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        
        # Plot 4: Final performance bar chart
        ax = axes[1, 1]
        metrics = ['Success\nRate', 'Avg\nReward', 'Final 10\nSuccess']
        x = np.arange(len(metrics))
        width = 0.35
        
        vals1 = [run1.success_rate, run1.avg_reward / 20, 
                 np.mean(run1.success_history[-10:]) if len(run1.success_history) >= 10 else run1.success_rate]
        vals2 = [run2.success_rate, run2.avg_reward / 20,
                 np.mean(run2.success_history[-10:]) if len(run2.success_history) >= 10 else run2.success_rate]
        
        bars1 = ax.bar(x - width/2, vals1, width, label=run1.display_name, color=color1)
        bars2 = ax.bar(x + width/2, vals2, width, label=run2.display_name, color=color2)
        
        ax.set_ylabel('Score')
        ax.set_title('Final Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filename = f"comparison_{run1.algorithm}_vs_{run2.algorithm}"
            save_figure(fig, filename, self.output_dir)
        
        return fig
    
    def plot_ablation(
        self, 
        runs: List[TrainingRun],
        group_by: str = 'K',
        save: bool = True
    ):
        """
        Create ablation study visualization
        
        Args:
            runs: List of runs to compare
            group_by: Hyperparameter to group by ('K', 'learning_rate', etc.)
        """
        if len(runs) < 2:
            print("[WARN] Need at least 2 runs for ablation study")
            return None
        
        fig, axes = create_figure(1, 2, figsize=(14, 5))
        fig.suptitle(f"Ablation Study: Effect of {group_by}", 
                     fontsize=14, fontweight='bold')
        
        # Get colors for different parameter values
        param_values = []
        for run in runs:
            val = run.hyperparameters.get(group_by, 'unknown')
            if val not in param_values:
                param_values.append(val)
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(param_values)))
        color_map = {val: colors[i] for i, val in enumerate(param_values)}
        
        # Plot 1: Success rate curves
        ax = axes[0]
        for run in runs:
            val = run.hyperparameters.get(group_by, 'unknown')
            episodes = np.arange(1, len(run.success_history) + 1)
            smoothed = smooth_data(run.success_history, 10)
            ax.plot(episodes[:len(smoothed)], smoothed, 
                   color=color_map[val], linewidth=2, 
                   label=f'{group_by}={val}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate by Parameter Value')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        
        # Plot 2: Final performance comparison
        ax = axes[1]
        x = np.arange(len(runs))
        success_rates = [run.success_rate for run in runs]
        colors_bars = [color_map[run.hyperparameters.get(group_by, 'unknown')] for run in runs]
        labels = [f'{group_by}={run.hyperparameters.get(group_by, "?")}' for run in runs]
        
        bars = ax.bar(x, success_rates, color=colors_bars)
        ax.set_ylabel('Success Rate')
        ax.set_title('Final Success Rate by Parameter')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, val in zip(bars, success_rates):
            ax.annotate(f'{val:.1%}',
                       xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            filename = f"ablation_{group_by}"
            save_figure(fig, filename, self.output_dir)
        
        return fig
    
    def plot_all_runs_overview(self, save: bool = True):
        """Create overview of all training runs"""
        if not self.runs:
            print("[WARN] No runs to visualize")
            return None
        
        n_runs = len(self.runs)
        fig, axes = create_figure(1, 2, figsize=(14, 5))
        fig.suptitle(f"Training Runs Overview ({n_runs} runs)", 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: All success curves
        ax = axes[0]
        for run in self.runs:
            episodes = np.arange(1, len(run.success_history) + 1)
            smoothed = smooth_data(run.success_history, 10)
            color = STYLE.get_color(run.algorithm)
            ax.plot(episodes[:len(smoothed)], smoothed, 
                   color=color, linewidth=2, alpha=0.7,
                   label=run.display_name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Success Rate Comparison')
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        
        # Plot 2: Summary bar chart
        ax = axes[1]
        x = np.arange(n_runs)
        success_rates = [run.success_rate for run in self.runs]
        colors = [STYLE.get_color(run.algorithm) for run in self.runs]
        labels = [run.display_name for run in self.runs]
        
        bars = ax.bar(x, success_rates, color=colors)
        ax.set_ylabel('Success Rate')
        ax.set_title('Final Success Rate')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        
        for bar, val in zip(bars, success_rates):
            ax.annotate(f'{val:.1%}',
                       xy=(bar.get_x() + bar.get_width()/2, val),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save:
            save_figure(fig, "overview_all_runs", self.output_dir)
        
        return fig
    
    def generate_report(self, save: bool = True):
        """Generate comprehensive report with all visualizations"""
        print("\n" + "="*70)
        print("GENERATING VISUALIZATION REPORT")
        print("="*70)
        
        if not self.runs:
            self.load_runs()
        
        if not self.runs:
            print("[ERROR] No training runs found in", self.checkpoint_dir)
            return
        
        print(f"\nFound {len(self.runs)} training run(s)")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Overview
        print("\n[1/4] Generating overview...")
        self.plot_all_runs_overview(save=save)
        
        # 2. Individual runs
        print("[2/4] Generating individual run plots...")
        for run in self.runs:
            self.plot_single_run(run, save=save)
        
        # 3. DQN vs DDQ comparison (if both exist)
        print("[3/4] Generating comparisons...")
        dqn_runs = self.loader.get_dqn_runs()
        ddq_runs = self.loader.get_ddq_runs()
        
        if dqn_runs and ddq_runs:
            # Compare latest of each
            dqn_latest = max(dqn_runs, key=lambda r: r.timestamp)
            ddq_latest = max(ddq_runs, key=lambda r: r.timestamp)
            self.plot_comparison(dqn_latest, ddq_latest, save=save)
            
            # Print metrics
            MetricsCalculator.print_comparison(dqn_latest, ddq_latest)
        else:
            print("  [SKIP] Need both DQN and DDQ runs for comparison")
        
        # 4. Ablation study (if multiple DDQ runs with different K)
        print("[4/4] Checking for ablation study data...")
        k_values = set(r.hyperparameters.get('K') for r in ddq_runs if r.hyperparameters.get('K'))
        if len(k_values) > 1:
            self.plot_ablation(ddq_runs, group_by='K', save=save)
        else:
            print("  [SKIP] Need DDQ runs with different K values for ablation")
        
        print("\n" + "="*70)
        print(f"REPORT COMPLETE - Figures saved to: {self.output_dir}/")
        print("="*70)
        
        # Print summary
        self.loader.print_summary()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Visualize training runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize.py                    # Generate full report
    python visualize.py --summary          # Print summary only
    python visualize.py --compare          # Compare DQN vs DDQ
    python visualize.py --show             # Show plots interactively
        """
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing training history files')
    parser.add_argument('--output-dir', type=str, default='figures',
                        help='Directory to save figures')
    parser.add_argument('--summary', action='store_true',
                        help='Print summary table only (no plots)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate DQN vs DDQ comparison only')
    parser.add_argument('--ablation', action='store_true',
                        help='Generate ablation study plots only')
    parser.add_argument('--show', action='store_true',
                        help='Show plots interactively (don\'t just save)')
    parser.add_argument('--run', type=str,
                        help='Visualize specific run file')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = Visualizer(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir
    )
    
    # Load runs
    viz.load_runs()
    
    if args.summary:
        # Summary only
        viz.loader.print_summary()
        return
    
    if args.run:
        # Specific run
        for run in viz.runs:
            if args.run in run.filepath:
                viz.plot_single_run(run, save=not args.show)
                if args.show:
                    plt.show()
                return
        print(f"[ERROR] Run not found: {args.run}")
        return
    
    if args.compare:
        # Comparison only
        dqn_runs = viz.loader.get_dqn_runs()
        ddq_runs = viz.loader.get_ddq_runs()
        if dqn_runs and ddq_runs:
            viz.plot_comparison(
                max(dqn_runs, key=lambda r: r.timestamp),
                max(ddq_runs, key=lambda r: r.timestamp),
                save=not args.show
            )
            MetricsCalculator.print_comparison(
                max(dqn_runs, key=lambda r: r.timestamp),
                max(ddq_runs, key=lambda r: r.timestamp)
            )
            if args.show:
                plt.show()
        else:
            print("[ERROR] Need both DQN and DDQ runs for comparison")
        return
    
    if args.ablation:
        # Ablation only
        ddq_runs = viz.loader.get_ddq_runs()
        if len(ddq_runs) >= 2:
            viz.plot_ablation(ddq_runs, group_by='K', save=not args.show)
            if args.show:
                plt.show()
        else:
            print("[ERROR] Need at least 2 DDQ runs with different K values")
        return
    
    # Full report
    viz.generate_report(save=True)
    
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
