"""
History Loader
Load and parse training history files from checkpoints directory
Supports both old format (flat JSON) and new format (with metadata)
"""

import os
import json
import glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class TrainingRun:
    """Represents a single training run with all its data and metadata"""
    
    # Metadata
    algorithm: str
    num_episodes: int
    timestamp: str
    filepath: str
    
    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Training data
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_history: List[int] = field(default_factory=list)
    loss_history: List[float] = field(default_factory=list)
    
    # Optional: action history (if recorded)
    action_history: List[int] = field(default_factory=list)
    
    # Optional: per-episode details
    persona_history: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate"""
        if not self.success_history:
            return 0.0
        return sum(self.success_history) / len(self.success_history)
    
    @property
    def avg_reward(self) -> float:
        """Calculate average reward"""
        if not self.episode_rewards:
            return 0.0
        return sum(self.episode_rewards) / len(self.episode_rewards)
    
    @property
    def avg_length(self) -> float:
        """Calculate average episode length"""
        if not self.episode_lengths:
            return 0.0
        return sum(self.episode_lengths) / len(self.episode_lengths)
    
    @property
    def final_loss(self) -> float:
        """Get final training loss"""
        if not self.loss_history:
            return 0.0
        return self.loss_history[-1]
    
    @property 
    def display_name(self) -> str:
        """Human-readable name for this run"""
        k_value = self.hyperparameters.get('K', 'N/A')
        if self.algorithm.lower() == 'ddq':
            return f"DDQ (K={k_value})"
        return self.algorithm.upper()
    
    def get_smoothed_rewards(self, window: int = 10) -> List[float]:
        """Get smoothed rewards using moving average"""
        return self._smooth(self.episode_rewards, window)
    
    def get_smoothed_success(self, window: int = 10) -> List[float]:
        """Get smoothed success rate using moving average"""
        return self._smooth(self.success_history, window)
    
    def get_cumulative_success(self) -> List[float]:
        """Get cumulative success rate over episodes"""
        if not self.success_history:
            return []
        cumulative = []
        for i in range(len(self.success_history)):
            cumulative.append(sum(self.success_history[:i+1]) / (i+1))
        return cumulative
    
    @staticmethod
    def _smooth(data: List[float], window: int) -> List[float]:
        """Apply moving average smoothing"""
        if not data or len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(sum(data[start:i+1]) / (i - start + 1))
        return smoothed


class HistoryLoader:
    """
    Load and manage training history files
    
    Supports:
    - New format with metadata (algorithm_Nep_timestamp.json)
    - Old format without metadata (training_history.json, dqn_100ep_old.json)
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """
        Initialize loader
        
        Args:
            checkpoint_dir: Directory containing history files
        """
        self.checkpoint_dir = checkpoint_dir
        self.runs: List[TrainingRun] = []
        
    def discover_and_load(self) -> List[TrainingRun]:
        """
        Discover all training history files and load them
        
        Returns:
            List of TrainingRun objects
        """
        self.runs = []
        
        # Find all JSON files in checkpoint directory
        pattern = os.path.join(self.checkpoint_dir, "*.json")
        json_files = glob.glob(pattern)
        
        for filepath in json_files:
            try:
                run = self._load_file(filepath)
                if run is not None:
                    self.runs.append(run)
            except Exception as e:
                print(f"[WARN] Failed to load {filepath}: {e}")
        
        # Sort by algorithm then timestamp
        self.runs.sort(key=lambda r: (r.algorithm, r.timestamp))
        
        return self.runs
    
    def _load_file(self, filepath: str) -> Optional[TrainingRun]:
        """Load a single history file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        filename = os.path.basename(filepath)
        
        # Check if new format (has metadata)
        if 'metadata' in data:
            return self._parse_new_format(data, filepath)
        else:
            return self._parse_old_format(data, filepath, filename)
    
    def _parse_new_format(self, data: Dict, filepath: str) -> TrainingRun:
        """Parse new format with metadata"""
        metadata = data['metadata']
        
        return TrainingRun(
            algorithm=metadata.get('algorithm', 'unknown'),
            num_episodes=metadata.get('num_episodes', len(data.get('episode_rewards', []))),
            timestamp=metadata.get('timestamp', 'unknown'),
            filepath=filepath,
            hyperparameters=metadata.get('hyperparameters', {}),
            episode_rewards=data.get('episode_rewards', []),
            episode_lengths=data.get('episode_lengths', []),
            success_history=data.get('success_history', []),
            loss_history=data.get('loss_history', []),
            action_history=data.get('action_history', []),
            persona_history=data.get('persona_history', [])
        )
    
    def _parse_old_format(self, data: Dict, filepath: str, filename: str) -> TrainingRun:
        """Parse old format without metadata - infer from filename"""
        
        # Try to infer algorithm and episodes from filename
        # Examples: dqn_100ep_old.json, training_history.json
        algorithm = 'unknown'
        num_episodes = len(data.get('episode_rewards', []))
        
        filename_lower = filename.lower()
        if 'dqn' in filename_lower and 'ddq' not in filename_lower:
            algorithm = 'dqn'
        elif 'ddq' in filename_lower:
            algorithm = 'ddq'
        
        # Try to extract episode count from filename
        import re
        ep_match = re.search(r'(\d+)ep', filename_lower)
        if ep_match:
            num_episodes = int(ep_match.group(1))
        
        # Use file modification time as timestamp
        try:
            mtime = os.path.getmtime(filepath)
            timestamp = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
        except:
            timestamp = "unknown"
        
        return TrainingRun(
            algorithm=algorithm,
            num_episodes=num_episodes,
            timestamp=timestamp,
            filepath=filepath,
            hyperparameters={},
            episode_rewards=data.get('episode_rewards', []),
            episode_lengths=data.get('episode_lengths', []),
            success_history=data.get('success_history', []),
            loss_history=data.get('loss_history', [])
        )
    
    def get_by_algorithm(self, algorithm: str) -> List[TrainingRun]:
        """Get all runs for a specific algorithm"""
        return [r for r in self.runs if r.algorithm.lower() == algorithm.lower()]
    
    def get_dqn_runs(self) -> List[TrainingRun]:
        """Get all DQN runs"""
        return self.get_by_algorithm('dqn')
    
    def get_ddq_runs(self) -> List[TrainingRun]:
        """Get all DDQ runs"""
        return self.get_by_algorithm('ddq')
    
    def get_latest(self, algorithm: str = None) -> Optional[TrainingRun]:
        """Get the most recent run (optionally filtered by algorithm)"""
        runs = self.runs if algorithm is None else self.get_by_algorithm(algorithm)
        if not runs:
            return None
        return max(runs, key=lambda r: r.timestamp)
    
    def get_by_k_value(self, k: int) -> List[TrainingRun]:
        """Get DDQ runs with specific K value"""
        return [
            r for r in self.runs 
            if r.algorithm.lower() == 'ddq' and r.hyperparameters.get('K') == k
        ]
    
    def print_summary(self):
        """Print summary of all loaded runs"""
        print("\n" + "="*70)
        print("TRAINING RUNS SUMMARY")
        print("="*70)
        
        if not self.runs:
            print("No training runs found.")
            return
        
        print(f"\nFound {len(self.runs)} training run(s):\n")
        
        for i, run in enumerate(self.runs, 1):
            print(f"{i}. {run.display_name}")
            print(f"   Episodes: {run.num_episodes}")
            print(f"   Success Rate: {run.success_rate:.1%}")
            print(f"   Avg Reward: {run.avg_reward:.2f}")
            print(f"   Timestamp: {run.timestamp}")
            print(f"   File: {os.path.basename(run.filepath)}")
            if run.hyperparameters:
                k = run.hyperparameters.get('K', 'N/A')
                lr = run.hyperparameters.get('learning_rate', 'N/A')
                print(f"   K={k}, LR={lr}")
            print()
        
        print("="*70)


# Convenience function
def load_all_runs(checkpoint_dir: str = "checkpoints") -> List[TrainingRun]:
    """Load all training runs from checkpoint directory"""
    loader = HistoryLoader(checkpoint_dir)
    return loader.discover_and_load()
