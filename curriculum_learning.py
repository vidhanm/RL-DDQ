"""
Curriculum Learning for Debt Collection Agent

Progressive training from easy to hard scenarios:
- Stage 1: Cooperative debtors (high compliance)
- Stage 2: Add Sad/Overwhelmed debtors
- Stage 3: Add Avoidant debtors
- Stage 4: Add Angry debtors (most challenging)

Features:
1. CurriculumScheduler - Controls persona distribution
2. AdaptiveCurriculum - Auto-advances based on performance
3. CurriculumTrainer - Wraps training loop with curriculum
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CURRICULUM STAGES
# ============================================================================

class CurriculumStage(Enum):
    """Training curriculum stages from easiest to hardest"""
    STAGE_1 = 1  # Cooperative only
    STAGE_2 = 2  # Cooperative + Sad/Overwhelmed
    STAGE_3 = 3  # + Avoidant
    STAGE_4 = 4  # + Angry (full difficulty)


@dataclass
class StageConfig:
    """Configuration for a curriculum stage"""
    stage: CurriculumStage
    personas: List[str]
    persona_weights: List[float]  # Sampling probabilities
    min_episodes: int = 50  # Minimum episodes before advancing
    success_threshold: float = 0.6  # Required success rate to advance
    description: str = ""


# Default curriculum configuration
DEFAULT_CURRICULUM = {
    CurriculumStage.STAGE_1: StageConfig(
        stage=CurriculumStage.STAGE_1,
        personas=["cooperative"],
        persona_weights=[1.0],
        min_episodes=30,
        success_threshold=0.7,
        description="Training on cooperative debtors only"
    ),
    CurriculumStage.STAGE_2: StageConfig(
        stage=CurriculumStage.STAGE_2,
        personas=["cooperative", "sad_overwhelmed"],
        persona_weights=[0.6, 0.4],
        min_episodes=50,
        success_threshold=0.6,
        description="Adding sad/overwhelmed debtors"
    ),
    CurriculumStage.STAGE_3: StageConfig(
        stage=CurriculumStage.STAGE_3,
        personas=["cooperative", "sad_overwhelmed", "avoidant"],
        persona_weights=[0.4, 0.3, 0.3],
        min_episodes=75,
        success_threshold=0.55,
        description="Adding avoidant debtors"
    ),
    CurriculumStage.STAGE_4: StageConfig(
        stage=CurriculumStage.STAGE_4,
        personas=["cooperative", "sad_overwhelmed", "avoidant", "angry"],
        persona_weights=[0.25, 0.25, 0.25, 0.25],
        min_episodes=100,
        success_threshold=0.5,
        description="Full difficulty with all personas"
    ),
}


# ============================================================================
# CURRICULUM SCHEDULER
# ============================================================================

class CurriculumScheduler:
    """
    Controls the training curriculum
    
    Manages:
    - Current stage and persona distribution
    - Stage advancement logic
    - Performance tracking per stage
    """
    
    def __init__(
        self,
        curriculum: Dict[CurriculumStage, StageConfig] = None,
        start_stage: CurriculumStage = CurriculumStage.STAGE_1,
        auto_advance: bool = True,
        window_size: int = 20  # Episodes to consider for success rate
    ):
        self.curriculum = curriculum or DEFAULT_CURRICULUM
        self.current_stage = start_stage
        self.auto_advance = auto_advance
        self.window_size = window_size
        
        # Tracking
        self.stage_history: Dict[CurriculumStage, List[bool]] = {
            stage: [] for stage in CurriculumStage
        }
        self.episodes_in_stage = 0
        self.total_episodes = 0
        self.stage_transitions: List[Tuple[int, CurriculumStage]] = [(0, start_stage)]
        
        logger.info(f"Curriculum initialized at {start_stage.name}")
    
    @property
    def current_config(self) -> StageConfig:
        """Get current stage configuration"""
        return self.curriculum[self.current_stage]
    
    def sample_persona(self) -> str:
        """Sample a persona based on current stage distribution"""
        config = self.current_config
        idx = np.random.choice(len(config.personas), p=config.persona_weights)
        return config.personas[idx]
    
    def get_persona_distribution(self) -> Dict[str, float]:
        """Get current persona probability distribution"""
        config = self.current_config
        return dict(zip(config.personas, config.persona_weights))
    
    def record_episode(self, success: bool) -> Optional[CurriculumStage]:
        """
        Record episode outcome and potentially advance stage
        
        Args:
            success: Whether the episode was successful
            
        Returns:
            New stage if advanced, None otherwise
        """
        self.stage_history[self.current_stage].append(success)
        self.episodes_in_stage += 1
        self.total_episodes += 1
        
        # Check for advancement
        if self.auto_advance and self._should_advance():
            new_stage = self._advance_stage()
            return new_stage
        
        return None
    
    def _should_advance(self) -> bool:
        """Check if should advance to next stage"""
        config = self.current_config
        
        # Can't advance from final stage
        if self.current_stage == CurriculumStage.STAGE_4:
            return False
        
        # Need minimum episodes
        if self.episodes_in_stage < config.min_episodes:
            return False
        
        # Check success rate in window
        recent_history = self.stage_history[self.current_stage][-self.window_size:]
        if len(recent_history) < self.window_size:
            return False
        
        success_rate = sum(recent_history) / len(recent_history)
        return success_rate >= config.success_threshold
    
    def _advance_stage(self) -> CurriculumStage:
        """Advance to next curriculum stage"""
        stage_order = list(CurriculumStage)
        current_idx = stage_order.index(self.current_stage)
        
        if current_idx < len(stage_order) - 1:
            new_stage = stage_order[current_idx + 1]
            
            logger.info(f"Advancing curriculum: {self.current_stage.name} → {new_stage.name}")
            logger.info(f"  After {self.episodes_in_stage} episodes")
            
            self.current_stage = new_stage
            self.episodes_in_stage = 0
            self.stage_transitions.append((self.total_episodes, new_stage))
            
            return new_stage
        
        return self.current_stage
    
    def force_stage(self, stage: CurriculumStage):
        """Force transition to a specific stage"""
        logger.info(f"Forcing stage transition to {stage.name}")
        self.current_stage = stage
        self.episodes_in_stage = 0
        self.stage_transitions.append((self.total_episodes, stage))
    
    def get_success_rate(self, stage: CurriculumStage = None, window: int = None) -> float:
        """Get success rate for a stage"""
        stage = stage or self.current_stage
        window = window or self.window_size
        
        history = self.stage_history[stage]
        if not history:
            return 0.0
        
        recent = history[-window:]
        return sum(recent) / len(recent)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get curriculum statistics"""
        stats = {
            'current_stage': self.current_stage.name,
            'current_stage_episodes': self.episodes_in_stage,
            'total_episodes': self.total_episodes,
            'stage_transitions': [
                (ep, stage.name) for ep, stage in self.stage_transitions
            ],
        }
        
        # Add per-stage stats
        for stage in CurriculumStage:
            history = self.stage_history[stage]
            if history:
                stats[f'{stage.name}_episodes'] = len(history)
                stats[f'{stage.name}_success_rate'] = sum(history) / len(history)
        
        return stats
    
    def print_status(self):
        """Print current curriculum status"""
        config = self.current_config
        print("\n" + "="*50)
        print("CURRICULUM STATUS")
        print("="*50)
        print(f"Current Stage: {self.current_stage.name}")
        print(f"Description: {config.description}")
        print(f"Episodes in stage: {self.episodes_in_stage}/{config.min_episodes}")
        print(f"Current success rate: {self.get_success_rate():.1%}")
        print(f"Required for advancement: {config.success_threshold:.1%}")
        print(f"\nPersona distribution:")
        for persona, weight in zip(config.personas, config.persona_weights):
            print(f"  {persona}: {weight:.0%}")
        print("="*50)


# ============================================================================
# ADAPTIVE CURRICULUM
# ============================================================================

class AdaptiveCurriculum(CurriculumScheduler):
    """
    Curriculum that adapts persona weights based on performance
    
    Features:
    - Increases weight of difficult personas as agent improves
    - Can regress if performance drops
    - Smooth weight transitions
    """
    
    def __init__(
        self,
        curriculum: Dict[CurriculumStage, StageConfig] = None,
        start_stage: CurriculumStage = CurriculumStage.STAGE_1,
        adaptation_rate: float = 0.1,
        regression_enabled: bool = True,
        regression_threshold: float = 0.3
    ):
        super().__init__(curriculum, start_stage, auto_advance=True)
        self.adaptation_rate = adaptation_rate
        self.regression_enabled = regression_enabled
        self.regression_threshold = regression_threshold
        
        # Track per-persona performance
        self.persona_success: Dict[str, deque] = {}
        for stage_config in self.curriculum.values():
            for persona in stage_config.personas:
                if persona not in self.persona_success:
                    self.persona_success[persona] = deque(maxlen=50)
    
    def record_episode_with_persona(
        self, 
        success: bool, 
        persona: str
    ) -> Optional[CurriculumStage]:
        """Record episode with persona information"""
        # Track persona-specific performance
        if persona in self.persona_success:
            self.persona_success[persona].append(success)
        
        # Adapt weights
        self._adapt_weights()
        
        # Check for regression
        if self.regression_enabled:
            self._check_regression()
        
        # Normal advancement check
        return self.record_episode(success)
    
    def _adapt_weights(self):
        """Adapt persona weights based on performance"""
        config = self.current_config
        
        # Calculate performance for each persona
        performances = {}
        for persona in config.personas:
            history = self.persona_success.get(persona, [])
            if len(history) >= 10:
                performances[persona] = sum(history) / len(history)
            else:
                performances[persona] = 0.5  # Default
        
        # Increase weight of harder personas (lower performance)
        # This creates a natural difficulty progression
        new_weights = []
        for i, persona in enumerate(config.personas):
            perf = performances[persona]
            current_weight = config.persona_weights[i]
            
            # Higher performance → slightly reduce weight (easier)
            # Lower performance → slightly increase weight (harder)
            adjustment = self.adaptation_rate * (0.5 - perf)
            new_weight = current_weight + adjustment
            new_weight = max(0.1, min(0.9, new_weight))  # Clamp
            new_weights.append(new_weight)
        
        # Normalize
        total = sum(new_weights)
        config.persona_weights = [w / total for w in new_weights]
    
    def _check_regression(self):
        """Check if should regress to easier stage"""
        if self.current_stage == CurriculumStage.STAGE_1:
            return
        
        # Check if struggling too much
        success_rate = self.get_success_rate()
        if success_rate < self.regression_threshold and self.episodes_in_stage > 30:
            logger.warning(f"Performance too low ({success_rate:.1%}), regressing curriculum")
            
            stage_order = list(CurriculumStage)
            current_idx = stage_order.index(self.current_stage)
            
            if current_idx > 0:
                new_stage = stage_order[current_idx - 1]
                self.force_stage(new_stage)
    
    def get_persona_performance(self) -> Dict[str, float]:
        """Get success rate per persona"""
        result = {}
        for persona, history in self.persona_success.items():
            if history:
                result[persona] = sum(history) / len(history)
        return result


# ============================================================================
# CURRICULUM TRAINER
# ============================================================================

class CurriculumTrainer:
    """
    Training wrapper that integrates curriculum learning
    
    Wraps the standard training loop with:
    - Persona selection based on curriculum
    - Stage-aware logging
    - Curriculum state saving/loading
    """
    
    def __init__(
        self,
        agent,
        env,
        curriculum: CurriculumScheduler = None,
        log_dir: str = "curriculum_logs",
        save_curriculum: bool = True
    ):
        self.agent = agent
        self.env = env
        self.curriculum = curriculum or AdaptiveCurriculum()
        self.log_dir = log_dir
        self.save_curriculum = save_curriculum
        
        # Tracking
        self.episode_log: List[Dict] = []
        
        # Create log directory
        if save_curriculum:
            os.makedirs(log_dir, exist_ok=True)
    
    def train_episode(self) -> Tuple[float, bool, Dict]:
        """
        Run a single training episode with curriculum
        
        Returns:
            (total_reward, success, episode_info)
        """
        # Sample persona from curriculum
        persona_type = self.curriculum.sample_persona()
        
        # Reset environment with selected persona
        state, info = self.env.reset(persona_type=persona_type)
        
        total_reward = 0.0
        done = False
        step = 0
        
        while not done:
            # Select action
            action = self.agent.select_action(state)
            
            # Step environment
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store experience
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train
            self.agent.train_step()
            
            state = next_state
            total_reward += reward
            step += 1
        
        # Determine success
        success = info.get('success', total_reward > 0)
        
        # Record in curriculum
        if isinstance(self.curriculum, AdaptiveCurriculum):
            new_stage = self.curriculum.record_episode_with_persona(success, persona_type)
        else:
            new_stage = self.curriculum.record_episode(success)
        
        # Log episode
        episode_info = {
            'episode': self.curriculum.total_episodes,
            'stage': self.curriculum.current_stage.name,
            'persona': persona_type,
            'reward': total_reward,
            'success': success,
            'steps': step,
            'stage_advanced': new_stage is not None
        }
        self.episode_log.append(episode_info)
        
        # Notify agent of episode end
        if hasattr(self.agent, 'episode_done'):
            self.agent.episode_done(self.curriculum.total_episodes)
        
        return total_reward, success, episode_info
    
    def train(
        self,
        num_episodes: int,
        print_every: int = 10,
        save_every: int = 50,
        callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        """
        Run curriculum training loop
        
        Args:
            num_episodes: Total episodes to train
            print_every: Print status every N episodes
            save_every: Save checkpoint every N episodes
            callback: Optional callback after each episode
            
        Returns:
            Training summary
        """
        logger.info(f"Starting curriculum training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            reward, success, info = self.train_episode()
            
            # Callback
            if callback:
                callback(info)
            
            # Print status
            if (episode + 1) % print_every == 0:
                self._print_progress(episode + 1)
            
            # Save checkpoint
            if self.save_curriculum and (episode + 1) % save_every == 0:
                self._save_checkpoint(episode + 1)
        
        # Final save
        if self.save_curriculum:
            self._save_checkpoint(num_episodes, final=True)
        
        return self._get_training_summary()
    
    def _print_progress(self, episode: int):
        """Print training progress"""
        config = self.curriculum.current_config
        success_rate = self.curriculum.get_success_rate()
        
        recent_rewards = [e['reward'] for e in self.episode_log[-20:]]
        avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0
        
        print(f"\nEpisode {episode} | Stage: {self.curriculum.current_stage.name}")
        print(f"  Success Rate: {success_rate:.1%} | Avg Reward: {avg_reward:.2f}")
        print(f"  Episodes in stage: {self.curriculum.episodes_in_stage}/{config.min_episodes}")
    
    def _save_checkpoint(self, episode: int, final: bool = False):
        """Save curriculum checkpoint"""
        suffix = "final" if final else f"ep{episode}"
        
        # Save curriculum state
        curriculum_path = os.path.join(self.log_dir, f"curriculum_{suffix}.json")
        with open(curriculum_path, 'w') as f:
            json.dump(self.curriculum.get_statistics(), f, indent=2)
        
        # Save episode log
        log_path = os.path.join(self.log_dir, f"episode_log_{suffix}.json")
        with open(log_path, 'w') as f:
            json.dump(self.episode_log, f, indent=2)
        
        logger.info(f"Saved curriculum checkpoint at episode {episode}")
    
    def _get_training_summary(self) -> Dict:
        """Get training summary statistics"""
        summary = self.curriculum.get_statistics()
        
        # Add reward statistics
        rewards = [e['reward'] for e in self.episode_log]
        summary['total_reward'] = sum(rewards)
        summary['avg_reward'] = sum(rewards) / len(rewards) if rewards else 0
        
        # Per-stage summaries
        for stage in CurriculumStage:
            stage_episodes = [e for e in self.episode_log if e['stage'] == stage.name]
            if stage_episodes:
                stage_rewards = [e['reward'] for e in stage_episodes]
                summary[f'{stage.name}_avg_reward'] = sum(stage_rewards) / len(stage_rewards)
        
        return summary
    
    def load_curriculum(self, path: str):
        """Load curriculum state from file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Restore current stage
        stage_name = data.get('current_stage', 'STAGE_1')
        self.curriculum.current_stage = CurriculumStage[stage_name]
        self.curriculum.total_episodes = data.get('total_episodes', 0)
        self.curriculum.episodes_in_stage = data.get('current_stage_episodes', 0)
        
        logger.info(f"Loaded curriculum from {path}")


# ============================================================================
# CURRICULUM VISUALIZATION
# ============================================================================

def plot_curriculum_progress(
    episode_log: List[Dict],
    save_path: Optional[str] = None
):
    """
    Plot curriculum training progress
    
    Shows:
    - Reward over time with stage boundaries
    - Success rate per stage
    - Persona distribution over time
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Extract data
    episodes = [e['episode'] for e in episode_log]
    rewards = [e['reward'] for e in episode_log]
    stages = [e['stage'] for e in episode_log]
    successes = [e['success'] for e in episode_log]
    
    # 1. Rewards with stage coloring
    ax1 = axes[0]
    stage_colors = {
        'STAGE_1': '#4CAF50',  # Green
        'STAGE_2': '#2196F3',  # Blue
        'STAGE_3': '#FF9800',  # Orange
        'STAGE_4': '#F44336',  # Red
    }
    
    for i, (ep, rew, stage) in enumerate(zip(episodes, rewards, stages)):
        ax1.scatter(ep, rew, c=stage_colors.get(stage, 'gray'), s=20, alpha=0.6)
    
    # Add moving average
    window = 20
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], ma, 'k-', linewidth=2, label='Moving Avg')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards by Curriculum Stage')
    ax1.legend()
    
    # Add stage legend
    for stage, color in stage_colors.items():
        ax1.scatter([], [], c=color, label=stage, s=50)
    ax1.legend(loc='upper left')
    
    # 2. Success rate by stage
    ax2 = axes[1]
    stage_success = {}
    for e in episode_log:
        stage = e['stage']
        if stage not in stage_success:
            stage_success[stage] = []
        stage_success[stage].append(e['success'])
    
    stage_names = list(stage_success.keys())
    success_rates = [sum(s)/len(s) for s in stage_success.values()]
    colors = [stage_colors.get(s, 'gray') for s in stage_names]
    
    ax2.bar(stage_names, success_rates, color=colors)
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate per Stage')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='50% baseline')
    ax2.legend()
    
    # 3. Stage progression over time
    ax3 = axes[2]
    stage_nums = [int(s.split('_')[1]) for s in stages]
    ax3.plot(episodes, stage_nums, 'b-', linewidth=2)
    ax3.fill_between(episodes, stage_nums, alpha=0.3)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Stage')
    ax3.set_title('Curriculum Stage Progression')
    ax3.set_yticks([1, 2, 3, 4])
    ax3.set_yticklabels(['Stage 1', 'Stage 2', 'Stage 3', 'Stage 4'])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved curriculum plot to {save_path}")
    
    return fig


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_curriculum(
    curriculum_type: str = 'adaptive',
    start_stage: int = 1,
    **kwargs
) -> CurriculumScheduler:
    """
    Factory function to create curriculum schedulers
    
    Args:
        curriculum_type: 'standard' or 'adaptive'
        start_stage: Starting stage (1-4)
        **kwargs: Additional arguments for scheduler
        
    Returns:
        CurriculumScheduler instance
    """
    stage = CurriculumStage(start_stage)
    
    if curriculum_type == 'adaptive':
        return AdaptiveCurriculum(start_stage=stage, **kwargs)
    else:
        return CurriculumScheduler(start_stage=stage, **kwargs)
