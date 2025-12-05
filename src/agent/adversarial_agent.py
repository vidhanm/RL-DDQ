"""
Adversarial Debtor Agent for Self-Play Training

This agent plays as a challenging debtor, learning to resist collector strategies.
Used in adversarial self-play to train a robust collector agent.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any

from .ddq_agent import DDQAgent
from src.config import SelfPlayConfig, EnvironmentConfig, DeviceConfig


class AdversarialDebtorAgent(DDQAgent):
    """
    Adversarial debtor agent for self-play training.
    
    Learns to be a challenging debtor that exposes collector weaknesses.
    Uses the same DDQ architecture as the collector for consistency.
    
    Key Differences from Collector:
    - Different action space (resistance strategies instead of collection tactics)
    - Inverse reward structure (benefits from collector failure)
    - Goal: Prolong conversation without commitment or make collector give up
    """
    
    def __init__(
        self,
        state_dim: int = EnvironmentConfig.NLU_STATE_DIM,
        action_dim: int = SelfPlayConfig.NUM_ADVERSARY_ACTIONS,
        role: str = "adversary",
        **kwargs
    ):
        """
        Initialize Adversarial Debtor Agent.
        
        Args:
            state_dim: State dimension (same as collector observes)
            action_dim: Number of resistance strategies (7)
            role: Agent role identifier ("adversary")
            **kwargs: Additional arguments passed to DDQAgent
        """
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)
        
        self.role = role
        self.response_strategies = list(SelfPlayConfig.ADVERSARY_ACTIONS.values())
        
        # Track strategy usage for analysis
        self.strategy_counts = {strategy: 0 for strategy in self.response_strategies}
        self.strategy_successes = {strategy: 0 for strategy in self.response_strategies}
        
        # Episode tracking
        self.current_episode_strategies = []
        self.episodes_resisted = 0  # Count of episodes where collector failed
        self.total_episodes = 0
    
    def select_strategy(self, state: np.ndarray) -> int:
        """
        Select a resistance strategy based on current state.
        
        Args:
            state: Current conversation state features
            
        Returns:
            Strategy index (0-6)
        """
        action = self.select_action(state)
        strategy_name = self.get_strategy_name(action)
        
        # Track usage
        self.strategy_counts[strategy_name] += 1
        self.current_episode_strategies.append(action)
        
        return action
    
    def get_strategy_name(self, action: int) -> str:
        """Get strategy name from action index."""
        return SelfPlayConfig.ADVERSARY_ACTIONS.get(action, "unknown")
    
    def calculate_adversary_reward(
        self,
        collector_reward: float,
        turn_number: int,
        episode_ended: bool,
        commitment_achieved: bool
    ) -> float:
        """
        Calculate adversary reward based on collector's outcome.
        
        Args:
            collector_reward: The reward the collector received
            turn_number: Current turn number
            episode_ended: Whether the episode ended this turn
            commitment_achieved: Whether collector got payment commitment
            
        Returns:
            Adversary reward (float)
        """
        reward = 0.0
        
        # Core: Inverse of collector reward (scaled)
        reward -= collector_reward * SelfPlayConfig.ZERO_SUM_COEFFICIENT
        
        # Bonus for stalling (each turn without commitment)
        if not commitment_achieved:
            reward += SelfPlayConfig.STALL_BONUS_PER_TURN
        
        # Episode-end rewards
        if episode_ended:
            if not commitment_achieved:
                # Collector failed - adversary wins!
                reward += SelfPlayConfig.RESIST_COMMITMENT_BONUS
            else:
                # Collector succeeded - adversary loses
                reward -= SelfPlayConfig.RESIST_COMMITMENT_BONUS * 0.5
        
        return reward * SelfPlayConfig.ADVERSARY_REWARD_SCALE
    
    def episode_end(self, collector_succeeded: bool):
        """
        Called at end of episode to update statistics.
        
        Args:
            collector_succeeded: Whether collector got a commitment
        """
        self.total_episodes += 1
        
        if not collector_succeeded:
            self.episodes_resisted += 1
            # Credit all strategies used this episode
            for action in self.current_episode_strategies:
                strategy_name = self.get_strategy_name(action)
                self.strategy_successes[strategy_name] += 1
        
        self.current_episode_strategies = []
    
    def get_resistance_rate(self) -> float:
        """Get the rate of successfully resisting collector."""
        if self.total_episodes == 0:
            return 0.0
        return self.episodes_resisted / self.total_episodes
    
    def get_strategy_distribution(self) -> Dict[str, float]:
        """Get the distribution of strategies used."""
        total = sum(self.strategy_counts.values())
        if total == 0:
            return {s: 1.0/len(self.response_strategies) for s in self.response_strategies}
        return {s: count/total for s, count in self.strategy_counts.items()}
    
    def get_strategy_effectiveness(self) -> Dict[str, float]:
        """Get effectiveness (success rate) of each strategy."""
        effectiveness = {}
        for strategy in self.response_strategies:
            if self.strategy_counts[strategy] == 0:
                effectiveness[strategy] = 0.0
            else:
                effectiveness[strategy] = (
                    self.strategy_successes[strategy] / self.strategy_counts[strategy]
                )
        return effectiveness
    
    def get_action_entropy(self) -> float:
        """
        Calculate entropy of strategy distribution.
        Higher entropy = more diverse strategies (good).
        """
        dist = self.get_strategy_distribution()
        entropy = 0.0
        for prob in dist.values():
            if prob > 0:
                entropy -= prob * np.log(prob + 1e-10)
        return entropy
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics including adversary-specific metrics."""
        base_stats = super().get_statistics()
        
        adversary_stats = {
            "role": self.role,
            "resistance_rate": self.get_resistance_rate(),
            "total_episodes": self.total_episodes,
            "episodes_resisted": self.episodes_resisted,
            "strategy_distribution": self.get_strategy_distribution(),
            "strategy_effectiveness": self.get_strategy_effectiveness(),
            "action_entropy": self.get_action_entropy(),
        }
        
        return {**base_stats, **adversary_stats}
    
    def reset_statistics(self):
        """Reset episode statistics (but keep learned policy)."""
        self.strategy_counts = {strategy: 0 for strategy in self.response_strategies}
        self.strategy_successes = {strategy: 0 for strategy in self.response_strategies}
        self.episodes_resisted = 0
        self.total_episodes = 0
        self.current_episode_strategies = []
    
    def save(self, filepath: str):
        """Save agent including adversary-specific data."""
        # Save base agent
        super().save(filepath)
        
        # Save adversary statistics
        stats_path = filepath.replace('.pt', '_adversary_stats.pt')
        torch.save({
            'strategy_counts': self.strategy_counts,
            'strategy_successes': self.strategy_successes,
            'episodes_resisted': self.episodes_resisted,
            'total_episodes': self.total_episodes,
            'role': self.role,
        }, stats_path)
    
    def load(self, filepath: str):
        """Load agent including adversary-specific data."""
        # Load base agent
        super().load(filepath)
        
        # Load adversary statistics if available
        stats_path = filepath.replace('.pt', '_adversary_stats.pt')
        try:
            stats = torch.load(stats_path, map_location=DeviceConfig.DEVICE)
            self.strategy_counts = stats.get('strategy_counts', self.strategy_counts)
            self.strategy_successes = stats.get('strategy_successes', self.strategy_successes)
            self.episodes_resisted = stats.get('episodes_resisted', 0)
            self.total_episodes = stats.get('total_episodes', 0)
            self.role = stats.get('role', 'adversary')
        except FileNotFoundError:
            pass  # Stats file doesn't exist, use defaults


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_adversarial_agent(
    state_dim: int = EnvironmentConfig.NLU_STATE_DIM,
    use_world_model: bool = True,
    **kwargs
) -> AdversarialDebtorAgent:
    """
    Factory function to create an adversarial debtor agent.
    
    Args:
        state_dim: State dimension
        use_world_model: Whether to use DDQ world model (recommended)
        **kwargs: Additional DDQAgent arguments
        
    Returns:
        Configured AdversarialDebtorAgent
    """
    return AdversarialDebtorAgent(
        state_dim=state_dim,
        action_dim=SelfPlayConfig.NUM_ADVERSARY_ACTIONS,
        role="adversary",
        **kwargs
    )


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing AdversarialDebtorAgent")
    print("=" * 50)
    
    # Create agent
    agent = create_adversarial_agent()
    print(f"Created adversary with {agent.action_dim} strategies")
    print(f"Strategies: {agent.response_strategies}")
    
    # Test strategy selection
    dummy_state = np.random.randn(EnvironmentConfig.NLU_STATE_DIM)
    
    print("\nTesting 10 strategy selections:")
    for i in range(10):
        action = agent.select_strategy(dummy_state)
        strategy = agent.get_strategy_name(action)
        print(f"  Turn {i+1}: {strategy}")
    
    # Test reward calculation
    print("\nTesting reward calculation:")
    test_cases = [
        (5.0, 3, False, False),   # Collector doing well mid-episode
        (-2.0, 5, False, False),  # Collector struggling
        (10.0, 8, True, True),    # Collector succeeded
        (-5.0, 10, True, False),  # Collector failed (adversary won)
    ]
    
    for collector_reward, turn, ended, committed in test_cases:
        adv_reward = agent.calculate_adversary_reward(
            collector_reward, turn, ended, committed
        )
        print(f"  Collector: {collector_reward:+.1f}, Turn: {turn}, "
              f"Ended: {ended}, Committed: {committed} → Adversary: {adv_reward:+.2f}")
    
    # End episode and check stats
    agent.episode_end(collector_succeeded=False)
    agent.episode_end(collector_succeeded=True)
    agent.episode_end(collector_succeeded=False)
    
    print(f"\nStatistics:")
    print(f"  Resistance rate: {agent.get_resistance_rate():.1%}")
    print(f"  Action entropy: {agent.get_action_entropy():.3f}")
    print(f"  Strategy distribution: {agent.get_strategy_distribution()}")
    
    print("\n✓ Test complete!")
