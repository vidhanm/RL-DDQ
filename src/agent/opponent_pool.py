"""
Opponent Pool for Self-Play Training

Maintains a pool of historical agent versions for diverse training.
Prevents strategy oscillation by training against variety of opponents.
"""

import os
import random
from typing import List, Optional, Dict, Any
from pathlib import Path


class OpponentPool:
    """
    Maintains pool of historical agent versions for diverse training.
    
    Key Features:
    - Stores checkpoints of agent versions from different generations
    - Supports multiple sampling strategies (uniform, prioritized, latest)
    - Prevents mode collapse by ensuring diversity in training opponents
    """
    
    def __init__(
        self,
        pool_dir: str = "checkpoints/opponent_pool",
        max_size: int = 10,
        agent_type: str = "collector"  # or "adversary"
    ):
        """
        Initialize opponent pool.
        
        Args:
            pool_dir: Directory to store checkpoints
            max_size: Maximum number of opponents to keep
            agent_type: Type of agent this pool contains
        """
        self.pool_dir = Path(pool_dir) / agent_type
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size = max_size
        self.agent_type = agent_type
        
        # Track pool contents
        self.pool: List[str] = []
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Load existing checkpoints if any
        self._load_existing_pool()
    
    def _load_existing_pool(self):
        """Load existing checkpoints from pool directory."""
        if not self.pool_dir.exists():
            return
        
        # Find all .pt files
        checkpoints = sorted(self.pool_dir.glob("*.pt"))
        
        for ckpt in checkpoints:
            if ckpt.name not in self.pool:
                self.pool.append(str(ckpt))
        
        # Keep only most recent if over max size
        if len(self.pool) > self.max_size:
            to_remove = self.pool[:-self.max_size]
            for path in to_remove:
                try:
                    os.remove(path)
                except OSError:
                    pass
            self.pool = self.pool[-self.max_size:]
    
    def add(
        self,
        agent,  # DDQAgent or AdversarialDebtorAgent
        generation: int,
        win_rate: float = 0.0,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add new agent version to pool.
        
        Args:
            agent: Agent to save
            generation: Generation number
            win_rate: Win rate achieved at this generation
            extra_metadata: Additional metadata to store
            
        Returns:
            Path to saved checkpoint
        """
        # Generate filename
        filename = f"{self.agent_type}_gen{generation:04d}.pt"
        filepath = str(self.pool_dir / filename)
        
        # Save agent
        agent.save(filepath)
        
        # Store metadata
        self.metadata[filepath] = {
            "generation": generation,
            "win_rate": win_rate,
            **(extra_metadata or {})
        }
        
        # Add to pool
        self.pool.append(filepath)
        
        # Remove oldest if over max size
        if len(self.pool) > self.max_size:
            oldest = self.pool.pop(0)
            try:
                os.remove(oldest)
                # Also remove stats file if exists
                stats_path = oldest.replace('.pt', '_adversary_stats.pt')
                if os.path.exists(stats_path):
                    os.remove(stats_path)
            except OSError:
                pass
            if oldest in self.metadata:
                del self.metadata[oldest]
        
        return filepath
    
    def sample(self, strategy: str = "uniform") -> Optional[str]:
        """
        Sample opponent from pool.
        
        Args:
            strategy: Sampling strategy
                - "uniform": Equal probability for all
                - "latest": Always return latest version
                - "prioritized": Weight towards more recent
                - "performance": Weight by win rate
                
        Returns:
            Path to opponent checkpoint, or None if pool is empty
        """
        if not self.pool:
            return None
        
        if strategy == "uniform":
            return random.choice(self.pool)
        
        elif strategy == "latest":
            return self.pool[-1]
        
        elif strategy == "prioritized":
            # Weight towards more recent opponents
            n = len(self.pool)
            weights = [i + 1 for i in range(n)]
            return random.choices(self.pool, weights=weights)[0]
        
        elif strategy == "performance":
            # Weight by win rate (higher win rate = more likely to be sampled)
            weights = []
            for path in self.pool:
                meta = self.metadata.get(path, {})
                win_rate = meta.get("win_rate", 0.5)
                weights.append(max(0.1, win_rate))  # Min weight of 0.1
            return random.choices(self.pool, weights=weights)[0]
        
        else:
            # Default to uniform
            return random.choice(self.pool)
    
    def sample_multiple(self, n: int, strategy: str = "uniform") -> List[str]:
        """
        Sample multiple opponents from pool.
        
        Args:
            n: Number of opponents to sample
            strategy: Sampling strategy
            
        Returns:
            List of checkpoint paths
        """
        if not self.pool:
            return []
        
        # Sample with replacement if pool is smaller than n
        return [self.sample(strategy) for _ in range(n)]
    
    def get_latest(self) -> Optional[str]:
        """Get the latest (most recent) opponent."""
        return self.pool[-1] if self.pool else None
    
    def get_best(self) -> Optional[str]:
        """Get the opponent with highest win rate."""
        if not self.pool:
            return None
        
        best_path = None
        best_rate = -1.0
        
        for path in self.pool:
            meta = self.metadata.get(path, {})
            rate = meta.get("win_rate", 0.0)
            if rate > best_rate:
                best_rate = rate
                best_path = path
        
        return best_path or self.pool[-1]
    
    def size(self) -> int:
        """Get current pool size."""
        return len(self.pool)
    
    def is_empty(self) -> bool:
        """Check if pool is empty."""
        return len(self.pool) == 0
    
    def get_all(self) -> List[str]:
        """Get all opponent paths."""
        return list(self.pool)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if not self.pool:
            return {"size": 0, "generations": [], "avg_win_rate": 0.0}
        
        generations = []
        win_rates = []
        
        for path in self.pool:
            meta = self.metadata.get(path, {})
            generations.append(meta.get("generation", 0))
            win_rates.append(meta.get("win_rate", 0.5))
        
        return {
            "size": len(self.pool),
            "generations": generations,
            "min_generation": min(generations) if generations else 0,
            "max_generation": max(generations) if generations else 0,
            "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
            "best_win_rate": max(win_rates) if win_rates else 0.0,
        }
    
    def clear(self):
        """Clear the pool (delete all checkpoints)."""
        for path in self.pool:
            try:
                os.remove(path)
                stats_path = path.replace('.pt', '_adversary_stats.pt')
                if os.path.exists(stats_path):
                    os.remove(stats_path)
            except OSError:
                pass
        
        self.pool = []
        self.metadata = {}


# =============================================================================
# DUAL POOL MANAGER
# =============================================================================

class DualPoolManager:
    """
    Manages both collector and adversary opponent pools together.
    
    Provides convenience methods for self-play training loop.
    """
    
    def __init__(
        self,
        pool_dir: str = "checkpoints/opponent_pool",
        max_size: int = 10
    ):
        """
        Initialize dual pool manager.
        
        Args:
            pool_dir: Base directory for pools
            max_size: Max size per pool
        """
        self.collector_pool = OpponentPool(pool_dir, max_size, "collector")
        self.adversary_pool = OpponentPool(pool_dir, max_size, "adversary")
    
    def add_collector(self, agent, generation: int, win_rate: float = 0.0) -> str:
        """Add collector to pool."""
        return self.collector_pool.add(agent, generation, win_rate)
    
    def add_adversary(self, agent, generation: int, resistance_rate: float = 0.0) -> str:
        """Add adversary to pool."""
        return self.adversary_pool.add(agent, generation, resistance_rate)
    
    def sample_collector_opponent(self, strategy: str = "uniform") -> Optional[str]:
        """Sample a collector opponent for adversary training."""
        return self.collector_pool.sample(strategy)
    
    def sample_adversary_opponent(self, strategy: str = "uniform") -> Optional[str]:
        """Sample an adversary opponent for collector training."""
        return self.adversary_pool.sample(strategy)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for both pools."""
        return {
            "collector_pool": self.collector_pool.get_statistics(),
            "adversary_pool": self.adversary_pool.get_statistics(),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing OpponentPool")
    print("=" * 50)
    
    # Create a test pool
    pool = OpponentPool(
        pool_dir="checkpoints/test_pool",
        max_size=5,
        agent_type="test"
    )
    
    print(f"Initial pool size: {pool.size()}")
    print(f"Pool is empty: {pool.is_empty()}")
    
    # Test adding (mock - we'll just test the logic without real agents)
    class MockAgent:
        def save(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).touch()
    
    mock_agent = MockAgent()
    
    # Add several generations
    for gen in range(8):
        win_rate = 0.4 + gen * 0.05
        path = pool.add(mock_agent, generation=gen, win_rate=win_rate)
        print(f"Added gen {gen} at {path}")
    
    print(f"\nPool size after additions: {pool.size()}")
    print(f"Should be max {pool.max_size}")
    
    # Test sampling
    print("\nSampling tests:")
    print(f"  Uniform: {pool.sample('uniform')}")
    print(f"  Latest: {pool.sample('latest')}")
    print(f"  Prioritized: {pool.sample('prioritized')}")
    print(f"  Performance: {pool.sample('performance')}")
    
    # Test statistics
    print(f"\nStatistics: {pool.get_statistics()}")
    
    # Cleanup
    pool.clear()
    print(f"\nAfter clear: {pool.size()}")
    
    # Remove test directory
    import shutil
    shutil.rmtree("checkpoints/test_pool", ignore_errors=True)
    
    print("\n✓ Test complete!")
