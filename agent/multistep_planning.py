"""
Multi-step Planning Module
Advanced planning strategies for DDQ agent

Planning Strategies:
1. RolloutPlanner - Monte Carlo rollouts with world model
2. TreeSearchPlanner - Tree search with action branching
3. MPCPlanner - Model Predictive Control with action optimization
4. UncertaintyAwarePlanner - Planning that considers world model uncertainty

All planners share a common interface for integration with DDQ agent.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import heapq

from config import EnvironmentConfig, RLConfig, DDQConfig, DeviceConfig


# ============================================================================
# BASE PLANNER CLASS
# ============================================================================

class BasePlanner(ABC):
    """Abstract base class for all planners"""
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        policy_net: torch.nn.Module,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        discount: float = RLConfig.GAMMA,
        device: str = None
    ):
        self.world_model = world_model
        self.policy_net = policy_net
        self.action_dim = action_dim
        self.discount = discount
        self.device = device or DeviceConfig.DEVICE
        
    @abstractmethod
    def plan(self, state: torch.Tensor, **kwargs) -> int:
        """
        Plan and return the best action
        
        Args:
            state: Current state tensor
            **kwargs: Planning parameters
            
        Returns:
            Best action index
        """
        pass
    
    def _get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Get Q-values from policy network"""
        with torch.no_grad():
            return self.policy_net(state)
    
    def _one_hot_action(self, action: int) -> torch.Tensor:
        """Convert action index to one-hot tensor"""
        action_one_hot = torch.zeros(self.action_dim, device=self.device)
        action_one_hot[action] = 1.0
        return action_one_hot


# ============================================================================
# 1. ROLLOUT PLANNER
# ============================================================================

class RolloutPlanner(BasePlanner):
    """
    Monte Carlo rollout planning
    
    For each action, simulate multiple trajectories using the world model,
    accumulate discounted rewards, and select the action with highest 
    expected return.
    """
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        policy_net: torch.nn.Module,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        discount: float = RLConfig.GAMMA,
        horizon: int = 5,
        num_rollouts: int = 10,
        rollout_policy: str = 'epsilon_greedy',
        rollout_epsilon: float = 0.3,
        device: str = None
    ):
        super().__init__(world_model, policy_net, action_dim, discount, device)
        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.rollout_policy = rollout_policy
        self.rollout_epsilon = rollout_epsilon
    
    def plan(self, state: torch.Tensor, **kwargs) -> int:
        """
        Plan using Monte Carlo rollouts
        
        Args:
            state: Current state [state_dim]
            
        Returns:
            Best action index
        """
        horizon = kwargs.get('horizon', self.horizon)
        num_rollouts = kwargs.get('num_rollouts', self.num_rollouts)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        action_returns = torch.zeros(self.action_dim, device=self.device)
        
        # Evaluate each first action
        for first_action in range(self.action_dim):
            total_return = 0.0
            
            for _ in range(num_rollouts):
                rollout_return = self._simulate_rollout(
                    state.squeeze(0), 
                    first_action, 
                    horizon
                )
                total_return += rollout_return
            
            action_returns[first_action] = total_return / num_rollouts
        
        return action_returns.argmax().item()
    
    def _simulate_rollout(
        self, 
        start_state: torch.Tensor, 
        first_action: int, 
        horizon: int
    ) -> float:
        """Simulate a single rollout starting with first_action"""
        current_state = start_state.clone()
        total_return = 0.0
        
        # First step with specified action
        action_one_hot = self._one_hot_action(first_action)
        next_state, reward = self.world_model.predict(
            current_state.unsqueeze(0),
            action_one_hot.unsqueeze(0)
        )
        total_return = reward.item()
        current_state = next_state.squeeze(0)
        
        # Continue rollout with policy
        for step in range(1, horizon):
            action = self._select_rollout_action(current_state)
            action_one_hot = self._one_hot_action(action)
            
            next_state, reward = self.world_model.predict(
                current_state.unsqueeze(0),
                action_one_hot.unsqueeze(0)
            )
            
            total_return += (self.discount ** step) * reward.item()
            current_state = next_state.squeeze(0)
        
        # Add terminal value estimate from Q-network
        terminal_value = self._get_q_values(current_state.unsqueeze(0)).max().item()
        total_return += (self.discount ** horizon) * terminal_value
        
        return total_return
    
    def _select_rollout_action(self, state: torch.Tensor) -> int:
        """Select action during rollout"""
        if self.rollout_policy == 'random':
            return np.random.randint(self.action_dim)
        elif self.rollout_policy == 'epsilon_greedy':
            if np.random.random() < self.rollout_epsilon:
                return np.random.randint(self.action_dim)
            else:
                q_values = self._get_q_values(state.unsqueeze(0))
                return q_values.argmax().item()
        else:  # greedy
            q_values = self._get_q_values(state.unsqueeze(0))
            return q_values.argmax().item()


# ============================================================================
# 2. TREE SEARCH PLANNER
# ============================================================================

class TreeNode:
    """Node in the search tree"""
    
    def __init__(
        self, 
        state: torch.Tensor, 
        action: Optional[int] = None,
        reward: float = 0.0,
        depth: int = 0,
        parent: 'TreeNode' = None
    ):
        self.state = state
        self.action = action  # Action that led to this state
        self.reward = reward  # Immediate reward
        self.depth = depth
        self.parent = parent
        self.children: Dict[int, 'TreeNode'] = {}
        self.value = 0.0  # Estimated value
        self.visits = 0
        
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def __lt__(self, other):
        # For heap operations
        return self.value > other.value


class TreeSearchPlanner(BasePlanner):
    """
    Tree search planning with action branching
    
    Builds a search tree by expanding promising nodes,
    using the world model to predict transitions.
    Supports:
    - Best-first search
    - Limited depth search
    - Pruning based on value estimates
    """
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        policy_net: torch.nn.Module,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        discount: float = RLConfig.GAMMA,
        max_depth: int = 5,
        max_expansions: int = 100,
        expansion_policy: str = 'best_first',
        device: str = None
    ):
        super().__init__(world_model, policy_net, action_dim, discount, device)
        self.max_depth = max_depth
        self.max_expansions = max_expansions
        self.expansion_policy = expansion_policy
        
    def plan(self, state: torch.Tensor, **kwargs) -> int:
        """
        Plan using tree search
        
        Args:
            state: Current state [state_dim]
            
        Returns:
            Best action index
        """
        max_depth = kwargs.get('max_depth', self.max_depth)
        max_expansions = kwargs.get('max_expansions', self.max_expansions)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        # Create root node
        root = TreeNode(state.squeeze(0), depth=0)
        root.value = self._estimate_value(root.state)
        
        # Priority queue for best-first search
        frontier = [root]
        expansions = 0
        
        while frontier and expansions < max_expansions:
            if self.expansion_policy == 'best_first':
                # Pop highest value node
                node = heapq.heappop(frontier)
                # Re-negate since we stored negative for min-heap behavior
                heapq.heappush(frontier, node)
                node = frontier[0]
                frontier = frontier[1:]
            else:  # depth_first
                node = frontier.pop()
            
            if node.depth >= max_depth:
                continue
            
            # Expand node
            self._expand_node(node)
            expansions += 1
            
            # Add children to frontier
            for child in node.children.values():
                frontier.append(child)
                if self.expansion_policy == 'best_first':
                    heapq.heapify(frontier)
        
        # Backup values through tree
        self._backup_values(root)
        
        # Select best first action
        best_action = None
        best_value = float('-inf')
        
        for action, child in root.children.items():
            if child.value > best_value:
                best_value = child.value
                best_action = action
        
        return best_action if best_action is not None else 0
    
    def _expand_node(self, node: TreeNode):
        """Expand node by trying all actions"""
        for action in range(self.action_dim):
            action_one_hot = self._one_hot_action(action)
            
            next_state, reward = self.world_model.predict(
                node.state.unsqueeze(0),
                action_one_hot.unsqueeze(0)
            )
            
            child = TreeNode(
                state=next_state.squeeze(0),
                action=action,
                reward=reward.item(),
                depth=node.depth + 1,
                parent=node
            )
            child.value = self._estimate_value(child.state)
            node.children[action] = child
    
    def _estimate_value(self, state: torch.Tensor) -> float:
        """Estimate state value using Q-network"""
        q_values = self._get_q_values(state.unsqueeze(0))
        return q_values.max().item()
    
    def _backup_values(self, root: TreeNode):
        """Backup values from leaves to root"""
        def _backup(node: TreeNode) -> float:
            if node.is_leaf():
                return node.value
            
            # Get max child value
            child_values = []
            for action, child in node.children.items():
                child_return = child.reward + self.discount * _backup(child)
                child_values.append((action, child_return))
                child.value = child_return
            
            if child_values:
                node.value = max(v for _, v in child_values)
            
            return node.value
        
        _backup(root)


# ============================================================================
# 3. MODEL PREDICTIVE CONTROL (MPC) PLANNER
# ============================================================================

class MPCPlanner(BasePlanner):
    """
    Model Predictive Control for action sequence optimization
    
    Optimizes an action sequence over a horizon using:
    - Random shooting: Sample many random action sequences
    - Cross-entropy method (CEM): Iteratively refine action distribution
    """
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        policy_net: torch.nn.Module,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        discount: float = RLConfig.GAMMA,
        horizon: int = 5,
        num_samples: int = 100,
        num_elites: int = 10,
        cem_iterations: int = 3,
        method: str = 'cem',
        device: str = None
    ):
        super().__init__(world_model, policy_net, action_dim, discount, device)
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.cem_iterations = cem_iterations
        self.method = method
        
    def plan(self, state: torch.Tensor, **kwargs) -> int:
        """
        Plan using MPC
        
        Args:
            state: Current state [state_dim]
            
        Returns:
            Best first action
        """
        horizon = kwargs.get('horizon', self.horizon)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        if self.method == 'random_shooting':
            return self._random_shooting(state, horizon)
        else:  # cem
            return self._cross_entropy_method(state, horizon)
    
    def _random_shooting(self, state: torch.Tensor, horizon: int) -> int:
        """Random shooting: sample random action sequences, pick best"""
        best_return = float('-inf')
        best_first_action = 0
        
        for _ in range(self.num_samples):
            # Sample random action sequence
            actions = torch.randint(0, self.action_dim, (horizon,))
            
            # Evaluate sequence
            total_return = self._evaluate_sequence(state, actions)
            
            if total_return > best_return:
                best_return = total_return
                best_first_action = actions[0].item()
        
        return best_first_action
    
    def _cross_entropy_method(self, state: torch.Tensor, horizon: int) -> int:
        """Cross-entropy method: iteratively refine action distribution"""
        # Initialize uniform distribution over actions
        action_probs = torch.ones(horizon, self.action_dim, device=self.device)
        action_probs = action_probs / self.action_dim
        
        for _ in range(self.cem_iterations):
            # Sample action sequences from current distribution
            sequences = []
            returns = []
            
            for _ in range(self.num_samples):
                actions = torch.zeros(horizon, dtype=torch.long, device=self.device)
                for t in range(horizon):
                    actions[t] = torch.multinomial(action_probs[t], 1).item()
                
                total_return = self._evaluate_sequence(state, actions)
                sequences.append(actions)
                returns.append(total_return)
            
            # Select elite sequences
            returns = torch.tensor(returns, device=self.device)
            _, elite_indices = torch.topk(returns, self.num_elites)
            elite_sequences = [sequences[i] for i in elite_indices]
            
            # Update action distribution based on elites
            action_counts = torch.zeros(horizon, self.action_dim, device=self.device)
            for seq in elite_sequences:
                for t in range(horizon):
                    action_counts[t, seq[t]] += 1
            
            # Smoothed distribution update
            action_probs = (action_counts + 0.1) / (self.num_elites + 0.1 * self.action_dim)
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
        
        # Return most likely first action
        return action_probs[0].argmax().item()
    
    def _evaluate_sequence(
        self, 
        start_state: torch.Tensor, 
        actions: torch.Tensor
    ) -> float:
        """Evaluate an action sequence using world model"""
        current_state = start_state.squeeze(0)
        total_return = 0.0
        
        for t, action in enumerate(actions):
            action_one_hot = self._one_hot_action(action.item())
            
            next_state, reward = self.world_model.predict(
                current_state.unsqueeze(0),
                action_one_hot.unsqueeze(0)
            )
            
            total_return += (self.discount ** t) * reward.item()
            current_state = next_state.squeeze(0)
        
        # Add terminal value
        terminal_value = self._get_q_values(current_state.unsqueeze(0)).max().item()
        total_return += (self.discount ** len(actions)) * terminal_value
        
        return total_return


# ============================================================================
# 4. UNCERTAINTY-AWARE PLANNER
# ============================================================================

class UncertaintyAwarePlanner(BasePlanner):
    """
    Planning that considers world model uncertainty
    
    Uses ensemble disagreement or probabilistic outputs to:
    - Avoid unreliable predictions
    - Balance exploration and exploitation
    - Weight actions by prediction confidence
    """
    
    def __init__(
        self,
        world_model: torch.nn.Module,  # Should be ensemble or probabilistic
        policy_net: torch.nn.Module,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        discount: float = RLConfig.GAMMA,
        horizon: int = 3,
        num_rollouts: int = 10,
        uncertainty_penalty: float = 0.5,
        max_uncertainty: float = 1.0,
        device: str = None
    ):
        super().__init__(world_model, policy_net, action_dim, discount, device)
        self.horizon = horizon
        self.num_rollouts = num_rollouts
        self.uncertainty_penalty = uncertainty_penalty
        self.max_uncertainty = max_uncertainty
    
    def plan(self, state: torch.Tensor, **kwargs) -> int:
        """
        Plan with uncertainty awareness
        
        Args:
            state: Current state [state_dim]
            
        Returns:
            Best action index
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state = state.to(self.device)
        
        action_values = torch.zeros(self.action_dim, device=self.device)
        action_uncertainties = torch.zeros(self.action_dim, device=self.device)
        
        for action in range(self.action_dim):
            value, uncertainty = self._evaluate_action_with_uncertainty(
                state.squeeze(0), 
                action
            )
            action_values[action] = value
            action_uncertainties[action] = uncertainty
        
        # Compute uncertainty-penalized values
        penalized_values = action_values - self.uncertainty_penalty * action_uncertainties
        
        return penalized_values.argmax().item()
    
    def _evaluate_action_with_uncertainty(
        self, 
        state: torch.Tensor, 
        first_action: int
    ) -> Tuple[float, float]:
        """Evaluate action with uncertainty estimation"""
        total_return = 0.0
        total_uncertainty = 0.0
        
        for _ in range(self.num_rollouts):
            rollout_return, rollout_uncertainty = self._uncertain_rollout(
                state, 
                first_action
            )
            total_return += rollout_return
            total_uncertainty += rollout_uncertainty
        
        avg_return = total_return / self.num_rollouts
        avg_uncertainty = total_uncertainty / self.num_rollouts
        
        return avg_return, avg_uncertainty
    
    def _uncertain_rollout(
        self, 
        start_state: torch.Tensor, 
        first_action: int
    ) -> Tuple[float, float]:
        """Single rollout tracking uncertainty"""
        current_state = start_state.clone()
        total_return = 0.0
        total_uncertainty = 0.0
        
        # First step
        action_one_hot = self._one_hot_action(first_action)
        next_state, reward, uncertainty = self._predict_with_uncertainty(
            current_state, action_one_hot
        )
        total_return = reward
        total_uncertainty = uncertainty
        current_state = next_state
        
        # Continue rollout
        for step in range(1, self.horizon):
            # Stop if uncertainty is too high
            if total_uncertainty > self.max_uncertainty * step:
                break
            
            # Greedy action selection
            q_values = self._get_q_values(current_state.unsqueeze(0))
            action = q_values.argmax().item()
            action_one_hot = self._one_hot_action(action)
            
            next_state, reward, uncertainty = self._predict_with_uncertainty(
                current_state, action_one_hot
            )
            
            total_return += (self.discount ** step) * reward
            total_uncertainty += uncertainty
            current_state = next_state
        
        # Terminal value
        terminal_value = self._get_q_values(current_state.unsqueeze(0)).max().item()
        total_return += (self.discount ** self.horizon) * terminal_value
        
        return total_return, total_uncertainty
    
    def _predict_with_uncertainty(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, float, float]:
        """Get prediction with uncertainty estimate"""
        state_batch = state.unsqueeze(0)
        action_batch = action.unsqueeze(0)
        
        # Check if world model supports uncertainty
        if hasattr(self.world_model, 'predict_with_uncertainty'):
            next_state, reward, uncertainty = self.world_model.predict_with_uncertainty(
                state_batch, action_batch
            )
            return next_state.squeeze(0), reward.item(), uncertainty.item()
        elif hasattr(self.world_model, 'forward') and hasattr(self.world_model, 'get_uncertainty'):
            next_state, reward = self.world_model.predict(state_batch, action_batch)
            uncertainty = self.world_model.get_uncertainty(state_batch, action_batch)
            return next_state.squeeze(0), reward.item(), uncertainty.mean().item()
        else:
            # No uncertainty available
            next_state, reward = self.world_model.predict(state_batch, action_batch)
            return next_state.squeeze(0), reward.item(), 0.0


# ============================================================================
# PLANNER FACTORY
# ============================================================================

def create_planner(
    planner_type: str,
    world_model: torch.nn.Module,
    policy_net: torch.nn.Module,
    action_dim: int = EnvironmentConfig.NUM_ACTIONS,
    **kwargs
) -> BasePlanner:
    """
    Factory function to create planners
    
    Args:
        planner_type: One of ['rollout', 'tree', 'mpc', 'uncertainty']
        world_model: World model for predictions
        policy_net: Q-network for value estimates
        action_dim: Number of actions
        **kwargs: Additional planner-specific arguments
        
    Returns:
        Planner instance
    """
    planner_type = planner_type.lower()
    
    if planner_type == 'rollout':
        return RolloutPlanner(world_model, policy_net, action_dim, **kwargs)
    elif planner_type == 'tree':
        return TreeSearchPlanner(world_model, policy_net, action_dim, **kwargs)
    elif planner_type == 'mpc' or planner_type == 'cem':
        return MPCPlanner(world_model, policy_net, action_dim, **kwargs)
    elif planner_type == 'uncertainty':
        return UncertaintyAwarePlanner(world_model, policy_net, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown planner type: {planner_type}. "
                        f"Choose from: rollout, tree, mpc, uncertainty")


# ============================================================================
# PLANNER COMPARISON UTILITY
# ============================================================================

class PlannerComparison:
    """Utility for comparing different planning strategies"""
    
    def __init__(
        self,
        world_model: torch.nn.Module,
        policy_net: torch.nn.Module,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        device: str = 'cpu'
    ):
        self.world_model = world_model
        self.policy_net = policy_net
        self.action_dim = action_dim
        self.device = device
        self.planners = {}
        self.results = defaultdict(list)
    
    def add_planner(self, name: str, planner: BasePlanner):
        """Add a planner to comparison"""
        self.planners[name] = planner
    
    def compare_on_states(
        self, 
        states: List[torch.Tensor],
        q_network_actions: Optional[List[int]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare planners on a set of states
        
        Args:
            states: List of states to plan from
            q_network_actions: Optional baseline actions from Q-network
            
        Returns:
            Comparison results per planner
        """
        results = defaultdict(lambda: {'actions': [], 'times': [], 'agreement': 0})
        
        import time
        
        for i, state in enumerate(states):
            state = state.to(self.device)
            
            for name, planner in self.planners.items():
                start = time.time()
                action = planner.plan(state)
                elapsed = time.time() - start
                
                results[name]['actions'].append(action)
                results[name]['times'].append(elapsed)
                
                if q_network_actions is not None:
                    if action == q_network_actions[i]:
                        results[name]['agreement'] += 1
        
        # Compute summary statistics
        for name in results:
            n = len(states)
            results[name]['avg_time'] = np.mean(results[name]['times'])
            results[name]['agreement_rate'] = results[name]['agreement'] / n if n > 0 else 0
        
        return dict(results)
    
    def print_comparison(self, results: Dict[str, Dict[str, Any]]):
        """Pretty print comparison results"""
        print("\n" + "="*70)
        print("PLANNER COMPARISON")
        print("="*70)
        print(f"\n{'Planner':<20} {'Avg Time (ms)':>15} {'Q-Agreement':>15}")
        print("-"*70)
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['avg_time']):
            time_ms = metrics['avg_time'] * 1000
            agreement = metrics.get('agreement_rate', 0) * 100
            print(f"{name:<20} {time_ms:>15.2f} {agreement:>14.1f}%")
        
        print("="*70)
