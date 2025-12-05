"""
Replay Buffer for Experience Replay
Stores and samples experiences for training DQN and DDQ
"""

import random
import numpy as np
import torch
from collections import deque
from typing import List, Tuple, Optional


class ReplayBuffer:
    """Experience replay buffer for DQN"""

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of experiences to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.position = 0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool):
        """
        Add experience to buffer

        Args:
            state: Current state vector
            action: Action taken
            reward: Reward received
            next_state: Next state vector
            done: Whether episode terminated
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample random batch of experiences

        Args:
            batch_size: Number of experiences to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        return states, actions, rewards, next_states, dones

    def sample_states(self, batch_size: int) -> torch.Tensor:
        """
        Sample random states (for imagination in DDQ)

        Args:
            batch_size: Number of states to sample

        Returns:
            Tensor of sampled states
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # Sample random experiences
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Extract states (could be current state or next state)
        states = []
        for exp in batch:
            # Randomly choose current state or next state
            if random.random() < 0.5:
                states.append(exp[0])  # Current state
            else:
                states.append(exp[3])  # Next state

        return torch.FloatTensor(np.array(states))

    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences"""
        return len(self.buffer) >= min_size

    def clear(self):
        """Clear all experiences"""
        self.buffer.clear()
        self.position = 0

    def get_all_experiences(self) -> List:
        """Get all experiences (for world model training)"""
        return list(self.buffer)


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer
    
    Samples experiences based on:
    1. TD error (standard PER)
    2. Event-based bonuses (commitment, hostile, de-escalation)
    """
    
    # Event-based priority bonuses
    EVENT_BONUSES = {
        'commitment': 3.0,      # Commitment attempts (rare, high value)
        'hostile': 2.0,         # Hostile situations (important to learn from)
        'de_escalation': 2.5,   # Successful de-escalations
        'high_reward': 1.5,     # High reward experiences
        'negative': 1.5,        # Negative experiences (learn from mistakes)
    }

    def __init__(self, capacity: int = 10000, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer

        Args:
            capacity: Maximum number of experiences
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.priorities = deque(maxlen=capacity)
        self.event_tags = deque(maxlen=capacity)  # Track event types
        self.max_priority = 1.0

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, event_type: str = None):
        """
        Add experience with priority based on event type
        
        Args:
            state, action, reward, next_state, done: Standard experience
            event_type: Optional event tag ('commitment', 'hostile', 'de_escalation')
        """
        super().add(state, action, reward, next_state, done)
        
        # Calculate priority with event bonus
        priority = self.max_priority
        if event_type and event_type in self.EVENT_BONUSES:
            priority *= self.EVENT_BONUSES[event_type]
        
        # Also boost based on reward magnitude
        if abs(reward) > 5.0:
            priority *= self.EVENT_BONUSES.get('high_reward', 1.5)
        elif reward < -2.0:
            priority *= self.EVENT_BONUSES.get('negative', 1.5)
        
        self.priorities.append(priority)
        self.event_tags.append(event_type)
        self.max_priority = max(self.max_priority, priority)

    def add_with_info(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool, info: dict = None):
        """
        Add experience with info dict for automatic event detection
        
        Args:
            info: Environment info dict containing signals like 'has_committed', 'sentiment', etc.
        """
        event_type = None
        
        if info:
            # Detect commitment
            if info.get('has_committed', False):
                event_type = 'commitment'
            # Detect hostile situation
            elif info.get('sentiment', 0) < -0.5 or info.get('intent') == 'hostile':
                event_type = 'hostile'
            # Detect de-escalation (sentiment improved significantly)
            elif info.get('sentiment_improved', 0) > 0.3:
                event_type = 'de_escalation'
        
        self.add(state, action, reward, next_state, done, event_type)

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """
        Sample batch with prioritization

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer),
                                   size=min(batch_size, len(self.buffer)),
                                   p=probs,
                                   replace=False)

        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()  # Normalize

        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        weights = torch.FloatTensor(weights)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities based on TD errors

        Args:
            indices: Indices of experiences
            td_errors: TD errors for those experiences
        """
        for idx, error in zip(indices, td_errors):
            # Keep event bonus when updating priority
            event_bonus = 1.0
            if idx < len(self.event_tags):
                event_type = self.event_tags[idx]
                if event_type and event_type in self.EVENT_BONUSES:
                    event_bonus = self.EVENT_BONUSES[event_type]
            
            priority = (abs(error) + 1e-6) ** self.alpha * event_bonus
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def get_event_statistics(self) -> dict:
        """Get statistics about event types in buffer"""
        stats = {event: 0 for event in self.EVENT_BONUSES.keys()}
        stats['none'] = 0
        
        for tag in self.event_tags:
            if tag:
                stats[tag] = stats.get(tag, 0) + 1
            else:
                stats['none'] += 1
        
        return stats


# ============================================================================
# Helper Functions
# ============================================================================

def create_buffer(prioritized: bool = False, capacity: int = 10000) -> ReplayBuffer:
    """
    Create replay buffer

    Args:
        prioritized: Whether to use prioritized replay
        capacity: Buffer capacity

    Returns:
        ReplayBuffer instance
    """
    if prioritized:
        return PrioritizedReplayBuffer(capacity)
    else:
        return ReplayBuffer(capacity)
