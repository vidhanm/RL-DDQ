"""
DQN Agent
Combines DQN network with training logic and replay buffer
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional

from agent.dqn import create_dqn, copy_weights
from utils.replay_buffer import ReplayBuffer
from config import RLConfig, EnvironmentConfig, DeviceConfig


class DQNAgent:
    """DQN Agent for debt collection"""

    def __init__(
        self,
        state_dim: int = EnvironmentConfig.STATE_DIM,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        learning_rate: float = RLConfig.LEARNING_RATE,
        gamma: float = RLConfig.GAMMA,
        epsilon_start: float = RLConfig.EPSILON_START,
        epsilon_end: float = RLConfig.EPSILON_END,
        epsilon_decay: float = RLConfig.EPSILON_DECAY,
        buffer_size: int = RLConfig.REPLAY_BUFFER_SIZE,
        batch_size: int = RLConfig.BATCH_SIZE,
        target_update_freq: int = RLConfig.TARGET_UPDATE_FREQ,
        device: Optional[str] = None
    ):
        """
        Initialize DQN Agent

        Args:
            state_dim: State space dimension
            action_dim: Number of actions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Epsilon decay rate per episode
            buffer_size: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Update target network every N episodes
            device: Device ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Device
        if device is None:
            self.device = DeviceConfig.DEVICE
        else:
            self.device = device

        # Networks
        self.policy_net = create_dqn(state_dim, action_dim, device=self.device)
        self.target_net = create_dqn(state_dim, action_dim, device=self.device)
        copy_weights(self.policy_net, self.target_net)
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Loss function
        self.criterion = nn.MSELoss()

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # Statistics
        self.episodes_trained = 0
        self.steps_trained = 0
        self.total_loss = 0.0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """
        Select action using epsilon-greedy policy

        Args:
            state: Current state
            explore: Whether to use exploration

        Returns:
            Action index
        """
        state_tensor = torch.FloatTensor(state).to(self.device)

        if explore:
            epsilon = self.epsilon
        else:
            epsilon = 0.0

        return self.policy_net.get_action(state_tensor, epsilon)

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer"""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train_step(self) -> float:
        """
        Perform one training step

        Returns:
            Loss value
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update statistics
        self.steps_trained += 1
        self.total_loss += loss.item()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        copy_weights(self.policy_net, self.target_net)

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def episode_done(self, episode: int):
        """
        Called at end of episode

        Args:
            episode: Episode number
        """
        self.episodes_trained += 1

        # Update epsilon
        self.update_epsilon()

        # Update target network periodically
        if episode % self.target_update_freq == 0:
            self.update_target_network()

    def get_buffer_size(self) -> int:
        """Get current replay buffer size"""
        return len(self.replay_buffer)

    def save(self, filepath: str):
        """
        Save agent state

        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'steps_trained': self.steps_trained,
        }, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """
        Load agent state

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episodes_trained = checkpoint['episodes_trained']
        self.steps_trained = checkpoint['steps_trained']
        print(f"Agent loaded from {filepath}")

    def get_statistics(self) -> dict:
        """Get training statistics"""
        return {
            'episodes_trained': self.episodes_trained,
            'steps_trained': self.steps_trained,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': self.total_loss / max(1, self.steps_trained)
        }

    def print_statistics(self):
        """Print training statistics"""
        stats = self.get_statistics()
        print(f"\nAgent Statistics:")
        print(f"  Episodes trained: {stats['episodes_trained']}")
        print(f"  Steps trained: {stats['steps_trained']}")
        print(f"  Current epsilon: {stats['epsilon']:.4f}")
        print(f"  Buffer size: {stats['buffer_size']}")
        print(f"  Avg loss: {stats['avg_loss']:.4f}")
