"""
DQN Agent
Combines DQN network with training logic and replay buffer
Now with Double DQN and optional Prioritized Experience Replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional

from agent.dqn import create_dqn, copy_weights
from utils.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from config import RLConfig, EnvironmentConfig, DeviceConfig


class DQNAgent:
    """DQN Agent for debt collection (Double DQN + optional PER)"""

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
        use_prioritized_replay: bool = True,
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
            use_prioritized_replay: If True, use PER with TD-error priorities
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
        self.use_prioritized_replay = use_prioritized_replay

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
        self.criterion = nn.MSELoss(reduction='none')  # Per-sample loss for PER

        # Replay buffer (prioritized or uniform)
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        # PER hyperparameters
        self.per_beta_start = 0.4
        self.per_beta_end = 1.0
        self.per_beta = self.per_beta_start

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
        Perform one training step (Double DQN + optional PER)

        Returns:
            Loss value
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # Sample batch (different return for PER vs uniform)
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, weights, indices = \
                self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
            weights = weights.to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
            weights = None
            indices = None

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Double DQN
        # Step 6 of CRITICAL_FIXES: Reduces Q-value overestimation
        with torch.no_grad():
            # Double DQN: Use policy net to SELECT action, target net to EVALUATE
            best_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD errors (per-sample)
        td_errors = current_q_values - target_q_values
        losses = self.criterion(current_q_values, target_q_values)  # Per-sample MSE

        # Apply importance sampling weights for PER
        if self.use_prioritized_replay and weights is not None:
            weighted_losses = losses * weights
            loss = weighted_losses.mean()
            
            # Update priorities based on TD errors
            self.replay_buffer.update_priorities(indices, td_errors.abs().detach().cpu().numpy())
        else:
            loss = losses.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
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

    def episode_done(self, episode: int, total_episodes: int = 500):
        """
        Called at end of episode

        Args:
            episode: Episode number
            total_episodes: Total episodes for annealing schedules
        """
        self.episodes_trained += 1

        # Update epsilon
        self.update_epsilon()

        # Anneal PER beta from 0.4 to 1.0 over training
        # Step 6 of CRITICAL_FIXES: Beta controls importance sampling correction
        if self.use_prioritized_replay:
            progress = min(1.0, episode / total_episodes)
            self.per_beta = 0.4 + (1.0 - 0.4) * progress

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
