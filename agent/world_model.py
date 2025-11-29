"""
World Model for DDQ
Learns to predict (state, action) -> (next_state, reward)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

from config import EnvironmentConfig, DDQConfig, DeviceConfig


class WorldModel(nn.Module):
    """World Model predicts environment dynamics"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize World Model

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer size
        """
        super(WorldModel, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input: state + action (one-hot)
        input_dim = state_dim + action_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # State predictor
        self.state_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]
            action: Action tensor (one-hot) [batch_size, action_dim] or [action_dim]

        Returns:
            (predicted_next_state, predicted_reward)
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Encode
        features = self.encoder(x)

        # Predict next state and reward
        next_state = self.state_predictor(features)
        reward = self.reward_predictor(features).squeeze(-1)

        return next_state, reward

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict without gradient tracking (for imagination)

        Args:
            state: State tensor
            action: Action tensor (one-hot)

        Returns:
            (predicted_next_state, predicted_reward)
        """
        with torch.no_grad():
            return self.forward(state, action)

    def save(self, filepath: str):
        """Save model weights"""
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim
        }, filepath)

    def load(self, filepath: str):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['state_dict'])


class EnsembleWorldModel:
    """
    Ensemble of World Models for uncertainty estimation
    Optional advanced feature
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_models: int = 5,
        device: str = "cpu"
    ):
        """
        Initialize ensemble

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            num_models: Number of models in ensemble
            device: Device to use
        """
        self.num_models = num_models
        self.device = device

        # Create ensemble
        self.models = [
            WorldModel(state_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_models)
        ]

        # Separate optimizers
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=DDQConfig.WORLD_MODEL_LEARNING_RATE)
            for model in self.models
        ]

    def predict(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict with ensemble and return mean + variance

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            (mean_next_state, mean_reward, variance)
        """
        predictions_state = []
        predictions_reward = []

        for model in self.models:
            next_state, reward = model.predict(state, action)
            predictions_state.append(next_state)
            predictions_reward.append(reward)

        # Stack predictions
        states_stacked = torch.stack(predictions_state)
        rewards_stacked = torch.stack(predictions_reward)

        # Compute mean and variance
        mean_state = states_stacked.mean(dim=0)
        mean_reward = rewards_stacked.mean(dim=0)
        variance = states_stacked.var(dim=0).mean() + rewards_stacked.var(dim=0).mean()

        return mean_state, mean_reward, variance

    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                   next_states: torch.Tensor, rewards: torch.Tensor) -> float:
        """Train all models in ensemble"""
        total_loss = 0.0

        for model, optimizer in zip(self.models, self.optimizers):
            # Forward pass
            pred_next_state, pred_reward = model(states, actions)

            # Compute loss
            state_loss = F.mse_loss(pred_next_state, next_states)
            reward_loss = F.mse_loss(pred_reward, rewards)
            loss = state_loss + reward_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / self.num_models


class WorldModelTrainer:
    """Trainer for World Model"""

    def __init__(
        self,
        world_model: WorldModel,
        learning_rate: float = DDQConfig.WORLD_MODEL_LEARNING_RATE,
        device: str = None
    ):
        """
        Initialize trainer

        Args:
            world_model: World model to train
            learning_rate: Learning rate
            device: Device to use
        """
        self.world_model = world_model
        self.device = device if device else DeviceConfig.DEVICE

        self.optimizer = torch.optim.Adam(
            world_model.parameters(),
            lr=learning_rate
        )

        self.criterion_state = nn.MSELoss()
        self.criterion_reward = nn.MSELoss()

        # Statistics
        self.total_steps = 0
        self.total_loss = 0.0
        self.state_loss_history = []
        self.reward_loss_history = []

    def train_on_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Train on a batch of experiences

        Args:
            states: Batch of states [batch_size, state_dim]
            actions: Batch of actions (one-hot) [batch_size, action_dim]
            next_states: Batch of next states [batch_size, state_dim]
            rewards: Batch of rewards [batch_size]

        Returns:
            (total_loss, state_loss, reward_loss)
        """
        # Forward pass
        pred_next_states, pred_rewards = self.world_model(states, actions)

        # Compute losses
        state_loss = self.criterion_state(pred_next_states, next_states)
        reward_loss = self.criterion_reward(pred_rewards, rewards)
        total_loss = state_loss + reward_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update statistics
        self.total_steps += 1
        self.total_loss += total_loss.item()
        self.state_loss_history.append(state_loss.item())
        self.reward_loss_history.append(reward_loss.item())

        return total_loss.item(), state_loss.item(), reward_loss.item()

    def train_on_buffer(
        self,
        replay_buffer,
        batch_size: int = 32,
        num_epochs: int = 5
    ) -> dict:
        """
        Train on replay buffer

        Args:
            replay_buffer: ReplayBuffer with experiences
            batch_size: Batch size
            num_epochs: Number of epochs

        Returns:
            Training statistics
        """
        if len(replay_buffer) < batch_size:
            return {'total_loss': 0, 'state_loss': 0, 'reward_loss': 0, 'num_batches': 0}

        epoch_losses = []
        num_batches = 0

        for epoch in range(num_epochs):
            # Sample batch
            states, actions_idx, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Convert action indices to one-hot
            actions = torch.zeros(batch_size, self.world_model.action_dim)
            actions.scatter_(1, actions_idx.unsqueeze(1), 1.0)

            # Move to device
            states = states.to(self.device)
            actions = actions.to(self.device)
            next_states = next_states.to(self.device)
            rewards = rewards.to(self.device)

            # Train
            total_loss, state_loss, reward_loss = self.train_on_batch(
                states, actions, next_states, rewards
            )

            epoch_losses.append(total_loss)
            num_batches += 1

        return {
            'total_loss': np.mean(epoch_losses) if epoch_losses else 0,
            'state_loss': np.mean(self.state_loss_history[-num_batches:]) if self.state_loss_history else 0,
            'reward_loss': np.mean(self.reward_loss_history[-num_batches:]) if self.reward_loss_history else 0,
            'num_batches': num_batches
        }

    def get_statistics(self) -> dict:
        """Get training statistics"""
        return {
            'total_steps': self.total_steps,
            'avg_total_loss': self.total_loss / max(1, self.total_steps),
            'avg_state_loss': np.mean(self.state_loss_history[-100:]) if self.state_loss_history else 0,
            'avg_reward_loss': np.mean(self.reward_loss_history[-100:]) if self.reward_loss_history else 0,
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_world_model(
    state_dim: int = EnvironmentConfig.STATE_DIM,
    action_dim: int = EnvironmentConfig.NUM_ACTIONS,
    hidden_dim: int = DDQConfig.WORLD_MODEL_HIDDEN_DIM,
    device: str = None
) -> WorldModel:
    """
    Create world model

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer size
        device: Device to use

    Returns:
        WorldModel instance
    """
    if device is None:
        device = DeviceConfig.DEVICE

    model = WorldModel(state_dim, action_dim, hidden_dim)
    model = model.to(device)
    return model
