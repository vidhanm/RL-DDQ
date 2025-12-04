"""
Deep Q-Network (DQN)
Neural network for Q-value approximation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.config import EnvironmentConfig, RLConfig


class DQN(nn.Module):
    """Deep Q-Network for action-value approximation"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize DQN

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer size
        """
        super(DQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Network architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)

        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]

        Returns:
            Q-values for each action [batch_size, action_dim] or [action_dim]
        """
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)

        q_values = self.fc4(x)

        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Get action using epsilon-greedy policy

        Args:
            state: State tensor [state_dim]
            epsilon: Exploration rate

        Returns:
            Action index
        """
        if torch.rand(1).item() < epsilon:
            # Explore: random action
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            # Exploit: best action
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()

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


class DuelingDQN(nn.Module):
    """
    Dueling DQN Architecture (optional advanced feature)
    Separates state value and action advantages
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        Initialize Dueling DQN

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer size
        """
        super(DuelingDQN, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared feature extraction
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture

        Args:
            state: State tensor [batch_size, state_dim] or [state_dim]

        Returns:
            Q-values [batch_size, action_dim] or [action_dim]
        """
        features = self.feature_layer(state)

        # Compute value and advantages
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine using dueling formula: Q = V + (A - mean(A))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))

        return q_values

    def get_action(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """Get action using epsilon-greedy policy"""
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.argmax().item()

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


# ============================================================================
# Helper Functions
# ============================================================================

def create_dqn(
    state_dim: int = EnvironmentConfig.STATE_DIM,
    action_dim: int = EnvironmentConfig.NUM_ACTIONS,
    hidden_dim: int = RLConfig.HIDDEN_DIM,
    dueling: bool = False,
    device: str = "cpu"
) -> nn.Module:
    """
    Create DQN network

    Args:
        state_dim: State space dimension
        action_dim: Number of actions
        hidden_dim: Hidden layer size
        dueling: Whether to use Dueling DQN architecture
        device: Device to place network on

    Returns:
        DQN network
    """
    if dueling:
        network = DuelingDQN(state_dim, action_dim, hidden_dim)
    else:
        network = DQN(state_dim, action_dim, hidden_dim)

    network = network.to(device)
    return network


def copy_weights(source_network: nn.Module, target_network: nn.Module):
    """
    Copy weights from source to target network

    Args:
        source_network: Network to copy from
        target_network: Network to copy to
    """
    target_network.load_state_dict(source_network.state_dict())
