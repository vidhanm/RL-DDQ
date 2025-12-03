"""
Advanced World Model Architectures
Multiple world model variants for improved prediction and planning

Variants:
1. ProbabilisticWorldModel - Outputs Gaussian distributions (mean, variance)
2. RecurrentWorldModel - LSTM-based for temporal patterns
3. TransformerWorldModel - Self-attention over state-action history
4. EnhancedEnsembleWorldModel - Ensemble with disagreement-based uncertainty

All models share a common interface for easy swapping in DDQ agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Optional, Dict
from abc import ABC, abstractmethod

from config import EnvironmentConfig, DDQConfig, DeviceConfig


# ============================================================================
# BASE CLASS
# ============================================================================

class BaseWorldModel(ABC, nn.Module):
    """Abstract base class for all world models"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    @abstractmethod
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch, state_dim]
            action: One-hot action tensor [batch, action_dim]
            
        Returns:
            (predicted_next_state, predicted_reward)
        """
        pass
    
    def predict(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict without gradients"""
        with torch.no_grad():
            return self.forward(state, action)
    
    def get_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get prediction uncertainty (override in subclasses)"""
        return torch.zeros(state.shape[0])


# ============================================================================
# 1. PROBABILISTIC WORLD MODEL
# ============================================================================

class ProbabilisticWorldModel(BaseWorldModel):
    """
    World model that outputs Gaussian distributions
    
    Instead of point estimates, outputs (mean, log_variance) for both
    state and reward predictions. This enables:
    - Uncertainty-aware planning
    - Better exploration in imagination
    - Calibrated confidence estimates
    """
    
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int = 128,
        min_log_var: float = -10.0,
        max_log_var: float = 2.0
    ):
        super().__init__(state_dim, action_dim)
        
        self.hidden_dim = hidden_dim
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        
        input_dim = state_dim + action_dim
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # State prediction head (mean and log_variance)
        self.state_mean = nn.Linear(hidden_dim, state_dim)
        self.state_log_var = nn.Linear(hidden_dim, state_dim)
        
        # Reward prediction head (mean and log_variance)
        self.reward_mean = nn.Linear(hidden_dim, 1)
        self.reward_log_var = nn.Linear(hidden_dim, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor,
        return_distribution: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor
            action: One-hot action tensor
            return_distribution: If True, return (state_mean, state_var, reward_mean, reward_var)
            
        Returns:
            (predicted_next_state, predicted_reward) or full distribution
        """
        x = torch.cat([state, action], dim=-1)
        features = self.encoder(x)
        
        # State distribution
        state_mean = self.state_mean(features)
        state_log_var = self.state_log_var(features)
        state_log_var = torch.clamp(state_log_var, self.min_log_var, self.max_log_var)
        state_var = torch.exp(state_log_var)
        
        # Reward distribution
        reward_mean = self.reward_mean(features).squeeze(-1)
        reward_log_var = self.reward_log_var(features).squeeze(-1)
        reward_log_var = torch.clamp(reward_log_var, self.min_log_var, self.max_log_var)
        reward_var = torch.exp(reward_log_var)
        
        if return_distribution:
            return state_mean, state_var, reward_mean, reward_var
        
        # For standard interface, return means
        return state_mean, reward_mean
    
    def sample(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the predicted distribution"""
        state_mean, state_var, reward_mean, reward_var = self.forward(state, action, return_distribution=True)
        
        # Reparameterization trick
        state_std = torch.sqrt(state_var)
        reward_std = torch.sqrt(reward_var)
        
        state_sample = state_mean + state_std * torch.randn_like(state_std)
        reward_sample = reward_mean + reward_std * torch.randn_like(reward_std)
        
        return state_sample, reward_sample
    
    def get_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Get prediction uncertainty as mean variance"""
        _, state_var, _, reward_var = self.forward(state, action, return_distribution=True)
        return state_var.mean(dim=-1) + reward_var
    
    def compute_loss(
        self,
        pred_state_mean: torch.Tensor,
        pred_state_var: torch.Tensor,
        pred_reward_mean: torch.Tensor,
        pred_reward_var: torch.Tensor,
        target_state: torch.Tensor,
        target_reward: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss"""
        # State NLL
        state_diff = target_state - pred_state_mean
        state_loss = 0.5 * (torch.log(pred_state_var) + state_diff ** 2 / pred_state_var).mean()
        
        # Reward NLL
        reward_diff = target_reward - pred_reward_mean
        reward_loss = 0.5 * (torch.log(pred_reward_var) + reward_diff ** 2 / pred_reward_var).mean()
        
        return state_loss + reward_loss


# ============================================================================
# 2. RECURRENT WORLD MODEL (LSTM)
# ============================================================================

class RecurrentWorldModel(BaseWorldModel):
    """
    LSTM-based world model for temporal patterns
    
    Maintains hidden state across predictions, useful for:
    - Multi-step rollouts
    - Capturing conversation dynamics
    - Context-dependent predictions
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__(state_dim, action_dim)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        input_dim = state_dim + action_dim
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output heads
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Hidden state storage
        self.hidden = None
        
    def reset_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Reset LSTM hidden state"""
        self.hidden = (
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
            torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device)
        )
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch, state_dim] or [batch, seq, state_dim]
            action: Action tensor [batch, action_dim] or [batch, seq, action_dim]
            hidden: Optional hidden state tuple (h, c)
            
        Returns:
            (predicted_next_state, predicted_reward)
        """
        # Handle 2D input (single step)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]
            action = action.unsqueeze(1)  # [batch, 1, action_dim]
            squeeze_output = True
        else:
            squeeze_output = False
        
        x = torch.cat([state, action], dim=-1)  # [batch, seq, input_dim]
        
        # Initialize hidden if needed
        if hidden is None:
            if self.hidden is None:
                self.reset_hidden(x.shape[0], x.device)
            hidden = self.hidden
        
        # LSTM forward
        lstm_out, self.hidden = self.lstm(x, hidden)  # [batch, seq, hidden_dim]
        
        # Predictions from last timestep
        if squeeze_output:
            last_out = lstm_out.squeeze(1)  # [batch, hidden_dim]
        else:
            last_out = lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        next_state = self.state_head(last_out)
        reward = self.reward_head(last_out).squeeze(-1)
        
        return next_state, reward
    
    def predict_sequence(
        self,
        initial_state: torch.Tensor,
        action_sequence: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Predict a sequence of states given a sequence of actions
        
        Args:
            initial_state: Starting state [batch, state_dim]
            action_sequence: Sequence of actions [batch, seq_len, action_dim]
            
        Returns:
            (list of predicted states, list of predicted rewards)
        """
        self.reset_hidden(initial_state.shape[0], initial_state.device)
        
        states = [initial_state]
        rewards = []
        current_state = initial_state
        
        seq_len = action_sequence.shape[1]
        
        with torch.no_grad():
            for t in range(seq_len):
                action = action_sequence[:, t, :]
                next_state, reward = self.forward(current_state, action)
                states.append(next_state)
                rewards.append(reward)
                current_state = next_state
        
        return states, rewards


# ============================================================================
# 3. TRANSFORMER WORLD MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TransformerWorldModel(BaseWorldModel):
    """
    Transformer-based world model with self-attention
    
    Uses attention mechanism over state-action history for:
    - Long-range dependency modeling
    - Flexible context length
    - Parallel training
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 20,
        dropout: float = 0.1
    ):
        super().__init__(state_dim, action_dim)
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        input_dim = state_dim + action_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Context buffer for incremental prediction
        self.context_buffer: Optional[torch.Tensor] = None
    
    def reset_context(self):
        """Reset context buffer"""
        self.context_buffer = None
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor [batch, state_dim] or [batch, seq, state_dim]
            action: Action tensor [batch, action_dim] or [batch, seq, action_dim]
            context: Optional previous context [batch, prev_seq, hidden_dim]
            
        Returns:
            (predicted_next_state, predicted_reward)
        """
        # Handle 2D input
        if state.dim() == 2:
            state = state.unsqueeze(1)
            action = action.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)  # [batch, seq, input_dim]
        
        # Project to hidden dim
        x = self.input_proj(x)  # [batch, seq, hidden_dim]
        
        # Add context if provided
        if context is not None:
            x = torch.cat([context, x], dim=1)
        
        # Positional encoding
        x = self.pos_encoding(x)
        
        # Causal mask
        seq_len = x.shape[1]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        
        # Transformer forward
        transformer_out = self.transformer(x, mask=causal_mask)  # [batch, seq, hidden_dim]
        
        # Use last position for prediction
        last_out = transformer_out[:, -1, :]  # [batch, hidden_dim]
        
        # Predictions
        next_state = self.state_head(last_out)
        reward = self.reward_head(last_out).squeeze(-1)
        
        if squeeze_output:
            return next_state, reward
        
        return next_state, reward
    
    def predict_with_history(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        current_state: torch.Tensor,
        current_action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using full history context
        
        Args:
            state_history: Previous states [batch, hist_len, state_dim]
            action_history: Previous actions [batch, hist_len, action_dim]
            current_state: Current state [batch, state_dim]
            current_action: Current action [batch, action_dim]
            
        Returns:
            (predicted_next_state, predicted_reward)
        """
        # Combine history with current
        all_states = torch.cat([state_history, current_state.unsqueeze(1)], dim=1)
        all_actions = torch.cat([action_history, current_action.unsqueeze(1)], dim=1)
        
        return self.forward(all_states, all_actions)


# ============================================================================
# 4. ENHANCED ENSEMBLE WORLD MODEL
# ============================================================================

class EnhancedEnsembleWorldModel(nn.Module):
    """
    Enhanced ensemble with disagreement-based uncertainty and selective training
    
    Features:
    - Multiple diverse models (different initializations)
    - Disagreement-based uncertainty estimation
    - Thompson sampling for exploration
    - Selective training based on uncertainty
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_models: int = 5,
        device: str = 'cpu'
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_models = num_models
        self.device = device
        
        # Create diverse ensemble members
        self.models = nn.ModuleList([
            self._create_model(state_dim, action_dim, hidden_dim, seed=i)
            for i in range(num_models)
        ])
        
        # Separate optimizers for diversity
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=DDQConfig.WORLD_MODEL_LEARNING_RATE)
            for model in self.models
        ]
        
        self.to(device)
    
    def _create_model(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int,
        seed: int
    ) -> nn.Module:
        """Create a single ensemble member with specific initialization"""
        torch.manual_seed(seed * 1000)  # Different seed for each model
        
        input_dim = state_dim + action_dim
        
        model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        state_head = nn.Linear(hidden_dim, state_dim)
        reward_head = nn.Linear(hidden_dim, 1)
        
        return nn.ModuleDict({
            'encoder': model,
            'state_head': state_head,
            'reward_head': reward_head
        })
    
    def _forward_single(
        self, 
        model: nn.ModuleDict, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through a single ensemble member"""
        x = torch.cat([state, action], dim=-1)
        features = model['encoder'](x)
        next_state = model['state_head'](features)
        reward = model['reward_head'](features).squeeze(-1)
        return next_state, reward
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through all ensemble members
        
        Returns:
            (mean_next_state, mean_reward, disagreement)
        """
        state_preds = []
        reward_preds = []
        
        for model in self.models:
            next_state, reward = self._forward_single(model, state, action)
            state_preds.append(next_state)
            reward_preds.append(reward)
        
        # Stack predictions
        states_stack = torch.stack(state_preds, dim=0)  # [num_models, batch, state_dim]
        rewards_stack = torch.stack(reward_preds, dim=0)  # [num_models, batch]
        
        # Compute mean
        mean_state = states_stack.mean(dim=0)
        mean_reward = rewards_stack.mean(dim=0)
        
        # Compute disagreement (standard deviation across models)
        state_std = states_stack.std(dim=0).mean(dim=-1)  # [batch]
        reward_std = rewards_stack.std(dim=0)  # [batch]
        disagreement = state_std + reward_std
        
        return mean_state, mean_reward, disagreement
    
    def predict(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict without gradients (returns mean only)"""
        with torch.no_grad():
            mean_state, mean_reward, _ = self.forward(state, action)
        return mean_state, mean_reward
    
    def predict_with_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation"""
        with torch.no_grad():
            return self.forward(state, action)
    
    def thompson_sample(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thompson sampling: randomly select one model for prediction
        Good for exploration in planning
        """
        with torch.no_grad():
            model_idx = np.random.randint(self.num_models)
            return self._forward_single(self.models[model_idx], state, action)
    
    def train_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, float]:
        """Train all ensemble members"""
        total_loss = 0.0
        losses = []
        
        for model, optimizer in zip(self.models, self.optimizers):
            # Forward
            pred_state, pred_reward = self._forward_single(model, states, actions)
            
            # Loss
            state_loss = F.mse_loss(pred_state, next_states)
            reward_loss = F.mse_loss(pred_reward, rewards)
            loss = state_loss + reward_loss
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            losses.append(loss.item())
            total_loss += loss.item()
        
        return {
            'total_loss': total_loss / self.num_models,
            'losses': losses,
            'loss_std': np.std(losses)  # Diversity metric
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_advanced_world_model(
    model_type: str,
    state_dim: int = EnvironmentConfig.STATE_DIM,
    action_dim: int = EnvironmentConfig.NUM_ACTIONS,
    hidden_dim: int = DDQConfig.WORLD_MODEL_HIDDEN_DIM,
    device: str = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create any world model variant
    
    Args:
        model_type: One of ['mlp', 'probabilistic', 'recurrent', 'transformer', 'ensemble']
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer size
        device: Device to use
        **kwargs: Additional arguments for specific models
        
    Returns:
        World model instance
    """
    if device is None:
        device = DeviceConfig.DEVICE
    
    model_type = model_type.lower()
    
    if model_type == 'probabilistic':
        model = ProbabilisticWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif model_type == 'recurrent' or model_type == 'lstm':
        model = RecurrentWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif model_type == 'transformer':
        model = TransformerWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif model_type == 'ensemble':
        model = EnhancedEnsembleWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            device=device,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: probabilistic, recurrent, transformer, ensemble")
    
    return model.to(device)


# ============================================================================
# WORLD MODEL COMPARISON UTILITY
# ============================================================================

class WorldModelComparison:
    """Utility for comparing different world model architectures"""
    
    def __init__(self, state_dim: int, action_dim: int, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.models = {}
        self.histories = {}
    
    def add_model(self, name: str, model: nn.Module):
        """Add a model to comparison"""
        self.models[name] = model.to(self.device)
        self.histories[name] = {'loss': [], 'state_error': [], 'reward_error': []}
    
    def evaluate_all(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate all models on the same data"""
        results = {}
        
        for name, model in self.models.items():
            model.eval()
            with torch.no_grad():
                if isinstance(model, EnhancedEnsembleWorldModel):
                    pred_state, pred_reward, uncertainty = model.forward(states, actions)
                else:
                    pred_state, pred_reward = model(states, actions)
                    uncertainty = torch.zeros(1)
                
                state_error = F.mse_loss(pred_state, next_states).item()
                reward_error = F.mse_loss(pred_reward, rewards).item()
                
                results[name] = {
                    'state_mse': state_error,
                    'reward_mse': reward_error,
                    'total_error': state_error + reward_error,
                    'uncertainty': uncertainty.mean().item() if isinstance(uncertainty, torch.Tensor) else 0
                }
        
        return results
    
    def print_comparison(self, results: Dict[str, Dict[str, float]]):
        """Pretty print comparison results"""
        print("\n" + "="*60)
        print("WORLD MODEL COMPARISON")
        print("="*60)
        print(f"\n{'Model':<20} {'State MSE':>12} {'Reward MSE':>12} {'Total':>12}")
        print("-"*60)
        
        for name, metrics in sorted(results.items(), key=lambda x: x[1]['total_error']):
            print(f"{name:<20} {metrics['state_mse']:>12.6f} {metrics['reward_mse']:>12.6f} {metrics['total_error']:>12.6f}")
        
        print("="*60)
