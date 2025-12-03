"""
State Encoder/Decoder
Converts between dictionary state representations and numerical vectors for neural networks
"""

import torch
import numpy as np
from typing import Dict, List
from config import EnvironmentConfig


class StateEncoder:
    """Encodes conversation state into fixed-size numerical vector"""

    def __init__(self):
        """Initialize encoder with configuration"""
        self.state_dim = EnvironmentConfig.STATE_DIM
        self.num_actions = EnvironmentConfig.NUM_ACTIONS

    def encode(self, state_dict: Dict) -> torch.Tensor:
        """
        Convert state dictionary to numerical vector

        Args:
            state_dict: Dictionary containing state information

        Returns:
            torch.Tensor of shape [state_dim]
        """
        features = []

        # Turn number (normalized)
        turn = state_dict.get("turn", 0)
        features.append(turn / EnvironmentConfig.MAX_TURNS)

        # Debtor attributes (already normalized)
        features.append(state_dict.get("debtor_sentiment", 0.0))
        features.append(state_dict.get("debtor_cooperation", 0.0))
        features.append(state_dict.get("debtor_engagement", 0.5))
        features.append(state_dict.get("financial_stress", 0.7))

        # Binary flags
        features.append(1.0 if state_dict.get("mentioned_payment_plan", False) else 0.0)
        features.append(1.0 if state_dict.get("mentioned_consequences", False) else 0.0)
        features.append(1.0 if state_dict.get("debtor_shared_situation", False) else 0.0)
        features.append(1.0 if state_dict.get("feels_understood", False) else 0.0)
        features.append(1.0 if state_dict.get("has_committed", False) else 0.0)

        # Trends
        features.append(state_dict.get("sentiment_trend", 0.0))
        features.append(state_dict.get("cooperation_trend", 0.0))

        # Last action (one-hot encoded - 6 dimensions)
        last_action = state_dict.get("agent_last_action", None)
        action_one_hot = self._one_hot_action(last_action)
        features.extend(action_one_hot)

        # NOTE: Persona type removed from state encoding (Step 3 of CRITICAL_FIXES)
        # Agent should not have access to ground-truth persona during training
        # This forces learning robust strategies that work across personas
        # State dim is now 18: 1 turn + 4 attrs + 5 flags + 2 trends + 6 action

        # Ensure correct dimension
        vector = torch.tensor(features[:self.state_dim], dtype=torch.float32)

        # Pad if necessary
        if len(vector) < self.state_dim:
            padding = torch.zeros(self.state_dim - len(vector))
            vector = torch.cat([vector, padding])

        return vector

    def encode_batch(self, state_dicts: List[Dict]) -> torch.Tensor:
        """
        Encode a batch of states

        Args:
            state_dicts: List of state dictionaries

        Returns:
            torch.Tensor of shape [batch_size, state_dim]
        """
        encoded = [self.encode(state) for state in state_dicts]
        return torch.stack(encoded)

    def _one_hot_action(self, action: int) -> List[float]:
        """One-hot encode action (or zeros if None)"""
        one_hot = [0.0] * self.num_actions
        if action is not None and 0 <= action < self.num_actions:
            one_hot[action] = 1.0
        return one_hot

    def _one_hot_persona(self, persona: str) -> List[float]:
        """One-hot encode persona type"""
        personas = EnvironmentConfig.PERSONAS
        one_hot = [0.0] * len(personas)
        if persona in personas:
            idx = personas.index(persona)
            one_hot[idx] = 1.0
        return one_hot

    def action_to_one_hot(self, action: int) -> torch.Tensor:
        """
        Convert action index to one-hot tensor

        Args:
            action: Action index (0 to num_actions-1)

        Returns:
            torch.Tensor of shape [num_actions]
        """
        one_hot = torch.zeros(self.num_actions)
        if 0 <= action < self.num_actions:
            one_hot[action] = 1.0
        return one_hot

    def actions_to_one_hot_batch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert batch of action indices to one-hot

        Args:
            actions: Tensor of shape [batch_size] with action indices

        Returns:
            torch.Tensor of shape [batch_size, num_actions]
        """
        batch_size = actions.shape[0]
        one_hot = torch.zeros(batch_size, self.num_actions)
        one_hot.scatter_(1, actions.unsqueeze(1), 1.0)
        return one_hot


def create_state_dict(
    turn: int,
    debtor_state: Dict,
    agent_last_action: int = None,
    mentioned_payment_plan: bool = False,
    mentioned_consequences: bool = False,
    conversation_summary: str = ""
) -> Dict:
    """
    Helper function to create a state dictionary

    Args:
        turn: Current turn number
        debtor_state: Dictionary from DebtorPersona.to_dict()
        agent_last_action: Last action taken by agent
        mentioned_payment_plan: Whether payment plan was mentioned
        mentioned_consequences: Whether consequences were mentioned
        conversation_summary: Text summary of recent conversation

    Returns:
        State dictionary ready for encoding
    """
    state = {
        "turn": turn,
        "debtor_sentiment": debtor_state.get("sentiment", 0.0),
        "debtor_cooperation": debtor_state.get("cooperation", 0.5),
        "debtor_engagement": debtor_state.get("engagement", 0.5),
        "financial_stress": debtor_state.get("financial_stress", 0.7),
        "mentioned_payment_plan": mentioned_payment_plan,
        "mentioned_consequences": mentioned_consequences,
        "debtor_shared_situation": debtor_state.get("has_shared_situation", False),
        "feels_understood": debtor_state.get("feels_understood", False),
        "has_committed": debtor_state.get("has_committed", False),
        "sentiment_trend": debtor_state.get("sentiment_trend", 0.0),
        "cooperation_trend": debtor_state.get("cooperation_trend", 0.0),
        "agent_last_action": agent_last_action,
        "persona_type": debtor_state.get("persona_type", None),
        "conversation_summary": conversation_summary,
    }
    return state


# Global encoder instance
_encoder = None


def get_encoder() -> StateEncoder:
    """Get global state encoder instance"""
    global _encoder
    if _encoder is None:
        _encoder = StateEncoder()
    return _encoder
