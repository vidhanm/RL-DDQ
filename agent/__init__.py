"""Agent module for RL algorithms"""

from agent.dqn import DQN, DuelingDQN, create_dqn
from agent.dqn_agent import DQNAgent
from agent.world_model import WorldModel, create_world_model, WorldModelTrainer
from agent.ddq_agent import DDQAgent

__all__ = [
    'DQN', 'DuelingDQN', 'create_dqn',
    'DQNAgent',
    'WorldModel', 'create_world_model', 'WorldModelTrainer',
    'DDQAgent'
]
