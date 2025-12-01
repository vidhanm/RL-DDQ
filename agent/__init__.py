"""Agent module for RL algorithms"""

from agent.dqn import DQN, DuelingDQN, create_dqn
from agent.dqn_agent import DQNAgent
from agent.world_model import WorldModel, create_world_model, WorldModelTrainer
from agent.ddq_agent import DDQAgent

# Advanced world model variants
from agent.advanced_world_models import (
    BaseWorldModel,
    ProbabilisticWorldModel,
    RecurrentWorldModel,
    TransformerWorldModel,
    EnhancedEnsembleWorldModel,
    create_advanced_world_model,
    WorldModelComparison
)

# Multi-step planning
from agent.multistep_planning import (
    BasePlanner,
    RolloutPlanner,
    TreeSearchPlanner,
    MPCPlanner,
    UncertaintyAwarePlanner,
    create_planner,
    PlannerComparison
)

__all__ = [
    # Core DQN
    'DQN', 'DuelingDQN', 'create_dqn',
    'DQNAgent',
    
    # Original world model
    'WorldModel', 'create_world_model', 'WorldModelTrainer',
    
    # DDQ Agent
    'DDQAgent',
    
    # Advanced world models
    'BaseWorldModel',
    'ProbabilisticWorldModel',
    'RecurrentWorldModel',
    'TransformerWorldModel',
    'EnhancedEnsembleWorldModel',
    'create_advanced_world_model',
    'WorldModelComparison',
    
    # Multi-step planning
    'BasePlanner',
    'RolloutPlanner',
    'TreeSearchPlanner',
    'MPCPlanner',
    'UncertaintyAwarePlanner',
    'create_planner',
    'PlannerComparison'
]
