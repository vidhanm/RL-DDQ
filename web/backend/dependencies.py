"""
Dependencies for FastAPI
Shared state for models and LLM client
"""

import os
import sys
from typing import Optional

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config import EnvironmentConfig, DeviceConfig

# Global state
_agent = None
_llm_client = None
_loaded_model_type = None


def initialize_llm_client():
    """Initialize the LLM client"""
    global _llm_client
    
    try:
        from src.llm.nvidia_client import NVIDIAClient
        _llm_client = NVIDIAClient()
        print("[OK] LLM client initialized")
        return True
    except Exception as e:
        print(f"[WARN] LLM client failed: {e}")
        print("  Demo will run without actual LLM responses")
        return False


def load_model(model_type: str, checkpoint_path: str) -> bool:
    """
    Load a DQN or DDQ model
    
    Args:
        model_type: "dqn" or "ddq"
        checkpoint_path: Path to checkpoint file
        
    Returns:
        True if successful
    """
    global _agent, _loaded_model_type
    
    try:
        if model_type == "ddq":
            from src.agent.ddq_agent import DDQAgent
            _agent = DDQAgent(
                state_dim=EnvironmentConfig.NLU_STATE_DIM,
                action_dim=EnvironmentConfig.NUM_ACTIONS,
                device=DeviceConfig.DEVICE
            )
        else:
            from src.agent.dqn_agent import DQNAgent
            _agent = DQNAgent(
                state_dim=EnvironmentConfig.NLU_STATE_DIM,
                action_dim=EnvironmentConfig.NUM_ACTIONS,
                device=DeviceConfig.DEVICE
            )
        
        _agent.load(checkpoint_path)
        _loaded_model_type = model_type
        print(f"[OK] Loaded {model_type.upper()} from {checkpoint_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        _agent = None
        _loaded_model_type = None
        raise


def get_agent():
    """Get the loaded agent (or None if not loaded)"""
    return _agent


def get_llm_client():
    """Get the LLM client (or None if not available)"""
    return _llm_client


def get_loaded_model_type() -> Optional[str]:
    """Get the type of currently loaded model"""
    return _loaded_model_type


def preload_models(checkpoint_dir: str = "checkpoints"):
    """
    Preload both models at startup for faster switching
    
    Returns dict of loaded models
    """
    models = {}
    
    for model_type in ["dqn", "ddq"]:
        checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_final.pt")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_type}_episode_100.pt")
        
        if os.path.exists(checkpoint_path):
            try:
                if model_type == "ddq":
                    from src.agent.ddq_agent import DDQAgent
                    agent = DDQAgent(
                        state_dim=EnvironmentConfig.NLU_STATE_DIM,
                        action_dim=EnvironmentConfig.NUM_ACTIONS,
                        device=DeviceConfig.DEVICE
                    )
                else:
                    from src.agent.dqn_agent import DQNAgent
                    agent = DQNAgent(
                        state_dim=EnvironmentConfig.NLU_STATE_DIM,
                        action_dim=EnvironmentConfig.NUM_ACTIONS,
                        device=DeviceConfig.DEVICE
                    )
                
                agent.load(checkpoint_path)
                models[model_type] = agent
                print(f"[OK] Preloaded {model_type.upper()}")
            except Exception as e:
                print(f"[WARN] Failed to preload {model_type}: {e}")
    
    return models
