"""
Evaluate Router
Endpoints for running model evaluations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import os
import sys
import numpy as np
from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from web.backend.schemas.models import (
    EvaluateRequest, EvaluationResult, ConversationMessage
)
from web.backend.dependencies import get_llm_client

from src.environment.nlu_env import NLUDebtCollectionEnv
from src.agent.dqn_agent import DQNAgent
from src.agent.ddq_agent import DDQAgent
from src.config import EnvironmentConfig, DeviceConfig

router = APIRouter(prefix="/api/evaluate", tags=["evaluate"])

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "data", "checkpoints")


@router.get("/checkpoints")
async def list_checkpoints():
    """List available checkpoint files"""
    checkpoints = []
    
    if os.path.exists(CHECKPOINT_DIR):
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename.endswith(".pt"):
                # Determine algorithm from filename
                if "ddq" in filename.lower():
                    algorithm = "ddq"
                elif "dqn" in filename.lower():
                    algorithm = "dqn"
                else:
                    algorithm = "unknown"
                
                checkpoints.append({
                    "name": filename,
                    "path": os.path.join(CHECKPOINT_DIR, filename),
                    "algorithm": algorithm
                })
    
    return {"checkpoints": checkpoints}


@router.post("/run", response_model=EvaluationResult)
async def run_evaluation(request: EvaluateRequest):
    """
    Run evaluation on a checkpoint
    
    Returns success rate, avg reward, and sample conversations
    """
    # Resolve checkpoint path
    if os.path.isabs(request.checkpoint):
        checkpoint_path = request.checkpoint
    else:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, request.checkpoint)
    
    if not os.path.exists(checkpoint_path):
        raise HTTPException(status_code=404, detail=f"Checkpoint not found: {request.checkpoint}")
    
    # Create agent
    try:
        if request.algorithm.lower() == "ddq":
            agent = DDQAgent(
                state_dim=EnvironmentConfig.NLU_STATE_DIM,
                action_dim=EnvironmentConfig.NUM_ACTIONS,
                device=DeviceConfig.DEVICE
            )
        else:
            agent = DQNAgent(
                state_dim=EnvironmentConfig.NLU_STATE_DIM,
                action_dim=EnvironmentConfig.NUM_ACTIONS,
                device=DeviceConfig.DEVICE
            )
        agent.load(checkpoint_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {e}")
    
    # Get LLM client
    llm_client = None
    if request.use_llm:
        llm_client = get_llm_client()
    
    # Create environment
    env = NLUDebtCollectionEnv(
        llm_client=llm_client,
        render_mode=None,
        use_domain_randomization=True
    )
    
    # Run evaluation
    successes = []
    rewards = []
    lengths = []
    sample_conversations: List[List[ConversationMessage]] = []
    
    for ep in range(request.num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        done = False
        conversation = []
        
        while not done:
            action = agent.select_action(state, explore=False)
            next_state, reward, terminated, truncated, step_info = env.step(action)
            
            episode_reward += reward
            done = terminated or truncated
            state = next_state
            
            # Record conversation
            if env.conversation_history:
                last_turn = env.conversation_history[-1]
                conversation.append(ConversationMessage(
                    agent_utterance=last_turn.get('agent_utterance', '[Action taken]'),
                    debtor_response=last_turn.get('debtor_response', '[Response]')
                ))
        
        successes.append(1 if env.state.has_commitment_signal else 0)
        rewards.append(episode_reward)
        lengths.append(env.state.turn)
        
        # Save first 3 conversations as samples
        if len(sample_conversations) < 3 and conversation:
            sample_conversations.append(conversation)
    
    return EvaluationResult(
        success_rate=round(float(np.mean(successes)), 3),
        avg_reward=round(float(np.mean(rewards)), 2),
        avg_length=round(float(np.mean(lengths)), 1),
        num_episodes=request.num_episodes,
        sample_conversations=sample_conversations
    )
