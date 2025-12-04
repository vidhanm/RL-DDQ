"""
Conversation Router
Endpoints for managing conversations with the RL agent
"""

from fastapi import APIRouter, HTTPException, Depends
import numpy as np
import torch
from typing import List

import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from web.backend.schemas.models import (
    StartConversationRequest, StartConversationResponse,
    ActionRequest, ActionResponse,
    AutoPlayRequest, AutoPlayResponse,
    StateDisplay, QValueItem, ConversationMessage, EpisodeStatus
)
from web.backend.services.session import session_manager
from web.backend.dependencies import get_agent, get_llm_client

from src.environment.nlu_env import NLUDebtCollectionEnv
from src.config import EnvironmentConfig

router = APIRouter(prefix="/api/conversation", tags=["conversation"])

# Action names for display
ACTION_NAMES = [
    "😌 Empathetic Listening",
    "❓ Ask About Situation",
    "📋 Firm Reminder",
    "💳 Offer Payment Plan",
    "🤝 Propose Settlement",
    "⚠️ Hard Close"
]


def format_state(env: NLUDebtCollectionEnv) -> StateDisplay:
    """Format environment state for API response"""
    state = env.state
    return StateDisplay(
        turn=state.turn,
        max_turns=EnvironmentConfig.MAX_TURNS,
        sentiment=round(state.sentiment, 2),
        cooperation=round(state.cooperation, 2),
        engagement=round(state.cooperation, 2),  # Use cooperation as engagement proxy
        mentioned_payment_plan=state.mentioned_payment_plan,
        shared_situation=state.has_shared_situation,
        has_committed=state.has_commitment_signal
    )


def format_q_values(agent, state: np.ndarray) -> List[QValueItem]:
    """Get Q-values from agent for current state"""
    if agent is None or state is None:
        return []
    
    state_tensor = torch.FloatTensor(state).to(agent.device)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).cpu().numpy()
    
    best_action = int(np.argmax(q_values))
    
    return [
        QValueItem(
            action_id=i,
            action_name=ACTION_NAMES[i],
            value=round(float(q), 3),
            is_best=(i == best_action)
        )
        for i, q in enumerate(q_values)
    ]


@router.post("/start", response_model=StartConversationResponse)
async def start_conversation(request: StartConversationRequest):
    """
    Start a new conversation with a debtor persona
    
    Returns session ID, initial state, and Q-values
    """
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")
    
    llm_client = get_llm_client()
    
    # Create NLU environment with domain randomization
    env = NLUDebtCollectionEnv(
        llm_client=llm_client, 
        render_mode=None,
        use_domain_randomization=True
    )
    
    # Start episode (domain randomization creates diverse profiles)
    state, info = env.reset()
    profile_desc = f"randomized (agreeableness={env.profile.agreeableness:.1f})"
    
    # Create session
    session = session_manager.create_session(persona=profile_desc, env=env)
    session.current_state = state
    
    # Get state and Q-values
    state_display = format_state(env)
    q_values = format_q_values(agent, state)
    
    return StartConversationResponse(
        session_id=session.session_id,
        persona=profile_desc,
        state=state_display,
        q_values=q_values,
        message=f"🎬 Started conversation with domain-randomized debtor"
    )


@router.post("/action", response_model=ActionResponse)
async def take_action(request: ActionRequest):
    """
    Take an action in the conversation
    
    If action is None, uses agent's recommended action
    """
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    session = session_manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if session.is_episode_done:
        raise HTTPException(status_code=400, detail="Episode already ended. Start a new conversation.")
    
    env = session.env
    
    # Determine action
    if request.action is not None:
        action = request.action
    else:
        action = agent.select_action(session.current_state, explore=False)
    
    # Take step
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    # Update conversation history
    if env.conversation_history:
        last_turn = env.conversation_history[-1]
        session.conversation_history.append(ConversationMessage(
            agent_utterance=last_turn.get('agent_utterance', f"[Action: {ACTION_NAMES[action]}]"),
            debtor_response=last_turn.get('debtor_response', "[No response]")
        ))
    
    # Update session state
    session.current_state = next_state
    session.is_episode_done = done
    
    # Format response
    state_display = format_state(env)
    q_values = format_q_values(agent, next_state)
    
    # Status message
    if done:
        if env.state.has_commitment_signal:
            message = f"✅ SUCCESS! Debtor committed to payment. Reward: {reward:.2f}"
            success = True
        else:
            message = f"❌ Episode ended without commitment. Reward: {reward:.2f}"
            success = False
    else:
        message = f"Turn {env.state.turn}/{EnvironmentConfig.MAX_TURNS} | Reward: {reward:.2f}"
        success = None
    
    return ActionResponse(
        conversation=session.conversation_history,
        state=state_display,
        q_values=q_values,
        status=EpisodeStatus(
            is_done=done,
            success=success,
            reward=round(reward, 2),
            message=message
        )
    )


@router.post("/auto-play", response_model=AutoPlayResponse)
async def auto_play(request: AutoPlayRequest):
    """
    Let the agent play automatically until episode ends
    """
    agent = get_agent()
    if agent is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    session = session_manager.get_session(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    if session.is_episode_done:
        raise HTTPException(status_code=400, detail="Episode already ended")
    
    env = session.env
    total_reward = 0.0
    
    while not session.is_episode_done:
        # Get agent's action
        action = agent.select_action(session.current_state, explore=False)
        
        # Take step
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Update conversation
        if env.conversation_history:
            last_turn = env.conversation_history[-1]
            session.conversation_history.append(ConversationMessage(
                agent_utterance=last_turn.get('agent_utterance', f"[Action: {ACTION_NAMES[action]}]"),
                debtor_response=last_turn.get('debtor_response', "[No response]")
            ))
        
        session.current_state = next_state
        session.is_episode_done = done
    
    # Final status
    if env.state.has_commitment_signal:
        message = f"✅ SUCCESS! Debtor committed after {env.state.turn} turns"
        success = True
    else:
        message = f"❌ Failed to get commitment after {env.state.turn} turns"
        success = False
    
    return AutoPlayResponse(
        conversation=session.conversation_history,
        state=format_state(env),
        status=EpisodeStatus(
            is_done=True,
            success=success,
            reward=round(total_reward, 2),
            message=message
        ),
        total_reward=round(total_reward, 2)
    )


@router.get("/state/{session_id}")
async def get_state(session_id: str):
    """Get current state and Q-values for a session"""
    agent = get_agent()
    session = session_manager.get_session(session_id)
    
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return {
        "state": format_state(session.env),
        "q_values": format_q_values(agent, session.current_state),
        "conversation": session.conversation_history,
        "is_done": session.is_episode_done
    }


@router.delete("/{session_id}")
async def end_conversation(session_id: str):
    """End and cleanup a conversation session"""
    if session_manager.delete_session(session_id):
        return {"message": "Session ended"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
