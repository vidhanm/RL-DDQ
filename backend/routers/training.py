"""
Training Router
Endpoints for accessing training history, figures, and live training via WebSocket
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import os
import json
import asyncio
import threading
from datetime import datetime
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.schemas.models import (
    TrainingHistoryResponse, TrainingHistoryItem,
    FiguresListResponse, StartTrainingRequest, TrainingStatus
)
from backend.dependencies import get_llm_client

from environment.nlu_env import NLUDebtCollectionEnv
from agent.dqn_agent import DQNAgent
from agent.ddq_agent import DDQAgent
from config import EnvironmentConfig, DeviceConfig

router = APIRouter(prefix="/api/training", tags=["training"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")


@router.get("/history", response_model=TrainingHistoryResponse)
async def get_training_history():
    """Get list of training history files with summaries"""
    history = []
    
    if os.path.exists(CHECKPOINT_DIR):
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename.endswith(".json"):
                filepath = os.path.join(CHECKPOINT_DIR, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Handle both old and new format
                    if "metadata" in data:
                        # New format
                        meta = data["metadata"]
                        episodes = data.get("episodes", [])
                        rewards = [ep.get("reward", 0) for ep in episodes]
                        successes = [ep.get("success", False) for ep in episodes]
                        
                        history.append(TrainingHistoryItem(
                            filename=filename,
                            algorithm=meta.get("algorithm", "unknown"),
                            episodes=len(episodes),
                            avg_reward=round(sum(rewards) / len(rewards), 2) if rewards else 0,
                            success_rate=round(sum(successes) / len(successes) * 100, 1) if successes else 0,
                            timestamp=meta.get("timestamp")
                        ))
                    else:
                        # Old format - flat list
                        rewards = data if isinstance(data, list) else []
                        algorithm = "dqn" if "dqn" in filename.lower() else "ddq" if "ddq" in filename.lower() else "unknown"
                        
                        history.append(TrainingHistoryItem(
                            filename=filename,
                            algorithm=algorithm,
                            episodes=len(rewards),
                            avg_reward=round(sum(rewards) / len(rewards), 2) if rewards else 0,
                            success_rate=0,  # Not available in old format
                            timestamp=None
                        ))
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    continue
    
    return TrainingHistoryResponse(history=history)


@router.get("/history/{filename}")
async def get_history_file(filename: str):
    """Get full content of a training history file"""
    filepath = os.path.join(CHECKPOINT_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="History file not found")
    
    if not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/figures", response_model=FiguresListResponse)
async def list_figures():
    """List available training figures"""
    figures = []
    
    if os.path.exists(FIGURES_DIR):
        for filename in os.listdir(FIGURES_DIR):
            if filename.endswith((".png", ".jpg", ".svg")):
                figures.append(filename)
    
    return FiguresListResponse(figures=figures)


@router.get("/figures/{filename}")
async def get_figure(filename: str):
    """Serve a training figure image"""
    filepath = os.path.join(FIGURES_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Figure not found")
    
    # Determine media type
    if filename.endswith(".png"):
        media_type = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media_type = "image/jpeg"
    elif filename.endswith(".svg"):
        media_type = "image/svg+xml"
    else:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    return FileResponse(filepath, media_type=media_type)


# ============== Training State ==============

class TrainingManager:
    """Manages training state and WebSocket connections"""
    
    def __init__(self):
        self.is_training = False
        self.should_stop = False
        self.current_episode = 0
        self.total_episodes = 0
        self.algorithm = None
        self.successes = []
        self.websocket: Optional[WebSocket] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    def get_status(self) -> TrainingStatus:
        success_rate = sum(self.successes) / len(self.successes) if self.successes else 0.0
        return TrainingStatus(
            is_training=self.is_training,
            algorithm=self.algorithm,
            current_episode=self.current_episode,
            total_episodes=self.total_episodes,
            success_rate=round(success_rate, 3)
        )
    
    async def send_message(self, message: dict):
        """Send message to WebSocket client"""
        if self.websocket:
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                print(f"WebSocket send error: {e}")
    
    def send_message_sync(self, message: dict):
        """Send message from sync context"""
        if self.websocket and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.send_message(message), 
                self.loop
            )


training_manager = TrainingManager()


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status"""
    return training_manager.get_status()


@router.post("/stop")
async def stop_training():
    """Request to stop training"""
    if not training_manager.is_training:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    training_manager.should_stop = True
    return {"message": "Stop requested"}


def run_training_loop(
    algorithm: str,
    episodes: int,
    use_llm: bool,
    difficulty: str
):
    """Run training in background thread"""
    try:
        # Get LLM client
        llm_client = None
        if use_llm:
            llm_client = get_llm_client()
        
        # Create environment
        env = NLUDebtCollectionEnv(
            llm_client=llm_client,
            render_mode=None,
            use_domain_randomization=True
        )
        
        # Create agent
        if algorithm.lower() == "ddq":
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
        
        training_manager.successes = []
        
        # Training loop
        for ep in range(episodes):
            if training_manager.should_stop:
                training_manager.send_message_sync({
                    "type": "stopped",
                    "message": f"Training stopped at episode {ep}"
                })
                break
            
            training_manager.current_episode = ep + 1
            
            # Curriculum difficulty
            if difficulty == "curriculum":
                progress = ep / episodes
                if progress < 0.3:
                    ep_difficulty = "easy"
                elif progress < 0.7:
                    ep_difficulty = "medium"
                else:
                    ep_difficulty = "hard"
            else:
                ep_difficulty = difficulty if difficulty != "random" else None
            
            # Run episode
            state, _ = env.reset(options={"difficulty": ep_difficulty} if ep_difficulty else None)
            episode_reward = 0.0
            done = False
            
            while not done:
                action = agent.select_action(state, explore=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                
                # Store experience
                agent.remember(state, action, reward, next_state, terminated or truncated)
                
                # Train
                if len(agent.memory) >= agent.batch_size:
                    agent.train_step()
                
                episode_reward += reward
                done = terminated or truncated
                state = next_state
                
                # Send dialogue update
                if env.conversation_history:
                    last_turn = env.conversation_history[-1]
                    training_manager.send_message_sync({
                        "type": "dialogue",
                        "episode": ep + 1,
                        "agent_utterance": last_turn.get('agent_utterance', ''),
                        "debtor_response": last_turn.get('debtor_response', '')
                    })
            
            # Episode complete
            success = env.state.has_commitment_signal
            training_manager.successes.append(1 if success else 0)
            success_rate = sum(training_manager.successes) / len(training_manager.successes)
            
            training_manager.send_message_sync({
                "type": "episode",
                "episode": ep + 1,
                "total_episodes": episodes,
                "reward": round(episode_reward, 2),
                "success": success,
                "success_rate": round(success_rate, 3)
            })
            
            # Update target network periodically
            if (ep + 1) % 10 == 0:
                agent.update_target_network()
        
        # Save checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{algorithm}_{episodes}ep_{timestamp}.pt")
        agent.save(checkpoint_path)
        
        # Training complete
        final_success_rate = sum(training_manager.successes) / len(training_manager.successes) if training_manager.successes else 0
        training_manager.send_message_sync({
            "type": "complete",
            "total_episodes": len(training_manager.successes),
            "success_rate": round(final_success_rate, 3),
            "checkpoint": checkpoint_path,
            "message": f"Training complete! Saved to {checkpoint_path}"
        })
        
    except Exception as e:
        training_manager.send_message_sync({
            "type": "error",
            "message": str(e)
        })
    finally:
        training_manager.is_training = False
        training_manager.should_stop = False


@router.websocket("/ws")
async def training_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live training updates
    
    Client sends: {"action": "start", "algorithm": "ddq", "episodes": 100, "use_llm": true}
    Server sends: {"type": "episode", "episode": 1, "reward": 5.2, "success": true, ...}
    """
    await websocket.accept()
    training_manager.websocket = websocket
    training_manager.loop = asyncio.get_event_loop()
    
    try:
        while True:
            # Wait for messages from client
            data = await websocket.receive_json()
            
            if data.get("action") == "start":
                if training_manager.is_training:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Training already in progress"
                    })
                    continue
                
                # Start training in background thread
                training_manager.is_training = True
                training_manager.should_stop = False
                training_manager.algorithm = data.get("algorithm", "ddq")
                training_manager.total_episodes = data.get("episodes", 100)
                training_manager.current_episode = 0
                
                await websocket.send_json({
                    "type": "started",
                    "message": f"Starting {training_manager.algorithm.upper()} training for {training_manager.total_episodes} episodes"
                })
                
                # Run training in thread
                thread = threading.Thread(
                    target=run_training_loop,
                    args=(
                        training_manager.algorithm,
                        training_manager.total_episodes,
                        data.get("use_llm", True),
                        data.get("difficulty", "curriculum")
                    )
                )
                thread.start()
            
            elif data.get("action") == "stop":
                if training_manager.is_training:
                    training_manager.should_stop = True
                    await websocket.send_json({
                        "type": "stopping",
                        "message": "Stop requested, finishing current episode..."
                    })
            
            elif data.get("action") == "status":
                await websocket.send_json({
                    "type": "status",
                    **training_manager.get_status().model_dump()
                })
    
    except WebSocketDisconnect:
        print("Training WebSocket disconnected")
    finally:
        training_manager.websocket = None
