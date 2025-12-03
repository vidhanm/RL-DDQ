"""
Training Router
Endpoints for accessing training history and figures
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import json

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.schemas.models import (
    TrainingHistoryResponse, TrainingHistoryItem,
    FiguresListResponse
)

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
