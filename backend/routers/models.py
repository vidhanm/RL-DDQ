"""
Models Router
Endpoints for loading and managing RL models
"""

from fastapi import APIRouter, HTTPException
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.schemas.models import (
    LoadModelRequest, LoadModelResponse,
    ModelsListResponse, ModelInfo
)
from backend.dependencies import load_model, get_loaded_model_type

from config import EnvironmentConfig, DeviceConfig

router = APIRouter(prefix="/api/models", tags=["models"])

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "checkpoints"
)


@router.get("", response_model=ModelsListResponse)
async def list_models():
    """List all available model checkpoints"""
    models = []
    
    if os.path.exists(CHECKPOINT_DIR):
        for filename in os.listdir(CHECKPOINT_DIR):
            if filename.endswith(".pt"):
                # Determine model type from filename
                if filename.startswith("ddq"):
                    model_type = "ddq"
                elif filename.startswith("dqn"):
                    model_type = "dqn"
                else:
                    model_type = "unknown"
                
                models.append(ModelInfo(
                    name=filename,
                    path=os.path.join(CHECKPOINT_DIR, filename),
                    type=model_type
                ))
    
    return ModelsListResponse(
        models=models,
        loaded_model=get_loaded_model_type()
    )


@router.post("/load", response_model=LoadModelResponse)
async def load_model_endpoint(request: LoadModelRequest):
    """Load a model (DQN or DDQ)"""
    model_type = request.model_type.lower()
    
    if model_type not in ["dqn", "ddq"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid model type: {model_type}. Must be 'dqn' or 'ddq'"
        )
    
    # Try to find checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_final.pt")
    
    if not os.path.exists(checkpoint_path):
        # Try episode checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_type}_episode_100.pt")
    
    if not os.path.exists(checkpoint_path):
        raise HTTPException(
            status_code=404,
            detail=f"No checkpoint found for {model_type.upper()}"
        )
    
    try:
        success = load_model(model_type, checkpoint_path)
        if success:
            return LoadModelResponse(
                success=True,
                message=f"✅ Loaded {model_type.upper()} from {os.path.basename(checkpoint_path)}",
                model_type=model_type
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current")
async def get_current_model():
    """Get currently loaded model info"""
    model_type = get_loaded_model_type()
    return {
        "loaded": model_type is not None,
        "model_type": model_type
    }
