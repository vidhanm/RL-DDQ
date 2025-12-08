"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ============== Request Schemas ==============

class LoadModelRequest(BaseModel):
    """Request to load a model"""
    model_type: str  # "dqn" or "ddq"


class StartConversationRequest(BaseModel):
    """Request to start a new conversation"""
    difficulty: str = "random"  # "easy", "medium", "hard", or "random"


class ActionRequest(BaseModel):
    """Request to take an action"""
    session_id: str
    action: Optional[int] = None  # If None, use agent's recommended action


class AutoPlayRequest(BaseModel):
    """Request to auto-play episode"""
    session_id: str


# ============== Response Schemas ==============

class QValueItem(BaseModel):
    """Single Q-value for an action"""
    action_id: int
    action_name: str
    value: float
    is_best: bool


class StateDisplay(BaseModel):
    """Current environment state for display"""
    turn: int
    max_turns: int
    sentiment: float
    cooperation: float
    engagement: float
    mentioned_payment_plan: bool
    shared_situation: bool
    has_committed: bool
    # NLU-specific fields
    intent: Optional[str] = None
    feels_understood: Optional[bool] = None


class ConversationMessage(BaseModel):
    """A single conversation turn"""
    agent_utterance: str
    debtor_response: str


class EpisodeStatus(BaseModel):
    """Current episode status"""
    is_done: bool
    success: Optional[bool] = None
    reward: float
    message: str


class LoadModelResponse(BaseModel):
    """Response after loading a model"""
    success: bool
    message: str
    model_type: Optional[str] = None


class StartConversationResponse(BaseModel):
    """Response after starting conversation"""
    session_id: str
    persona: str
    state: StateDisplay
    q_values: List[QValueItem]
    message: str
    initial_message: str = "Hello?"  # Debtor's initial greeting when picking up


class ActionResponse(BaseModel):
    """Response after taking an action"""
    conversation: List[ConversationMessage]
    state: StateDisplay
    q_values: List[QValueItem]
    status: EpisodeStatus


class AutoPlayResponse(BaseModel):
    """Response after auto-play completes"""
    conversation: List[ConversationMessage]
    state: StateDisplay
    status: EpisodeStatus
    total_reward: float


class ModelInfo(BaseModel):
    """Information about an available model"""
    name: str
    path: str
    type: str  # "dqn" or "ddq"


class ModelsListResponse(BaseModel):
    """List of available models"""
    models: List[ModelInfo]
    loaded_model: Optional[str] = None


class TrainingHistoryItem(BaseModel):
    """Summary of a training run"""
    filename: str
    algorithm: str
    episodes: int
    avg_reward: float
    success_rate: float
    timestamp: Optional[str] = None


class TrainingHistoryResponse(BaseModel):
    """List of training history files"""
    history: List[TrainingHistoryItem]


class FiguresListResponse(BaseModel):
    """List of available figures"""
    figures: List[str]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    llm_available: bool
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


# ============== Training Schemas ==============

class StartTrainingRequest(BaseModel):
    """Request to start training"""
    algorithm: str = "ddq"  # "dqn" or "ddq"
    episodes: int = 100
    use_llm: bool = True
    difficulty: str = "curriculum"  # "easy", "medium", "hard", "random", "curriculum"


class TrainingProgress(BaseModel):
    """Progress update during training (WebSocket message)"""
    type: str  # "episode", "dialogue", "complete", "error"
    episode: Optional[int] = None
    total_episodes: Optional[int] = None
    reward: Optional[float] = None
    success: Optional[bool] = None
    success_rate: Optional[float] = None
    agent_utterance: Optional[str] = None
    debtor_response: Optional[str] = None
    message: Optional[str] = None


class TrainingStatus(BaseModel):
    """Current training status"""
    is_training: bool
    algorithm: Optional[str] = None
    current_episode: int = 0
    total_episodes: int = 0
    success_rate: float = 0.0


# ============== Evaluation Schemas ==============

class EvaluateRequest(BaseModel):
    """Request to run evaluation"""
    checkpoint: str  # Path or name of checkpoint
    algorithm: str = "ddq"  # "dqn" or "ddq"
    num_episodes: int = 20
    use_llm: bool = True


class EvaluationResult(BaseModel):
    """Results from evaluation run"""
    success_rate: float
    avg_reward: float
    avg_length: float
    num_episodes: int
    sample_conversations: List[List[ConversationMessage]]
