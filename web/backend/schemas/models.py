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


# ============== Battle History Schemas ==============

class BattleTurnResponse(BaseModel):
    """A single turn in an adversarial battle"""
    id: int
    turn_num: int
    collector_strategy: Optional[str] = None
    collector_utterance: Optional[str] = None
    adversary_strategy: Optional[str] = None
    adversary_response: Optional[str] = None
    collector_reward: float = 0.0
    adversary_reward: float = 0.0


class EpisodeSummary(BaseModel):
    """Summary of a battle episode"""
    id: int
    episode_num: int
    outcome: str
    collector_total_reward: float
    adversary_total_reward: float
    num_turns: int
    completed_at: Optional[str] = None


class EpisodeDetailResponse(BaseModel):
    """Detailed episode with all turns"""
    id: int
    episode_num: int
    outcome: str
    collector_total_reward: float
    adversary_total_reward: float
    num_turns: int
    completed_at: Optional[str] = None
    turns: List[BattleTurnResponse] = []


class GenerationSummary(BaseModel):
    """Summary of a generation"""
    id: int
    generation_num: int
    collector_win_rate: float
    adversary_win_rate: float
    avg_collector_reward: float
    avg_adversary_reward: float
    episode_count: int = 0
    completed_at: Optional[str] = None


class GenerationDetailResponse(BaseModel):
    """Detailed generation with episodes"""
    id: int
    generation_num: int
    collector_win_rate: float
    adversary_win_rate: float
    avg_collector_reward: float
    avg_adversary_reward: float
    collector_strategy_dist: Optional[Dict[str, Any]] = None
    adversary_strategy_dist: Optional[Dict[str, Any]] = None
    completed_at: Optional[str] = None
    episodes: List[EpisodeSummary] = []


class SessionSummary(BaseModel):
    """Summary of a training session"""
    id: int
    started_at: str
    ended_at: Optional[str] = None
    total_generations: int
    episodes_per_gen: int
    use_llm: bool
    zero_sum: bool
    final_collector_win_rate: Optional[float] = None
    final_adversary_win_rate: Optional[float] = None
    status: str
    generation_count: int = 0


class SessionDetailResponse(BaseModel):
    """Detailed session with generations"""
    id: int
    started_at: str
    ended_at: Optional[str] = None
    total_generations: int
    episodes_per_gen: int
    use_llm: bool
    zero_sum: bool
    final_collector_win_rate: Optional[float] = None
    final_adversary_win_rate: Optional[float] = None
    status: str
    generations: List[GenerationSummary] = []


class BattleHistoryResponse(BaseModel):
    """Paginated list of training sessions"""
    sessions: List[SessionSummary]
    total: int
    limit: int
    offset: int

