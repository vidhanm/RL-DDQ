"""
Configuration file for DDQ Debt Collection Agent
Contains all hyperparameters and settings for training and evaluation
"""

import torch

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

class EnvironmentConfig:
    """Debt Collection Environment Configuration"""

    # Episode settings
    MAX_TURNS = 15              # Maximum turns per conversation
    MIN_TURNS = 3               # Minimum turns before allowing termination

    # Debtor personas
    PERSONAS = ["angry", "cooperative", "sad", "avoidant"]
    NUM_PERSONAS = len(PERSONAS)

    # State space
    STATE_DIM = 20              # Dimensionality of state vector

    # Action space
    NUM_ACTIONS = 6
    ACTIONS = {
        0: "empathetic_listening",
        1: "ask_about_situation",
        2: "firm_reminder",
        3: "offer_payment_plan",
        4: "propose_settlement",
        5: "hard_close"
    }

    # Reward weights
    REWARD_COMMITMENT = 10.0        # Payment commitment bonus
    REWARD_SENTIMENT_WEIGHT = 3.0   # Weight for sentiment change
    REWARD_COOPERATION_WEIGHT = 2.0 # Weight for cooperation change
    REWARD_ENGAGEMENT_BONUS = 0.5   # Bonus for high engagement
    REWARD_HOSTILITY_PENALTY = 3.0  # Penalty for extreme hostility
    REWARD_TURN_PENALTY = 0.1       # Small penalty per turn
    REWARD_FAILURE_PENALTY = 5.0    # Penalty for ending without commitment

    # Termination conditions
    SENTIMENT_THRESHOLD_QUIT = -0.9  # Debtor quits if sentiment drops below
    ENGAGEMENT_THRESHOLD_QUIT = 0.2  # Debtor quits if engagement drops below


# ============================================================================
# RL HYPERPARAMETERS
# ============================================================================

class RLConfig:
    """Reinforcement Learning Hyperparameters"""

    # DQN Network
    HIDDEN_DIM = 128            # Hidden layer size for DQN
    LEARNING_RATE = 0.0001      # DQN learning rate

    # Training
    GAMMA = 0.95                # Discount factor
    BATCH_SIZE = 32             # Minibatch size for training
    REPLAY_BUFFER_SIZE = 10000  # Maximum replay buffer size
    MIN_BUFFER_SIZE = 500       # Minimum experiences before training

    # Exploration
    EPSILON_START = 1.0         # Initial exploration rate
    EPSILON_END = 0.05          # Final exploration rate
    EPSILON_DECAY = 0.995       # Decay rate per episode

    # Target Network
    TARGET_UPDATE_FREQ = 10     # Update target network every N episodes

    # Training schedule
    TRAIN_FREQ = 1              # Train every N episodes
    NUM_TRAIN_STEPS = 1         # Number of training steps per training session


# ============================================================================
# DDQ HYPERPARAMETERS
# ============================================================================

class DDQConfig:
    """DDQ-specific (World Model + Imagination) Hyperparameters"""

    # World Model Architecture
    WORLD_MODEL_HIDDEN_DIM = 128     # Hidden layer size
    WORLD_MODEL_LEARNING_RATE = 0.001
    WORLD_MODEL_EPOCHS = 5           # Training epochs per session

    # Imagination
    K = 5                            # Imagination factor (rollouts per real experience)
    IMAGINATION_HORIZON = 1          # Steps to imagine forward (1=single-step)

    # Experience mixing
    REAL_RATIO = 0.75                # 75% real, 25% imagined in training batch

    # When to start using world model
    MIN_WORLD_MODEL_BUFFER = 500     # Minimum real experiences before imagination

    # Ensemble (optional - for uncertainty estimation)
    USE_ENSEMBLE = False             # Use ensemble of world models
    NUM_ENSEMBLE_MODELS = 5          # Number of models in ensemble
    ENSEMBLE_DISAGREEMENT_THRESHOLD = 0.1  # Skip if variance > threshold


# ============================================================================
# LLM SETTINGS
# ============================================================================

class LLMConfig:
    """OpenAI LLM Configuration"""

    # Model selection
    MODEL_NAME = "gpt-4"            # "gpt-4" or "gpt-3.5-turbo"
    MODEL_NAME_DEV = "gpt-3.5-turbo"  # Cheaper model for development
    USE_DEV_MODEL = False           # Set to True for cheaper development

    # Generation parameters
    TEMPERATURE_AGENT = 0.7         # Agent utterance generation
    TEMPERATURE_DEBTOR = 0.8        # Debtor response generation
    MAX_TOKENS = 300                # Maximum tokens per generation

    # API settings
    API_TIMEOUT = 30                # Seconds
    MAX_RETRIES = 3                 # Retry failed API calls


# ============================================================================
# TRAINING SETTINGS
# ============================================================================

class TrainingConfig:
    """Training Loop Configuration"""

    # Episodes
    NUM_EPISODES_DQN = 200          # Episodes for DQN baseline
    NUM_EPISODES_DDQ = 200          # Episodes for DDQ

    # Evaluation
    EVAL_FREQ = 10                  # Evaluate every N episodes
    EVAL_EPISODES = 10              # Number of episodes for evaluation

    # Checkpointing
    SAVE_FREQ = 50                  # Save checkpoint every N episodes
    CHECKPOINT_DIR = "checkpoints"

    # Logging
    LOG_FREQ = 1                    # Log every N episodes
    LOG_DIR = "logs"

    # Comparison mode
    COMPARE_MODE = False            # Train both DQN and DDQ for comparison


# ============================================================================
# DEVICE SETTINGS
# ============================================================================

class DeviceConfig:
    """Hardware Configuration"""

    # Auto-detect GPU
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Seed for reproducibility
    RANDOM_SEED = 42


# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

class VisualizationConfig:
    """Settings for plots and visualizations"""

    # Output
    PLOT_DIR = "plots"
    SAVE_PLOTS = True
    SHOW_PLOTS = False              # Set True for interactive display

    # Plot style
    FIGURE_SIZE = (12, 6)
    DPI = 100
    STYLE = "seaborn-v0_8-darkgrid"


# ============================================================================
# PERSONA DEFINITIONS
# ============================================================================

class PersonaConfig:
    """Detailed persona characteristics for debtor simulation"""

    PERSONA_TRAITS = {
        "angry": {
            "initial_sentiment_range": (-0.6, -0.4),
            "initial_cooperation_range": (0.1, 0.3),
            "triggers": ["pressure", "consequences", "judgment"],
            "responds_to": ["empathy", "understanding", "patience"],
            "personality": "Defensive, easily frustrated, feels attacked",
            "background": "Lost job recently, feels overwhelmed by debt",
            "communication_style": "Short, sharp responses, may raise voice"
        },
        "cooperative": {
            "initial_sentiment_range": (0.2, 0.4),
            "initial_cooperation_range": (0.6, 0.8),
            "triggers": ["unrealistic demands", "disrespect", "complications"],
            "responds_to": ["payment plans", "clear options", "respect"],
            "personality": "Willing to work together, responsible",
            "background": "Temporary financial setback, wants to resolve",
            "communication_style": "Professional, asks clarifying questions"
        },
        "sad": {
            "initial_sentiment_range": (-0.2, 0.0),
            "initial_cooperation_range": (0.4, 0.6),
            "triggers": ["pressure", "lack of empathy", "judgment"],
            "responds_to": ["empathy", "flexible options", "understanding"],
            "personality": "Overwhelmed, emotional, struggling",
            "background": "Medical bills or family crisis, emotionally drained",
            "communication_style": "Apologetic, may become emotional"
        },
        "avoidant": {
            "initial_sentiment_range": (-0.1, 0.1),
            "initial_cooperation_range": (0.2, 0.4),
            "triggers": ["long conversations", "pressure", "complexity"],
            "responds_to": ["quick solutions", "simplicity", "urgency"],
            "personality": "Wants to end conversation quickly, evasive",
            "background": "Procrastinator, hopes problem goes away",
            "communication_style": "Brief, tries to deflect or delay"
        }
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_active_model():
    """Get the active LLM model name"""
    if LLMConfig.USE_DEV_MODEL:
        return LLMConfig.MODEL_NAME_DEV
    return LLMConfig.MODEL_NAME


def print_config():
    """Print all configuration settings"""
    print("=" * 70)
    print("DDQ DEBT COLLECTION AGENT - CONFIGURATION")
    print("=" * 70)

    print("\n[ENVIRONMENT]")
    print(f"  Max turns: {EnvironmentConfig.MAX_TURNS}")
    print(f"  Personas: {EnvironmentConfig.PERSONAS}")
    print(f"  State dim: {EnvironmentConfig.STATE_DIM}")
    print(f"  Actions: {EnvironmentConfig.NUM_ACTIONS}")

    print("\n[RL HYPERPARAMETERS]")
    print(f"  Learning rate: {RLConfig.LEARNING_RATE}")
    print(f"  Gamma: {RLConfig.GAMMA}")
    print(f"  Epsilon: {RLConfig.EPSILON_START} -> {RLConfig.EPSILON_END} (decay: {RLConfig.EPSILON_DECAY})")
    print(f"  Batch size: {RLConfig.BATCH_SIZE}")
    print(f"  Buffer size: {RLConfig.REPLAY_BUFFER_SIZE}")

    print("\n[DDQ SETTINGS]")
    print(f"  Imagination factor K: {DDQConfig.K}")
    print(f"  Real/Imagined ratio: {DDQConfig.REAL_RATIO:.0%} / {1-DDQConfig.REAL_RATIO:.0%}")
    print(f"  World model LR: {DDQConfig.WORLD_MODEL_LEARNING_RATE}")
    print(f"  Imagination horizon: {DDQConfig.IMAGINATION_HORIZON}")

    print("\n[LLM]")
    print(f"  Model: {get_active_model()}")
    print(f"  Temperature (agent/debtor): {LLMConfig.TEMPERATURE_AGENT} / {LLMConfig.TEMPERATURE_DEBTOR}")
    print(f"  Max tokens: {LLMConfig.MAX_TOKENS}")

    print("\n[TRAINING]")
    print(f"  Episodes (DQN/DDQ): {TrainingConfig.NUM_EPISODES_DQN} / {TrainingConfig.NUM_EPISODES_DDQ}")
    print(f"  Eval frequency: every {TrainingConfig.EVAL_FREQ} episodes")
    print(f"  Save frequency: every {TrainingConfig.SAVE_FREQ} episodes")

    print("\n[DEVICE]")
    print(f"  Device: {DeviceConfig.DEVICE}")
    print(f"  Random seed: {DeviceConfig.RANDOM_SEED}")

    print("=" * 70)


# ============================================================================
# EXPORT ALL CONFIGS
# ============================================================================

__all__ = [
    'EnvironmentConfig',
    'RLConfig',
    'DDQConfig',
    'LLMConfig',
    'TrainingConfig',
    'DeviceConfig',
    'VisualizationConfig',
    'PersonaConfig',
    'get_active_model',
    'print_config'
]
