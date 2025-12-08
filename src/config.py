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

    # Debtor personas (legacy - used by old debtor_env.py)
    PERSONAS = ["angry", "cooperative", "sad", "avoidant"]
    NUM_PERSONAS = len(PERSONAS)

    # State space
    STATE_DIM = 18              # Legacy state dimension
    NLU_STATE_DIM = 19          # NLU-based state dimension (for nlu_env.py)
    
    # NLU Environment settings
    USE_NLU_ENV = True          # Use NLU-based environment (recommended)
    USE_DOMAIN_RANDOMIZATION = True  # Use random debtor profiles

    # Action space - Expanded with expert-recommended strategies
    NUM_ACTIONS = 9  # Expanded from 6 to 9
    ACTIONS = {
        # Original 6 actions
        0: "empathetic_listening",
        1: "ask_about_situation",
        2: "firm_reminder",
        3: "offer_payment_plan",
        4: "propose_settlement",
        5: "hard_close",
        # New nuanced actions (based on expert research)
        6: "acknowledge_and_redirect",  # When debtor vents or goes off-topic
        7: "validate_then_offer",        # Acknowledge emotion, then solution
        8: "gentle_urgency"              # Create urgency without threats
    }

    # Reward weights
    REWARD_COMMITMENT = 10.0        # Payment commitment bonus
    REWARD_SENTIMENT_WEIGHT = 3.0   # Weight for sentiment change
    REWARD_COOPERATION_WEIGHT = 2.0 # Weight for cooperation change
    REWARD_ENGAGEMENT_BONUS = 0.5   # Bonus for high engagement
    REWARD_HOSTILITY_PENALTY = 3.0  # Penalty for extreme hostility
    REWARD_TURN_PENALTY = 0.1       # Small penalty per turn
    REWARD_FAILURE_PENALTY = 5.0    # Penalty for ending without commitment
    
    # =========================================================================
    # EXPERT KNOWLEDGE REWARDS
    # Based on debt collection industry best practices research
    # =========================================================================
    
    # Positive rewards - encourage expert behaviors
    EXPERT_REWARDS = {
        # Opening with empathy builds trust (expert best practice)
        'empathy_before_pressure': 2.0,
        
        # Acknowledging situation before asking for payment
        'acknowledge_situation': 2.0,
        
        # Offering flexible payment options increases commitment
        'offer_flexible_options': 2.0,
        
        # Successfully de-escalating hostile debtor (critical skill)
        'de_escalate_hostility': 3.0,
        
        # Asking open questions to understand situation
        'asked_open_question': 1.0,
        
        # Keeping debtor engaged in conversation
        'maintained_engagement': 1.0,
        
        # Recovery - improved situation after it got worse
        'recovered_from_negative': 2.5,
    }
    
    # Negative rewards - discourage beginner mistakes
    EXPERT_PENALTIES = {
        # Using hard close too early (before building rapport)
        'premature_hard_close': -3.0,
        
        # Ignoring debtor's stated circumstances
        'ignored_circumstance': -2.0,
        
        # Repeating same failed strategy
        'repeated_failed_strategy': -2.0,
        
        # Pressure tactics on already hostile debtor
        'pressure_on_hostile': -3.0,
        
        # Using firm/hard approach on sad/overwhelmed debtor
        'pressure_on_sad': -2.0,
        
        # Not offering alternatives when debtor shows willingness
        'missed_willing_opportunity': -1.5,
    }
    
    # Actions considered "empathetic" (should come before pressure)
    EMPATHETIC_ACTIONS = ['empathetic_listening', 'ask_about_situation', 'acknowledge_and_redirect', 'validate_then_offer']
    
    # Actions considered "pressure" (should come after empathy)
    PRESSURE_ACTIONS = ['firm_reminder', 'hard_close']
    
    # Actions considered "solution-oriented"
    SOLUTION_ACTIONS = ['offer_payment_plan', 'propose_settlement', 'validate_then_offer']
    
    # Actions for creating urgency (softer than pressure)
    URGENCY_ACTIONS = ['gentle_urgency']

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
# ADVERSARIAL SELF-PLAY SETTINGS
# ============================================================================

class SelfPlayConfig:
    """Adversarial Self-Play Training Configuration"""
    
    # -------------------------------------------------------------------------
    # ADVERSARIAL DEBTOR ACTIONS
    # -------------------------------------------------------------------------
    NUM_ADVERSARY_ACTIONS = 7
    ADVERSARY_ACTIONS = {
        0: "aggressive",        # Hostile, threatening to end call
        1: "evasive",           # Deflect, change subject, vague answers
        2: "emotional",         # Express distress, play victim
        3: "negotiate_hard",    # Demand unrealistic terms
        4: "partial_cooperate", # Give minimal ground
        5: "stall",             # Ask for delays, need to think
        6: "dispute"            # Challenge validity of debt
    }
    
    # -------------------------------------------------------------------------
    # TRAINING SETTINGS
    # -------------------------------------------------------------------------
    GENERATIONS = 50                    # Number of self-play generations
    EPISODES_PER_GENERATION = 100       # Training episodes per agent per generation
    OPPONENT_POOL_SIZE = 10             # Keep last N versions as opponents
    OPPONENT_SAMPLE_STRATEGY = "uniform"  # "uniform", "prioritized", or "latest"
    
    # Evaluation
    FROZEN_EVAL_FREQ = 5                # Evaluate against fixed debtors every N gens
    EVAL_EPISODES = 20                  # Episodes per evaluation
    
    # -------------------------------------------------------------------------
    # REWARD STRUCTURE
    # -------------------------------------------------------------------------
    # Collector rewards (standard)
    COLLECTOR_REWARD_SCALE = 1.0
    
    # Adversary rewards (mostly inverse of collector)
    ADVERSARY_REWARD_SCALE = 1.0
    ZERO_SUM_COEFFICIENT = 0.8          # 1.0 = pure zero-sum, 0.0 = independent
    
    # Specific adversary bonuses
    STALL_BONUS_PER_TURN = 0.1          # Bonus for each turn without commitment
    RESIST_COMMITMENT_BONUS = 3.0       # Bonus if conversation ends without payment
    MAKE_COLLECTOR_FAIL_BONUS = 2.0     # Bonus if collector gives up or fails
    
    # Collector bonuses against adversary
    DIFFICULT_CONVERSION_BONUS = 2.0    # Extra reward for converting adversarial debtor
    
    # -------------------------------------------------------------------------
    # REWARD EVENTS
    # -------------------------------------------------------------------------
    REWARD_EVENTS = {
        # Event: (collector_reward, adversary_reward)
        "payment_commitment": (10.0, -8.0),
        "conversation_end_no_commit": (-3.0, 3.0),
        "debtor_hangs_up": (-5.0, 2.0),
        "escalation_triggered": (-2.0, 1.0),
        "per_turn_no_commit": (-0.05, 0.1),
    }
    
    # -------------------------------------------------------------------------
    # CONVERGENCE CRITERIA
    # -------------------------------------------------------------------------
    WIN_RATE_THRESHOLD = 0.65           # Stop if collector achieves this win rate
    MIN_GENERATIONS = 20                # Minimum generations before stopping
    STRATEGY_ENTROPY_MIN = 0.3          # Minimum action entropy (avoid mode collapse)


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
    MAX_TOKENS = 800                # Maximum tokens per generation (increased for Hindi/Hinglish + Qwen thinking)

    # API settings
    API_TIMEOUT = 30                # Seconds
    MAX_RETRIES = 5                 # Retry failed API calls (increased from 3)


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
    'SelfPlayConfig',
    'LLMConfig',
    'TrainingConfig',
    'DeviceConfig',
    'VisualizationConfig',
    'PersonaConfig',
    'get_active_model',
    'print_config'
]
