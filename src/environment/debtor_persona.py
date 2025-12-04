"""
Debtor Persona Definitions and Behaviors
Defines the characteristics and response patterns for different debtor types
"""

import random
from typing import Dict, Tuple
from src.config import PersonaConfig


class DebtorPersona:
    """Represents a debtor with specific personality traits and state"""

    def __init__(self, persona_type: str):
        """
        Initialize a debtor with given persona type

        Args:
            persona_type: One of ["angry", "cooperative", "sad", "avoidant"]
        """
        if persona_type not in PersonaConfig.PERSONA_TRAITS:
            raise ValueError(f"Invalid persona type: {persona_type}")

        self.persona_type = persona_type
        self.traits = PersonaConfig.PERSONA_TRAITS[persona_type]

        # Initialize attributes based on persona
        sent_min, sent_max = self.traits["initial_sentiment_range"]
        coop_min, coop_max = self.traits["initial_cooperation_range"]

        # Core attributes
        self.sentiment = random.uniform(sent_min, sent_max)
        self.cooperation = random.uniform(coop_min, coop_max)
        self.engagement = 0.5  # Start neutral
        self.financial_stress = random.uniform(0.6, 0.9)  # Most debtors are stressed

        # State flags
        self.has_committed_to_pay = False
        self.has_shared_situation = False
        self.agent_mentioned_payment_plan = False
        self.agent_mentioned_consequences = False
        self.feels_understood = False

        # Conversation tracking
        self.turn_count = 0
        self.sentiment_history = [self.sentiment]
        self.cooperation_history = [self.cooperation]

    def update_from_interaction(self, agent_action: str, action_effectiveness: float):
        """
        Update debtor state based on agent's action

        Args:
            agent_action: The strategy agent used
            action_effectiveness: How well the action worked (-1 to 1)
        """
        self.turn_count += 1

        # Base changes from action effectiveness
        sentiment_change = action_effectiveness * 0.2
        cooperation_change = action_effectiveness * 0.15

        # Persona-specific modifiers
        if agent_action in self.traits["responds_to"]:
            sentiment_change *= 1.5
            cooperation_change *= 1.5
            self.engagement = min(1.0, self.engagement + 0.1)
        elif agent_action in self.traits["triggers"]:
            sentiment_change *= -1.5
            cooperation_change *= -1.5
            self.engagement = max(0.0, self.engagement - 0.2)

        # Apply changes with bounds
        self.sentiment = max(-1.0, min(1.0, self.sentiment + sentiment_change))
        self.cooperation = max(0.0, min(1.0, self.cooperation + cooperation_change))

        # Deterministic flag updates (decoupled from LLM)
        # shared_situation: triggered by ask_about_situation when debtor is not too hostile
        if agent_action == "ask_about_situation" and self.sentiment > -0.5:
            self.has_shared_situation = True
        
        # feels_understood: triggered by empathetic_listening when effective
        if agent_action == "empathetic_listening" and action_effectiveness > 0.3:
            self.feels_understood = True
        
        # Track if agent mentioned payment plan or consequences
        if agent_action == "offer_payment_plan":
            self.agent_mentioned_payment_plan = True
        if agent_action in ["firm_reminder", "hard_close"]:
            self.agent_mentioned_consequences = True

        # Update histories
        self.sentiment_history.append(self.sentiment)
        self.cooperation_history.append(self.cooperation)

    def get_sentiment_trend(self) -> float:
        """Get sentiment change over last 2 turns"""
        if len(self.sentiment_history) < 2:
            return 0.0
        return self.sentiment_history[-1] - self.sentiment_history[-3] if len(self.sentiment_history) >= 3 else 0.0

    def get_cooperation_trend(self) -> float:
        """Get cooperation change over last 2 turns"""
        if len(self.cooperation_history) < 2:
            return 0.0
        return self.cooperation_history[-1] - self.cooperation_history[-3] if len(self.cooperation_history) >= 3 else 0.0

    def should_quit(self) -> bool:
        """Check if debtor would quit the conversation"""
        from config import EnvironmentConfig

        # Too hostile
        if self.sentiment < EnvironmentConfig.SENTIMENT_THRESHOLD_QUIT:
            return True

        # Too disengaged
        if self.engagement < EnvironmentConfig.ENGAGEMENT_THRESHOLD_QUIT:
            return True

        # Avoidant persona has lower patience - deterministic threshold
        if self.persona_type == "avoidant" and self.turn_count > 10:
            return True  # Avoidant always quits after 10 turns

        return False

    def check_commitment(self) -> bool:
        """
        Check if debtor is ready to commit to payment
        Based on cooperation, sentiment, and persona
        """
        if self.has_committed_to_pay:
            return True

        # Need both high cooperation and positive sentiment
        cooperation_threshold = 0.7
        sentiment_threshold = 0.3

        # Different personas have different thresholds
        if self.persona_type == "cooperative":
            cooperation_threshold = 0.6
            sentiment_threshold = 0.2
        elif self.persona_type == "angry":
            cooperation_threshold = 0.8
            sentiment_threshold = 0.4
        elif self.persona_type == "sad":
            cooperation_threshold = 0.7
            sentiment_threshold = 0.3
        elif self.persona_type == "avoidant":
            cooperation_threshold = 0.7
            sentiment_threshold = 0.4

        # Check thresholds
        if self.cooperation >= cooperation_threshold and self.sentiment >= sentiment_threshold:
            # Additional requirement: must feel understood or have payment plan offered
            if self.feels_understood or self.agent_mentioned_payment_plan:
                # Deterministic commitment - above thresholds = commit
                # This makes the reward signal consistent for same state-action pairs
                self.has_committed_to_pay = True
                return True

        return False

    def to_dict(self) -> Dict:
        """Convert debtor state to dictionary for state representation"""
        return {
            "persona_type": self.persona_type,
            "sentiment": self.sentiment,
            "cooperation": self.cooperation,
            "engagement": self.engagement,
            "financial_stress": self.financial_stress,
            "has_committed": self.has_committed_to_pay,
            "has_shared_situation": self.has_shared_situation,
            "feels_understood": self.feels_understood,
            "turn_count": self.turn_count,
            "sentiment_trend": self.get_sentiment_trend(),
            "cooperation_trend": self.get_cooperation_trend(),
        }

    def get_prompt_context(self) -> str:
        """Generate context for LLM prompt"""
        context = f"""DEBTOR PROFILE:
- Persona: {self.persona_type.upper()}
- Personality: {self.traits['personality']}
- Background: {self.traits['background']}
- Communication style: {self.traits['communication_style']}

CURRENT STATE:
- Sentiment: {self.sentiment:.2f} (-1=very hostile, 0=neutral, +1=very friendly)
- Cooperation level: {self.cooperation:.2f} (0=completely uncooperative, 1=very cooperative)
- Engagement: {self.engagement:.2f} (0=wants to end call, 1=actively engaged)
- Financial stress: {self.financial_stress:.2f} (0=comfortable, 1=extreme stress)
- Turn count: {self.turn_count}

INTERACTION HISTORY:
- Feels understood: {'Yes' if self.feels_understood else 'No'}
- Has shared situation: {'Yes' if self.has_shared_situation else 'No'}
- Agent mentioned payment plan: {'Yes' if self.agent_mentioned_payment_plan else 'No'}
- Agent mentioned consequences: {'Yes' if self.agent_mentioned_consequences else 'No'}
"""
        return context

    def __repr__(self) -> str:
        return (f"DebtorPersona(type={self.persona_type}, "
                f"sentiment={self.sentiment:.2f}, "
                f"cooperation={self.cooperation:.2f}, "
                f"committed={self.has_committed_to_pay})")


def create_random_debtor() -> DebtorPersona:
    """Create a debtor with random persona"""
    from config import EnvironmentConfig
    persona_type = random.choice(EnvironmentConfig.PERSONAS)
    return DebtorPersona(persona_type)


# ============================================================================
# Action Effectiveness Mapping
# ============================================================================

# How effective each action is for each persona type
# Scale: -1 (very bad) to +1 (very good)
ACTION_EFFECTIVENESS = {
    "angry": {
        "empathetic_listening": 0.8,      # Very effective
        "ask_about_situation": 0.6,       # Good
        "firm_reminder": -0.7,            # Bad - triggers defensiveness
        "offer_payment_plan": 0.4,        # Neutral to good
        "propose_settlement": 0.3,        # Slightly good
        "hard_close": -0.9,               # Very bad - escalates anger
    },
    "cooperative": {
        "empathetic_listening": 0.5,      # Good
        "ask_about_situation": 0.6,       # Good
        "firm_reminder": 0.3,             # Acceptable
        "offer_payment_plan": 0.9,        # Excellent
        "propose_settlement": 0.8,        # Very good
        "hard_close": -0.3,               # Slightly bad - feels pushy
    },
    "sad": {
        "empathetic_listening": 0.9,      # Excellent - needs empathy
        "ask_about_situation": 0.7,       # Very good - wants to share
        "firm_reminder": -0.5,            # Bad - feels judged
        "offer_payment_plan": 0.6,        # Good
        "propose_settlement": 0.5,        # Good
        "hard_close": -0.8,               # Very bad - overwhelming
    },
    "avoidant": {
        "empathetic_listening": 0.3,      # Slightly good
        "ask_about_situation": -0.2,      # Slightly bad - wants quick resolution
        "firm_reminder": 0.4,             # Neutral - might motivate
        "offer_payment_plan": 0.7,        # Good - quick solution
        "propose_settlement": 0.8,        # Very good - ends quickly
        "hard_close": 0.5,                # Neutral to good - creates urgency
    }
}


def get_action_effectiveness(persona_type: str, action_name: str, deterministic: bool = True) -> float:
    """
    Get how effective an action is for a given persona

    Args:
        persona_type: Debtor persona type
        action_name: Action taken by agent
        deterministic: If True, no randomness (default for training stability)

    Returns:
        Effectiveness score (-1 to 1)
    """
    if persona_type not in ACTION_EFFECTIVENESS:
        return 0.0
    if action_name not in ACTION_EFFECTIVENESS[persona_type]:
        return 0.0

    base_effectiveness = ACTION_EFFECTIVENESS[persona_type][action_name]
    
    # Optional noise for evaluation/demo (disabled by default for stable training)
    if not deterministic:
        noise = random.uniform(-0.2, 0.2)
        return max(-1.0, min(1.0, base_effectiveness + noise))
    
    return base_effectiveness
