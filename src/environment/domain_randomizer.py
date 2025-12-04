"""
Domain Randomization for Diverse Debtor Simulation

Replaces 4 discrete personas with continuous parameter space.
Creates millions of unique debtor profiles for robust generalization.

Key Principle: Agent never sees these parameters - they only affect
LLM prompt for simulation. Agent sees NLU-extracted behavioral features.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class LifeEvent(Enum):
    """Major life events that affect debtor situation"""
    NONE = "none"
    JOB_LOSS = "job_loss"
    MEDICAL = "medical_emergency"
    DIVORCE = "divorce"
    FAMILY_EMERGENCY = "family_emergency"
    BUSINESS_FAILURE = "business_failure"
    REDUCED_HOURS = "reduced_hours"


class CallHistory(Enum):
    """Previous interaction history"""
    FIRST_CALL = "first_call"
    REPEAT_CALLER = "repeat_caller"  
    ESCALATED = "escalated"
    DISPUTED = "disputed_previously"


@dataclass
class DebtorProfile:
    """
    Complete debtor profile for LLM simulation.
    
    All parameters are HIDDEN from the agent.
    Agent only sees NLU-extracted behavioral features.
    """
    
    # === Personality Traits (Big Five inspired) ===
    agreeableness: float = 0.5        # 0=disagreeable, 1=very agreeable
    emotional_stability: float = 0.5  # 0=volatile, 1=calm
    assertiveness: float = 0.5        # 0=passive, 1=assertive
    openness: float = 0.5             # 0=closed, 1=open to options
    
    # === Financial Situation ===
    debt_amount: float = 5000.0       # $500 - $50,000
    days_overdue: int = 60            # 30 - 365 days
    financial_stress: float = 0.5     # 0=low stress, 1=severe stress
    has_income: bool = True           # Currently has income
    can_afford_payment: bool = True   # Can actually make some payment
    
    # === Situational Factors ===
    life_event: LifeEvent = LifeEvent.NONE
    call_history: CallHistory = CallHistory.FIRST_CALL
    time_of_day_mood: float = 0.0     # -0.3 to +0.3 mood modifier
    previous_experience: str = "neutral"  # good, bad, neutral
    
    # === Communication Style ===
    verbosity: float = 0.5            # 0=terse, 1=verbose
    directness: float = 0.5           # 0=evasive, 1=direct
    formality: float = 0.5            # 0=casual, 1=formal
    
    # === Starting State ===
    initial_sentiment: float = 0.0    # Will be calculated from traits
    initial_cooperation: float = 0.5  # Will be calculated from traits
    
    def __post_init__(self):
        """Calculate derived initial states from traits"""
        # Initial sentiment based on traits and situation
        self.initial_sentiment = self._calculate_initial_sentiment()
        self.initial_cooperation = self._calculate_initial_cooperation()
    
    def _calculate_initial_sentiment(self) -> float:
        """Calculate initial sentiment from traits"""
        base = 0.0
        
        # Personality effects
        base += (self.emotional_stability - 0.5) * 0.4  # ±0.2
        base += (self.agreeableness - 0.5) * 0.3        # ±0.15
        
        # Situation effects
        base -= self.financial_stress * 0.3              # -0.3 at max stress
        
        # Life event effects
        if self.life_event != LifeEvent.NONE:
            base -= 0.15  # Recent life event = negative mood
        
        # Call history effects
        if self.call_history == CallHistory.ESCALATED:
            base -= 0.2
        elif self.call_history == CallHistory.REPEAT_CALLER:
            base -= 0.1
        
        # Time of day mood
        base += self.time_of_day_mood
        
        return max(-1.0, min(1.0, base))
    
    def _calculate_initial_cooperation(self) -> float:
        """Calculate initial cooperation willingness"""
        base = 0.5
        
        # Personality effects
        base += (self.agreeableness - 0.5) * 0.4        # ±0.2
        base += (self.openness - 0.5) * 0.2             # ±0.1
        base -= (self.assertiveness - 0.5) * 0.2        # assertive = less immediately cooperative
        
        # Situation effects
        if self.can_afford_payment:
            base += 0.1
        else:
            base -= 0.15
        
        if not self.has_income:
            base -= 0.2
        
        # Previous experience
        if self.previous_experience == "good":
            base += 0.1
        elif self.previous_experience == "bad":
            base -= 0.15
        
        return max(0.0, min(1.0, base))


class DomainRandomizer:
    """
    Sample random debtor profiles for training diversity.
    
    Uses domain randomization to ensure agent generalizes
    to any debtor type, not just 4 archetypes.
    """
    
    def __init__(
        self,
        debt_range: tuple = (500, 50000),
        overdue_range: tuple = (30, 365),
        seed: Optional[int] = None
    ):
        """
        Initialize domain randomizer.
        
        Args:
            debt_range: (min, max) debt amount
            overdue_range: (min, max) days overdue
            seed: Random seed for reproducibility
        """
        self.debt_range = debt_range
        self.overdue_range = overdue_range
        
        if seed is not None:
            random.seed(seed)
    
    def sample(self) -> DebtorProfile:
        """
        Sample a random debtor profile.
        
        Returns:
            DebtorProfile with randomized parameters
        """
        # Sample personality traits (uniform distribution)
        agreeableness = random.random()
        emotional_stability = random.random()
        assertiveness = random.random()
        openness = random.random()
        
        # Sample financial situation
        # Log-uniform for debt (more small debts, fewer large)
        log_min = 2.7  # log(500)
        log_max = 4.7  # log(50000)
        debt_amount = 10 ** random.uniform(log_min, log_max)
        
        days_overdue = random.randint(*self.overdue_range)
        financial_stress = random.random()
        has_income = random.random() > 0.2  # 80% have some income
        can_afford = random.random() > 0.3  # 70% can afford something
        
        # Sample situational factors
        life_event = random.choice(list(LifeEvent))
        call_history = random.choices(
            list(CallHistory),
            weights=[0.5, 0.3, 0.15, 0.05]  # First call most common
        )[0]
        time_mood = random.uniform(-0.3, 0.3)
        prev_exp = random.choices(
            ["good", "bad", "neutral"],
            weights=[0.2, 0.3, 0.5]
        )[0]
        
        # Sample communication style
        verbosity = random.random()
        directness = random.random()
        formality = random.random()
        
        return DebtorProfile(
            agreeableness=agreeableness,
            emotional_stability=emotional_stability,
            assertiveness=assertiveness,
            openness=openness,
            debt_amount=debt_amount,
            days_overdue=days_overdue,
            financial_stress=financial_stress,
            has_income=has_income,
            can_afford_payment=can_afford,
            life_event=life_event,
            call_history=call_history,
            time_of_day_mood=time_mood,
            previous_experience=prev_exp,
            verbosity=verbosity,
            directness=directness,
            formality=formality
        )
    
    def sample_easy(self) -> DebtorProfile:
        """Sample an easier debtor (for curriculum learning)"""
        profile = self.sample()
        # Bias toward cooperative traits
        profile.agreeableness = 0.5 + random.random() * 0.5  # 0.5-1.0
        profile.emotional_stability = 0.5 + random.random() * 0.5
        profile.openness = 0.5 + random.random() * 0.5
        profile.financial_stress = random.random() * 0.5  # 0-0.5
        profile.has_income = True
        profile.can_afford_payment = True
        profile.life_event = LifeEvent.NONE
        profile.call_history = CallHistory.FIRST_CALL
        profile.__post_init__()  # Recalculate initial states
        return profile
    
    def sample_medium(self) -> DebtorProfile:
        """Sample a medium difficulty debtor"""
        profile = self.sample()
        # Mix of traits
        profile.financial_stress = 0.3 + random.random() * 0.4  # 0.3-0.7
        profile.__post_init__()
        return profile
    
    def sample_hard(self) -> DebtorProfile:
        """Sample a harder debtor (for curriculum learning)"""
        profile = self.sample()
        # Bias toward difficult traits
        profile.agreeableness = random.random() * 0.5  # 0-0.5
        profile.emotional_stability = random.random() * 0.5
        profile.assertiveness = 0.5 + random.random() * 0.5  # 0.5-1.0
        profile.financial_stress = 0.5 + random.random() * 0.5  # 0.5-1.0
        profile.has_income = random.random() > 0.5  # Only 50% have income
        profile.can_afford_payment = random.random() > 0.5
        profile.life_event = random.choice([
            LifeEvent.JOB_LOSS, LifeEvent.MEDICAL, LifeEvent.DIVORCE
        ])
        profile.call_history = random.choice([
            CallHistory.REPEAT_CALLER, CallHistory.ESCALATED
        ])
        profile.__post_init__()
        return profile
    
    def to_prompt_context(self, profile: DebtorProfile) -> str:
        """
        Convert profile to natural language for LLM prompt.
        
        Args:
            profile: DebtorProfile to convert
            
        Returns:
            String description for LLM prompt
        """
        # Personality description
        personality_parts = []
        
        if profile.agreeableness > 0.7:
            personality_parts.append("generally agreeable and cooperative")
        elif profile.agreeableness < 0.3:
            personality_parts.append("disagreeable and resistant")
        
        if profile.emotional_stability > 0.7:
            personality_parts.append("calm and composed")
        elif profile.emotional_stability < 0.3:
            personality_parts.append("emotionally volatile")
        
        if profile.assertiveness > 0.7:
            personality_parts.append("assertive and firm")
        elif profile.assertiveness < 0.3:
            personality_parts.append("passive and hesitant")
        
        personality = ", ".join(personality_parts) if personality_parts else "neutral temperament"
        
        # Financial situation
        financial = f"${profile.debt_amount:,.0f} debt, {profile.days_overdue} days overdue"
        
        if profile.financial_stress > 0.7:
            financial += ", under severe financial stress"
        elif profile.financial_stress < 0.3:
            financial += ", manageable financial situation"
        
        if not profile.has_income:
            financial += ", currently no income"
        
        if not profile.can_afford_payment:
            financial += ", genuinely cannot afford payments right now"
        
        # Life event
        life_event_desc = ""
        if profile.life_event == LifeEvent.JOB_LOSS:
            life_event_desc = "Recently lost their job."
        elif profile.life_event == LifeEvent.MEDICAL:
            life_event_desc = "Dealing with a medical emergency."
        elif profile.life_event == LifeEvent.DIVORCE:
            life_event_desc = "Going through a divorce."
        elif profile.life_event == LifeEvent.FAMILY_EMERGENCY:
            life_event_desc = "Dealing with a family emergency."
        elif profile.life_event == LifeEvent.BUSINESS_FAILURE:
            life_event_desc = "Their business recently failed."
        elif profile.life_event == LifeEvent.REDUCED_HOURS:
            life_event_desc = "Work hours have been significantly reduced."
        
        # Call history
        call_desc = ""
        if profile.call_history == CallHistory.FIRST_CALL:
            call_desc = "This is their first call from collections."
        elif profile.call_history == CallHistory.REPEAT_CALLER:
            call_desc = "They've received several collection calls before."
        elif profile.call_history == CallHistory.ESCALATED:
            call_desc = "This is an escalated call after previous unsuccessful attempts."
        elif profile.call_history == CallHistory.DISPUTED:
            call_desc = "They previously disputed this debt."
        
        # Communication style
        comm_parts = []
        if profile.verbosity > 0.7:
            comm_parts.append("tends to give long explanations")
        elif profile.verbosity < 0.3:
            comm_parts.append("gives short, terse responses")
        
        if profile.directness > 0.7:
            comm_parts.append("very direct")
        elif profile.directness < 0.3:
            comm_parts.append("evasive and indirect")
        
        communication = ", ".join(comm_parts) if comm_parts else ""
        
        # Build full context
        context = f"""You are a debtor with the following characteristics:

PERSONALITY: {personality}

FINANCIAL SITUATION: {financial}

{life_event_desc}

{call_desc}

{f"COMMUNICATION STYLE: {communication}" if communication else ""}

IMPORTANT: Stay in character. Respond naturally as this person would.
Your initial mood is {"negative" if profile.initial_sentiment < -0.2 else "positive" if profile.initial_sentiment > 0.2 else "neutral"}.
You are {"willing to discuss options" if profile.initial_cooperation > 0.5 else "resistant to the conversation"}.
"""
        return context.strip()


# Quick test
if __name__ == "__main__":
    randomizer = DomainRandomizer(seed=42)
    
    print("=" * 70)
    print("Domain Randomization Test")
    print("=" * 70)
    
    # Sample a few profiles
    for i, difficulty in enumerate(['easy', 'medium', 'hard', 'random']):
        if difficulty == 'easy':
            profile = randomizer.sample_easy()
        elif difficulty == 'medium':
            profile = randomizer.sample_medium()
        elif difficulty == 'hard':
            profile = randomizer.sample_hard()
        else:
            profile = randomizer.sample()
        
        print(f"\n{'='*70}")
        print(f"Profile {i+1} ({difficulty.upper()})")
        print(f"{'='*70}")
        print(f"Debt: ${profile.debt_amount:,.0f}, {profile.days_overdue} days overdue")
        print(f"Personality: agree={profile.agreeableness:.2f}, stable={profile.emotional_stability:.2f}, assert={profile.assertiveness:.2f}")
        print(f"Stress: {profile.financial_stress:.2f}, Income: {profile.has_income}, Can Pay: {profile.can_afford_payment}")
        print(f"Life Event: {profile.life_event.value}")
        print(f"Initial: sentiment={profile.initial_sentiment:.2f}, cooperation={profile.initial_cooperation:.2f}")
        print(f"\nLLM Context Preview:")
        print("-" * 40)
        print(randomizer.to_prompt_context(profile)[:500] + "...")
