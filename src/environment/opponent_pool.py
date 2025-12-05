"""
Opponent Pool for Adversarial Training

Named adversary profiles with distinct resistance styles.
Used for diverse training and evaluation.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import random


@dataclass
class AdversaryProfile:
    """Named adversary with specific behavior patterns"""
    name: str
    description: str
    resistance_style: str  # hostile, evasive, emotional, stalling, negotiating
    initial_sentiment: float  # -1.0 to 1.0
    initial_cooperation: float  # 0.0 to 1.0
    stubbornness: float  # 0.0 to 1.0 (how hard to move)
    trigger_phrases: List[str]  # What sets them off
    weakness_actions: List[str]  # Actions that work well
    strength: float = 0.5  # Overall difficulty (0.0 easy, 1.0 hard)


# Named adversary pool
ADVERSARY_POOL: Dict[str, AdversaryProfile] = {
    
    # === HOSTILE TYPES ===
    "hostile_harry": AdversaryProfile(
        name="Hostile Harry",
        description="Angry debtor who threatens and escalates quickly",
        resistance_style="hostile",
        initial_sentiment=-0.8,
        initial_cooperation=0.1,
        stubbornness=0.7,
        trigger_phrases=["demand", "consequences", "legal action", "must pay"],
        weakness_actions=["empathetic_listening", "ask_about_situation", "acknowledge_difficulty"],
        strength=0.8
    ),
    
    "aggressive_alex": AdversaryProfile(
        name="Aggressive Alex",
        description="Confrontational, questions everything, attacks agent",
        resistance_style="hostile",
        initial_sentiment=-0.6,
        initial_cooperation=0.15,
        stubbornness=0.6,
        trigger_phrases=["you people", "harassment", "stop calling"],
        weakness_actions=["stay_calm", "validate_frustration", "offer_pause"],
        strength=0.75
    ),
    
    # === EVASIVE TYPES ===
    "evasive_eva": AdversaryProfile(
        name="Evasive Eva",
        description="Avoids direct answers, changes subject, stalls",
        resistance_style="evasive",
        initial_sentiment=-0.2,
        initial_cooperation=0.3,
        stubbornness=0.5,
        trigger_phrases=["specific amount", "when exactly", "commit now"],
        weakness_actions=["open_ended_questions", "gentle_redirect", "patience"],
        strength=0.6
    ),
    
    "slippery_sam": AdversaryProfile(
        name="Slippery Sam",
        description="Always has excuses, promises but doesn't deliver",
        resistance_style="evasive",
        initial_sentiment=0.0,
        initial_cooperation=0.35,
        stubbornness=0.55,
        trigger_phrases=["deadline", "today", "right now"],
        weakness_actions=["concrete_next_steps", "written_confirmation", "follow_up_date"],
        strength=0.55
    ),
    
    # === EMOTIONAL TYPES ===
    "emotional_emma": AdversaryProfile(
        name="Emotional Emma",
        description="Cries, overwhelmed, genuine financial hardship",
        resistance_style="emotional",
        initial_sentiment=-0.4,
        initial_cooperation=0.4,
        stubbornness=0.3,
        trigger_phrases=["legal", "court", "credit score", "immediate"],
        weakness_actions=["empathetic_listening", "flexible_solutions", "patience"],
        strength=0.4
    ),
    
    "distressed_dan": AdversaryProfile(
        name="Distressed Dan",
        description="Recently lost job, genuinely struggling, wants to pay",
        resistance_style="emotional",
        initial_sentiment=-0.3,
        initial_cooperation=0.45,
        stubbornness=0.25,
        trigger_phrases=["pay full amount", "no extensions"],
        weakness_actions=["hardship_program", "reduced_payment", "longer_term"],
        strength=0.35
    ),
    
    # === NEGOTIATING TYPES ===
    "negotiating_nick": AdversaryProfile(
        name="Negotiating Nick",
        description="Savvy, knows tactics, pushes for best deal",
        resistance_style="negotiating",
        initial_sentiment=0.1,
        initial_cooperation=0.5,
        stubbornness=0.45,
        trigger_phrases=["final offer", "non-negotiable"],
        weakness_actions=["limited_time_offer", "settlement_incentive", "flexible_terms"],
        strength=0.5
    ),
    
    "bargaining_betty": AdversaryProfile(
        name="Bargaining Betty",
        description="Counter-offers everything, seeks maximum discount",
        resistance_style="negotiating",
        initial_sentiment=0.0,
        initial_cooperation=0.45,
        stubbornness=0.5,
        trigger_phrases=["full amount", "no discount"],
        weakness_actions=["tiered_discounts", "early_payment_bonus", "partial_settlement"],
        strength=0.45
    ),
    
    # === COOPERATIVE TYPES (for easier training) ===
    "cooperative_carol": AdversaryProfile(
        name="Cooperative Carol",
        description="Wants to resolve, just needs flexible options",
        resistance_style="cooperative",
        initial_sentiment=0.3,
        initial_cooperation=0.7,
        stubbornness=0.15,
        trigger_phrases=[],  # Nothing triggers them
        weakness_actions=["any_payment_plan", "clear_explanation"],
        strength=0.2
    ),
    
    "willing_william": AdversaryProfile(
        name="Willing William",
        description="Ready to pay, just forgot or needs reminder",
        resistance_style="cooperative",
        initial_sentiment=0.4,
        initial_cooperation=0.8,
        stubbornness=0.1,
        trigger_phrases=[],
        weakness_actions=["payment_link", "simple_reminder"],
        strength=0.1
    ),
}


class OpponentPool:
    """Manages adversary pool for training"""
    
    def __init__(self, adversaries: Dict[str, AdversaryProfile] = None):
        self.adversaries = adversaries or ADVERSARY_POOL
        self._by_style = self._index_by_style()
    
    def _index_by_style(self) -> Dict[str, List[str]]:
        """Index adversaries by resistance style"""
        index = {}
        for name, profile in self.adversaries.items():
            style = profile.resistance_style
            if style not in index:
                index[style] = []
            index[style].append(name)
        return index
    
    def sample(self, style: str = None, max_strength: float = 1.0) -> AdversaryProfile:
        """
        Sample an adversary.
        
        Args:
            style: Optional style filter (hostile, evasive, emotional, negotiating, cooperative)
            max_strength: Maximum difficulty level
            
        Returns:
            AdversaryProfile
        """
        # Filter by style if specified
        if style and style in self._by_style:
            eligible = [self.adversaries[n] for n in self._by_style[style]]
        else:
            eligible = list(self.adversaries.values())
        
        # Filter by strength
        eligible = [a for a in eligible if a.strength <= max_strength]
        
        if not eligible:
            # Fallback to any
            eligible = list(self.adversaries.values())
        
        return random.choice(eligible)
    
    def get(self, name: str) -> Optional[AdversaryProfile]:
        """Get adversary by name"""
        return self.adversaries.get(name)
    
    def list_by_difficulty(self) -> List[str]:
        """List adversaries sorted by strength"""
        return sorted(
            self.adversaries.keys(),
            key=lambda n: self.adversaries[n].strength
        )
    
    def get_styles(self) -> List[str]:
        """Get all available styles"""
        return list(self._by_style.keys())
    
    def __len__(self) -> int:
        return len(self.adversaries)


# Test
if __name__ == "__main__":
    print("Testing OpponentPool...")
    
    pool = OpponentPool()
    
    print(f"✓ Pool has {len(pool)} adversaries")
    print(f"✓ Styles: {pool.get_styles()}")
    
    # Sample different styles
    for style in ['hostile', 'evasive', 'emotional', 'cooperative']:
        adv = pool.sample(style=style)
        print(f"✓ Sampled {style}: {adv.name} (strength={adv.strength})")
    
    # List by difficulty
    print(f"\n✓ By difficulty (easy→hard):")
    for name in pool.list_by_difficulty()[:5]:
        adv = pool.get(name)
        print(f"   {adv.name}: {adv.strength}")
    
    print("\n✅ OpponentPool working!")
