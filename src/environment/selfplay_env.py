"""
Self-Play Environment for Adversarial Training

Two-agent environment where:
- Collector Agent: Tries to secure payment commitment
- Adversary Agent: Tries to resist and exploit collector weaknesses

Both agents learn simultaneously, creating an ever-improving training curriculum.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass, field

from src.config import EnvironmentConfig, SelfPlayConfig
from src.environment.domain_randomizer import DomainRandomizer, DebtorProfile
from src.nlu.state_extractor import DebtorResponseAnalyzer, NLUFeatures
from src.llm.adversarial_prompts import (
    get_adversarial_response_prompt,
    ADVERSARIAL_DEBTOR_SYSTEM_PROMPT
)


@dataclass
class SelfPlayState:
    """Track state for both agents in self-play."""
    # Account info
    debt_amount: float = 5000.0
    days_overdue: int = 60
    
    # NLU-extracted features (updated each turn)
    sentiment: float = 0.0
    cooperation: float = 0.5
    intent: str = "neutral"
    
    # Behavioral signals
    has_shared_situation: bool = False
    feels_understood: bool = False
    has_commitment_signal: bool = False
    has_quit_signal: bool = False
    
    # Turn tracking
    turn: int = 0
    
    # Action history (for both agents)
    collector_actions: List[int] = field(default_factory=list)
    adversary_actions: List[int] = field(default_factory=list)
    
    # Conversation text history
    utterances: List[Dict[str, str]] = field(default_factory=list)
    
    # Episode outcome tracking
    collector_reward_total: float = 0.0
    adversary_reward_total: float = 0.0


class SelfPlayEnv(gym.Env):
    """
    Two-player debt collection environment for adversarial training.
    
    Turn Structure:
    1. Collector chooses action → generates utterance
    2. Adversary chooses response strategy → generates response
    3. NLU extracts new state
    4. Both agents receive rewards (zero-sum or weighted)
    
    The environment can run in two modes:
    - Training Mode: Both agents learn
    - Evaluation Mode: One agent is frozen (loaded from checkpoint)
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Intent mapping (same as NLUDebtCollectionEnv)
    INTENT_MAP = {
        "committing": 0, "willing": 1, "explaining": 2, "questioning": 3,
        "neutral": 4, "avoidant": 5, "refusing": 6, "hostile": 7
    }
    
    def __init__(
        self,
        llm_client=None,
        render_mode: Optional[str] = None,
        use_llm_for_collector: bool = True,
        use_llm_for_adversary: bool = True,
    ):
        """
        Initialize Self-Play Environment.
        
        Args:
            llm_client: LLM client for generating text
            render_mode: 'human' for text output
            use_llm_for_collector: Use LLM for collector utterances
            use_llm_for_adversary: Use LLM for adversary responses
        """
        super().__init__()
        
        self.llm_client = llm_client
        self.render_mode = render_mode
        self.use_llm_for_collector = use_llm_for_collector
        self.use_llm_for_adversary = use_llm_for_adversary
        
        # Initialize components
        self.domain_randomizer = DomainRandomizer()
        self.nlu_analyzer = DebtorResponseAnalyzer()
        
        # Action spaces for both agents
        self.collector_action_space = spaces.Discrete(EnvironmentConfig.NUM_ACTIONS)
        self.adversary_action_space = spaces.Discrete(SelfPlayConfig.NUM_ADVERSARY_ACTIONS)
        
        # Shared observation space (both agents see same state)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(EnvironmentConfig.NLU_STATE_DIM,),
            dtype=np.float32
        )
        
        # Episode state
        self.profile: Optional[DebtorProfile] = None
        self.state: Optional[SelfPlayState] = None
        
        # Statistics
        self.episodes_completed = 0
        self.collector_wins = 0
        self.adversary_wins = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset for new episode.
        
        Returns observation (same for both agents) and info dict.
        """
        super().reset(seed=seed)
        
        # Sample a debtor profile (provides context for adversary LLM)
        self.profile = self.domain_randomizer.sample()
        
        # Initialize state
        self.state = SelfPlayState(
            debt_amount=self.profile.debt_amount,
            days_overdue=self.profile.days_overdue,
            sentiment=self.profile.initial_sentiment,
            cooperation=self.profile.initial_cooperation
        )
        
        observation = self._encode_state()
        
        info = {
            "debt_amount": self.profile.debt_amount,
            "days_overdue": self.profile.days_overdue,
            "turn": 0
        }
        
        if self.render_mode == "human":
            print(f"\n{'='*70}")
            print(f"SELF-PLAY BATTLE - Collector vs Adversary")
            print(f"Debt: ${self.profile.debt_amount:,.0f}, {self.profile.days_overdue} days overdue")
            print(f"{'='*70}\n")
        
        return observation, info
    
    def step(
        self,
        collector_action: int,
        adversary_action: int
    ) -> Tuple[np.ndarray, float, float, bool, bool, Dict[str, Any]]:
        """
        Execute one turn of self-play.
        
        Args:
            collector_action: Collector's chosen strategy (0-8)
            adversary_action: Adversary's chosen resistance strategy (0-6)
            
        Returns:
            observation: State observation (shared)
            collector_reward: Reward for collector agent
            adversary_reward: Reward for adversary agent
            terminated: Episode ended normally
            truncated: Episode hit max turns
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Call reset() first")
        
        self.state.turn += 1
        
        # Get action names
        collector_action_name = EnvironmentConfig.ACTIONS[collector_action]
        adversary_action_name = SelfPlayConfig.ADVERSARY_ACTIONS[adversary_action]
        
        # Track actions
        self.state.collector_actions.append(collector_action)
        self.state.adversary_actions.append(adversary_action)
        
        # Store previous state for reward calculation
        prev_sentiment = self.state.sentiment
        prev_cooperation = self.state.cooperation
        
        # 1. Generate collector utterance
        if self.llm_client and self.use_llm_for_collector:
            collector_utterance = self._generate_collector_utterance(collector_action_name)
        else:
            collector_utterance = f"[COLLECTOR: {collector_action_name}]"
        
        # 2. Generate adversary response using chosen strategy
        if self.llm_client and self.use_llm_for_adversary:
            adversary_response = self._generate_adversary_response(
                collector_utterance, adversary_action_name
            )
        else:
            adversary_response = self._generate_rule_based_adversary_response(
                adversary_action_name
            )
        
        # Store in history
        self.state.utterances.append({
            "turn": self.state.turn,
            "collector_action": collector_action_name,
            "collector_utterance": collector_utterance,
            "adversary_action": adversary_action_name,
            "adversary_response": adversary_response
        })
        
        # 3. Extract NLU features from adversary response
        nlu_features = self.nlu_analyzer.analyze(adversary_response)
        
        # 4. Update state
        self._update_state_from_nlu(nlu_features)
        
        # 5. Check termination
        terminated, truncated, outcome = self._check_termination()
        
        # 6. Calculate rewards for both agents
        collector_reward, adversary_reward = self._calculate_rewards(
            collector_action_name,
            adversary_action_name,
            prev_sentiment,
            prev_cooperation,
            terminated,
            truncated,
            outcome
        )
        
        self.state.collector_reward_total += collector_reward
        self.state.adversary_reward_total += adversary_reward
        
        # 7. Get observation
        observation = self._encode_state()
        
        # 8. Build info dict
        info = {
            "collector_action": collector_action_name,
            "adversary_action": adversary_action_name,
            "sentiment": self.state.sentiment,
            "cooperation": self.state.cooperation,
            "intent": self.state.intent,
            "turn": self.state.turn,
            "outcome": outcome,
            "collector_total_reward": self.state.collector_reward_total,
            "adversary_total_reward": self.state.adversary_reward_total
        }
        
        if self.render_mode == "human":
            self._render_turn(
                collector_action_name, collector_utterance,
                adversary_action_name, adversary_response,
                collector_reward, adversary_reward
            )
        
        # Update stats on episode end
        if terminated or truncated:
            self.episodes_completed += 1
            if outcome == "collector_win":
                self.collector_wins += 1
            elif outcome == "adversary_win":
                self.adversary_wins += 1
            
            if self.render_mode == "human":
                self._render_episode_end(outcome)
        
        return observation, collector_reward, adversary_reward, terminated, truncated, info
    
    def _encode_state(self) -> np.ndarray:
        """Encode state to observation vector (same as NLUDebtCollectionEnv)."""
        obs = []
        
        # Normalized account info
        debt_norm = np.clip(self.state.debt_amount / 25000 - 1, -1, 1)
        days_norm = np.clip(self.state.days_overdue / 180 - 1, -1, 1)
        obs.extend([debt_norm, days_norm])
        
        # Behavioral features
        obs.append(self.state.sentiment)
        obs.append(self.state.cooperation * 2 - 1)
        
        # Intent one-hot (8 dimensions)
        intent_vec = [0.0] * 8
        intent_idx = self.INTENT_MAP.get(self.state.intent, 4)
        intent_vec[intent_idx] = 1.0
        obs.extend(intent_vec)
        
        # Boolean signals
        obs.append(1.0 if self.state.has_shared_situation else -1.0)
        obs.append(1.0 if self.state.feels_understood else -1.0)
        obs.append(1.0 if self.state.has_commitment_signal else -1.0)
        obs.append(1.0 if self.state.has_quit_signal else -1.0)
        
        # Turn progress
        turn_norm = np.clip(self.state.turn / EnvironmentConfig.MAX_TURNS * 2 - 1, -1, 1)
        obs.append(turn_norm)
        
        # Has mentioned payment/consequences (for collector)
        mentioned_plan = any(
            a in [3, 4, 7]  # payment_plan, settlement, validate_then_offer
            for a in self.state.collector_actions
        )
        mentioned_conseq = any(
            a in [2, 5]  # firm_reminder, hard_close
            for a in self.state.collector_actions
        )
        obs.append(1.0 if mentioned_plan else -1.0)
        obs.append(1.0 if mentioned_conseq else -1.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _update_state_from_nlu(self, features: NLUFeatures):
        """Update state from NLU extraction."""
        self.state.sentiment = features.sentiment
        self.state.cooperation = features.cooperation
        self.state.intent = features.intent
        
        if features.shared_situation:
            self.state.has_shared_situation = True
        if features.feels_understood:
            self.state.feels_understood = True
        if features.commitment_signal:
            self.state.has_commitment_signal = True
        if features.quit_signal:
            self.state.has_quit_signal = True
    
    def _check_termination(self) -> Tuple[bool, bool, str]:
        """
        Check if episode should end.
        
        Returns:
            terminated: Episode ended normally
            truncated: Hit max turns
            outcome: 'collector_win', 'adversary_win', 'draw', or 'ongoing'
        """
        terminated = False
        truncated = False
        outcome = "ongoing"
        
        # Collector wins: commitment achieved
        if self.state.has_commitment_signal and self.state.cooperation > 0.7:
            terminated = True
            outcome = "collector_win"
        # Adversary wins: debtor quits
        elif self.state.has_quit_signal or self.state.sentiment < -0.9:
            terminated = True
            outcome = "adversary_win"
        # Truncated: max turns
        elif self.state.turn >= EnvironmentConfig.MAX_TURNS:
            truncated = True
            # No commitment = adversary wins
            if not self.state.has_commitment_signal:
                outcome = "adversary_win"
            else:
                outcome = "draw"
        
        return terminated, truncated, outcome
    
    def _calculate_rewards(
        self,
        collector_action: str,
        adversary_action: str,
        prev_sentiment: float,
        prev_cooperation: float,
        terminated: bool,
        truncated: bool,
        outcome: str
    ) -> Tuple[float, float]:
        """
        Calculate rewards for both agents.
        
        Uses zero-sum coefficient to balance competitive/cooperative dynamics.
        """
        collector_reward = 0.0
        adversary_reward = 0.0
        
        # Get reward events from config
        events = SelfPlayConfig.REWARD_EVENTS
        
        # === EPISODE OUTCOME REWARDS ===
        if outcome == "collector_win":
            c_r, a_r = events["payment_commitment"]
            collector_reward += c_r
            adversary_reward += a_r
        elif outcome == "adversary_win":
            if self.state.has_quit_signal or self.state.sentiment < -0.9:
                c_r, a_r = events["debtor_hangs_up"]
            else:
                c_r, a_r = events["conversation_end_no_commit"]
            collector_reward += c_r
            adversary_reward += a_r
        
        # === PER-TURN REWARDS ===
        if outcome == "ongoing":
            c_r, a_r = events["per_turn_no_commit"]
            collector_reward += c_r
            adversary_reward += a_r
        
        # === BEHAVIORAL CHANGE REWARDS ===
        sentiment_change = self.state.sentiment - prev_sentiment
        cooperation_change = self.state.cooperation - prev_cooperation
        
        # Collector benefits from sentiment/cooperation improvement
        collector_reward += sentiment_change * 2.0
        collector_reward += cooperation_change * 2.0
        
        # Adversary benefits from sentiment/cooperation decline
        adversary_reward -= sentiment_change * 1.5
        adversary_reward -= cooperation_change * 1.5
        
        # === STRATEGY-SPECIFIC BONUSES ===
        
        # Adversary bonus for using effective resistance
        if self.state.intent in ["hostile", "refusing", "avoidant"]:
            adversary_reward += 0.5  # Kept debtor resistant
        
        # Collector bonus for de-escalation
        if prev_sentiment < -0.5 and sentiment_change > 0.2:
            collector_reward += SelfPlayConfig.DIFFICULT_CONVERSION_BONUS
        
        # === SCALE REWARDS ===
        collector_reward *= SelfPlayConfig.COLLECTOR_REWARD_SCALE
        adversary_reward *= SelfPlayConfig.ADVERSARY_REWARD_SCALE
        
        return collector_reward, adversary_reward
    
    def _generate_collector_utterance(self, action_name: str) -> str:
        """Generate collector utterance using LLM."""
        history_text = self._get_conversation_history_text()
        return self.llm_client.generate_agent_utterance(
            strategy=action_name,
            conversation_history=history_text,
            turn=self.state.turn
        )
    
    def _generate_adversary_response(
        self,
        collector_utterance: str,
        strategy: str
    ) -> str:
        """Generate adversary response using LLM with specific strategy."""
        history_text = self._get_conversation_history_text()
        
        prompt = get_adversarial_response_prompt(
            strategy=strategy,
            conversation_history=history_text,
            agent_utterance=collector_utterance,
            debt_amount=self.state.debt_amount,
            days_overdue=self.state.days_overdue
        )
        
        # Use LLM to generate adversarial response
        try:
            response_text = self.llm_client._call_api(
                system_prompt=ADVERSARIAL_DEBTOR_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=0.8
            )
            
            # Strip any JSON formatting, just return the response text
            if response_text:
                return response_text.strip()
            return f"[{strategy}] I don't know..."
        except Exception as e:
            print(f"[WARN] Adversary LLM error: {e}")
            return self._generate_rule_based_adversary_response(strategy)
    
    def _generate_rule_based_adversary_response(self, strategy: str) -> str:
        """Generate rule-based adversary response (for no-LLM mode)."""
        responses = {
            "aggressive": [
                "Stop calling me! This is harassment!",
                "I'm going to report you people!",
                "Get lost! I don't owe you anything!",
            ],
            "evasive": [
                "Hmm, I'm not sure... let me think about it...",
                "Can you call back later? This isn't a good time.",
                "What was that? I couldn't hear you clearly...",
            ],
            "emotional": [
                "I can't take this anymore... *crying*",
                "You don't understand how hard things have been...",
                "Please... I'm barely surviving as it is...",
            ],
            "negotiate_hard": [
                "I'll only pay if you give me 90% off.",
                "5 dollars a month, that's all I can do. Take it or leave it.",
                "Unless you remove all interest, forget it.",
            ],
            "partial_cooperate": [
                "Maybe I could possibly do something... let me check...",
                "I guess I do owe something... but I need to verify first.",
                "That might work... but I need to talk to my family first.",
            ],
            "stall": [
                "Send me all the documents first.",
                "Can we schedule a call for next month?",
                "I need to review my finances and get back to you.",
            ],
            "dispute": [
                "I don't recognize this debt. Prove it's mine.",
                "The amount is completely wrong! Where's the breakdown?",
                "This might be identity theft. I'm filing a report.",
            ],
        }
        
        import random
        options = responses.get(strategy, ["I'm not sure what you mean."])
        return random.choice(options)
    
    def _get_conversation_history_text(self) -> str:
        """Get formatted conversation history."""
        if not self.state.utterances:
            return "(First turn)"
        
        lines = []
        for turn in self.state.utterances:
            lines.append(f"Collector: {turn['collector_utterance']}")
            lines.append(f"Debtor: {turn['adversary_response']}")
        
        return "\n".join(lines)
    
    def _render_turn(
        self,
        collector_action: str,
        collector_utterance: str,
        adversary_action: str,
        adversary_response: str,
        collector_reward: float,
        adversary_reward: float
    ):
        """Render turn to console."""
        print(f"--- Turn {self.state.turn} ---")
        print(f"[COLLECTOR] Strategy: {collector_action}")
        print(f"  \"{collector_utterance}\"")
        print(f"[ADVERSARY] Strategy: {adversary_action}")
        print(f"  \"{adversary_response}\"")
        print(f"State: sentiment={self.state.sentiment:.2f}, coop={self.state.cooperation:.2f}, intent={self.state.intent}")
        print(f"Rewards: Collector={collector_reward:+.2f}, Adversary={adversary_reward:+.2f}")
        print()
    
    def _render_episode_end(self, outcome: str):
        """Render episode summary."""
        print(f"\n{'='*70}")
        if outcome == "collector_win":
            print("🏆 COLLECTOR WINS - Payment commitment achieved!")
        elif outcome == "adversary_win":
            print("🛡️ ADVERSARY WINS - Resisted until the end!")
        else:
            print("🤝 DRAW - Time ran out")
        
        print(f"\nEpisode Stats:")
        print(f"  Turns: {self.state.turn}")
        print(f"  Collector Total Reward: {self.state.collector_reward_total:+.2f}")
        print(f"  Adversary Total Reward: {self.state.adversary_reward_total:+.2f}")
        print(f"  Win Rates: Collector {self.collector_wins}/{self.episodes_completed} | Adversary {self.adversary_wins}/{self.episodes_completed}")
        print(f"{'='*70}\n")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics."""
        return {
            "episodes_completed": self.episodes_completed,
            "collector_wins": self.collector_wins,
            "adversary_wins": self.adversary_wins,
            "collector_win_rate": self.collector_wins / max(1, self.episodes_completed),
            "adversary_win_rate": self.adversary_wins / max(1, self.episodes_completed),
        }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing SelfPlayEnv (no LLM)")
    print("=" * 70)
    
    env = SelfPlayEnv(llm_client=None, render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a few random turns
    for _ in range(5):
        collector_action = env.collector_action_space.sample()
        adversary_action = env.adversary_action_space.sample()
        
        obs, c_reward, a_reward, terminated, truncated, info = env.step(
            collector_action, adversary_action
        )
        
        if terminated or truncated:
            break
    
    print("\nFinal Statistics:")
    print(env.get_statistics())
    print("\n✓ Test complete!")
