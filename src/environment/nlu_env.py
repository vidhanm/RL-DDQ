"""
NLU-Enhanced Debt Collection Environment

Key Changes from debtor_env.py:
1. Uses DomainRandomizer instead of 4 discrete personas
2. Uses NLU to extract state from LLM text (deterministic)
3. Agent never sees persona parameters - only NLU-extracted features
4. Same state representation works for training AND production
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from dataclasses import dataclass

from src.config import EnvironmentConfig
from src.environment.domain_randomizer import DomainRandomizer, DebtorProfile
from src.nlu.state_extractor import DebtorResponseAnalyzer, NLUFeatures
from src.nlu.debtor_classifier import DebtorTypeClassifier


@dataclass
class ConversationState:
    """Track observable conversation state (no hidden persona info)"""
    # Account info (known at call start)
    debt_amount: float = 5000.0
    days_overdue: int = 60
    
    # NLU-extracted behavioral features (updated from debtor text)
    sentiment: float = 0.0
    cooperation: float = 0.5
    intent: str = "neutral"
    
    # Behavioral signals (accumulated boolean flags)
    has_shared_situation: bool = False
    feels_understood: bool = False
    has_commitment_signal: bool = False
    has_quit_signal: bool = False
    
    # Conversation tracking
    turn: int = 0
    mentioned_payment_plan: bool = False
    mentioned_consequences: bool = False
    
    # Action history for expert reward calculations
    action_history: List[str] = None
    action_results: List[dict] = None  # Track if each action improved or worsened situation
    
    # History for computing changes
    sentiment_history: List[float] = None
    cooperation_history: List[float] = None
    
    def __post_init__(self):
        if self.sentiment_history is None:
            self.sentiment_history = [self.sentiment]
        if self.cooperation_history is None:
            self.cooperation_history = [self.cooperation]
        if self.action_history is None:
            self.action_history = []
        if self.action_results is None:
            self.action_results = []
    
    def update_from_nlu(self, features: NLUFeatures):
        """Update state from NLU extraction"""
        self.sentiment = features.sentiment
        self.cooperation = features.cooperation
        self.intent = features.intent
        
        # Accumulate boolean signals (once True, stays True)
        if features.shared_situation:
            self.has_shared_situation = True
        if features.feels_understood:
            self.feels_understood = True
        if features.commitment_signal:
            self.has_commitment_signal = True
        if features.quit_signal:
            self.has_quit_signal = True
        
        # Track history
        self.sentiment_history.append(self.sentiment)
        self.cooperation_history.append(self.cooperation)


class NLUDebtCollectionEnv(gym.Env):
    """
    NLU-Enhanced Debt Collection Environment
    
    Uses:
    - Domain randomization for diverse debtor profiles
    - NLU for deterministic state extraction from LLM text
    - Agent sees only observable behavioral features
    """
    
    metadata = {"render_modes": ["human"]}
    
    # Intent to index mapping for state encoding
    INTENT_MAP = {
        "committing": 0,
        "willing": 1,
        "explaining": 2,
        "questioning": 3,
        "neutral": 4,
        "avoidant": 5,
        "refusing": 6,
        "hostile": 7
    }
    
    def __init__(
        self,
        llm_client=None,
        render_mode: Optional[str] = None,
        use_domain_randomization: bool = True,
        difficulty: Optional[str] = None  # 'easy', 'medium', 'hard', or None for random
    ):
        """
        Initialize environment
        
        Args:
            llm_client: LLM client for generating text
            render_mode: 'human' for text output
            use_domain_randomization: If True, sample random profiles
            difficulty: Curriculum difficulty level
        """
        super().__init__()
        
        self.llm_client = llm_client
        self.render_mode = render_mode
        self.use_domain_randomization = use_domain_randomization
        self.difficulty = difficulty
        
        # Initialize components
        self.domain_randomizer = DomainRandomizer()
        self.nlu_analyzer = DebtorResponseAnalyzer()
        self.debtor_classifier = DebtorTypeClassifier(min_turns=2)
        
        # NLU history for opponent modeling (classify debtor type after 2+ turns)
        self.nlu_history: List[NLUFeatures] = []
        
        # Action space
        self.action_space = spaces.Discrete(EnvironmentConfig.NUM_ACTIONS)
        
        # Observation space: NLU-based state
        # [debt_norm, days_norm, sentiment, cooperation, intent_onehot(8), 
        #  shared_sit, feels_understood, commitment, quit, turn_norm, 
        #  mentioned_plan, mentioned_conseq]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(19,),  # 2 + 2 + 8 + 4 + 1 + 2 = 19
            dtype=np.float32
        )
        
        # Episode state
        self.profile: Optional[DebtorProfile] = None
        self.state: Optional[ConversationState] = None
        self.conversation_history: List[Dict] = []
        self.episode_reward: float = 0.0
        
        # Milestone tracking for step-by-step rewards
        self._milestone_shared_situation = False
        self._milestone_feels_understood = False
        self._milestone_discussing_options = False
        self._milestone_first_positive_sentiment = False  # NEW
        self._milestone_cooperation_above_50 = False       # NEW
        self._milestone_question_answered = False          # NEW
        self._prev_question_count = 0                      # Track if debtor asked questions
        
        # Statistics
        self.episodes_completed = 0
        self.successful_episodes = 0
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset for new episode"""
        super().reset(seed=seed)
        
        # Sample debtor profile (hidden from agent)
        difficulty = self.difficulty
        if options and "difficulty" in options:
            difficulty = options["difficulty"]
        
        if self.use_domain_randomization:
            if difficulty == "easy":
                self.profile = self.domain_randomizer.sample_easy()
            elif difficulty == "medium":
                self.profile = self.domain_randomizer.sample_medium()
            elif difficulty == "hard":
                self.profile = self.domain_randomizer.sample_hard()
            else:
                self.profile = self.domain_randomizer.sample()
        else:
            # Default profile for testing
            self.profile = DebtorProfile()
        
        # Initialize conversation state with observable info only
        self.state = ConversationState(
            debt_amount=self.profile.debt_amount,
            days_overdue=self.profile.days_overdue,
            sentiment=self.profile.initial_sentiment,
            cooperation=self.profile.initial_cooperation
        )
        
        self.conversation_history = []
        self.nlu_history = []  # Reset for opponent modeling
        self.episode_reward = 0.0
        
        # Reset step-by-step milestones
        self._milestone_shared_situation = False
        self._milestone_feels_understood = False
        self._milestone_discussing_options = False
        self._milestone_first_positive_sentiment = False
        self._milestone_cooperation_above_50 = False
        self._milestone_question_answered = False
        self._prev_question_count = 0
        
        observation = self._encode_state()
        
        info = {
            "debt_amount": self.profile.debt_amount,
            "days_overdue": self.profile.days_overdue,
            "initial_sentiment": self.profile.initial_sentiment,
            "initial_cooperation": self.profile.initial_cooperation
        }
        
        if self.render_mode == "human":
            print(f"\n{'='*70}")
            print(f"NEW CONVERSATION - NLU Environment")
            print(f"Debt: ${self.profile.debt_amount:,.0f}, {self.profile.days_overdue} days overdue")
            print(f"{'='*70}\n")
        
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step"""
        if self.state is None:
            raise RuntimeError("Call reset() first")
        
        action_name = EnvironmentConfig.ACTIONS[action]
        self.state.turn += 1
        
        # Update action flags
        if action_name == "offer_payment_plan":
            self.state.mentioned_payment_plan = True
        elif action_name in ["firm_reminder", "hard_close"]:
            self.state.mentioned_consequences = True
        
        # Generate agent utterance
        if self.llm_client:
            agent_utterance = self._generate_agent_utterance(action_name)
        else:
            agent_utterance = f"[{action_name}]"
        
        # Generate debtor response
        if self.llm_client:
            debtor_response = self._generate_debtor_response(agent_utterance)
        else:
            debtor_response = self._generate_rule_based_response(action_name)
        
        # Extract NLU features from debtor response (DETERMINISTIC)
        nlu_features = self.nlu_analyzer.analyze(debtor_response)
        
        # =====================================================================
        # OPPONENT MODELING: Classify debtor type after 2+ turns
        # =====================================================================
        self.nlu_history.append(nlu_features)
        
        if len(self.nlu_history) >= 2:
            debtor_type, confidence = self.debtor_classifier.classify_from_nlu_features(
                self.nlu_history
            )
            nlu_features.inferred_type = debtor_type
            nlu_features.type_confidence = confidence
        
        # Update state from NLU
        prev_sentiment = self.state.sentiment
        prev_cooperation = self.state.cooperation
        self.state.update_from_nlu(nlu_features)
        
        # Store conversation
        self.conversation_history.append({
            "turn": self.state.turn,
            "action": action_name,
            "agent_utterance": agent_utterance,
            "debtor_response": debtor_response,
            "nlu_features": nlu_features,
            "inferred_type": nlu_features.inferred_type  # Track for logging
        })
        
        # Check termination
        terminated = False
        truncated = False
        
        if self.state.has_commitment_signal and self.state.cooperation > 0.7:
            terminated = True  # Success
        elif self.state.has_quit_signal or self.state.sentiment < -0.9:
            terminated = True  # Debtor quit
        elif self.state.turn >= EnvironmentConfig.MAX_TURNS:
            truncated = True
        
        # Calculate reward
        reward = self._calculate_reward(
            action_name, 
            prev_sentiment, 
            prev_cooperation,
            terminated, 
            truncated
        )
        self.episode_reward += reward
        
        observation = self._encode_state()
        
        info = {
            "action_name": action_name,
            "sentiment": self.state.sentiment,
            "cooperation": self.state.cooperation,
            "intent": self.state.intent,
            "has_committed": self.state.has_commitment_signal and self.state.cooperation > 0.7,
            "turn": self.state.turn,
            "episode_reward": self.episode_reward
        }
        
        if self.render_mode == "human":
            self._render_turn(action_name, agent_utterance, debtor_response, nlu_features, reward)
        
        # Update stats
        if terminated or truncated:
            self.episodes_completed += 1
            if self.state.has_commitment_signal and self.state.cooperation > 0.7:
                self.successful_episodes += 1
            
            if self.render_mode == "human":
                self._render_episode_end(terminated, truncated)
        
        return observation, reward, terminated, truncated, info
    
    def _encode_state(self) -> np.ndarray:
        """Encode state to observation vector"""
        obs = []
        
        # Normalized account info
        debt_norm = np.clip(self.state.debt_amount / 25000 - 1, -1, 1)
        days_norm = np.clip(self.state.days_overdue / 180 - 1, -1, 1)
        obs.extend([debt_norm, days_norm])
        
        # Behavioral features from NLU
        obs.append(self.state.sentiment)
        obs.append(self.state.cooperation * 2 - 1)  # Map 0-1 to -1 to 1
        
        # Intent one-hot (8 dimensions)
        intent_vec = [0.0] * 8
        intent_idx = self.INTENT_MAP.get(self.state.intent, 4)  # Default neutral
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
        
        # Action history
        obs.append(1.0 if self.state.mentioned_payment_plan else -1.0)
        obs.append(1.0 if self.state.mentioned_consequences else -1.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(
        self,
        action_name: str,
        prev_sentiment: float,
        prev_cooperation: float,
        terminated: bool,
        truncated: bool
    ) -> float:
        """
        Calculate reward based on NLU-extracted features + expert knowledge.
        
        Expert rewards encourage behaviors proven effective by debt collection research:
        - Open with empathy before pressure tactics
        - De-escalate hostile situations
        - Offer flexible solutions
        - Avoid premature hard close
        """
        reward = 0.0
        
        # Track sentiment/cooperation changes for expert rewards
        sentiment_change = self.state.sentiment - prev_sentiment
        cooperation_change = self.state.cooperation - prev_cooperation
        action_improved = sentiment_change > 0 or cooperation_change > 0
        action_worsened = sentiment_change < -0.2 or cooperation_change < -0.1
        
        # Record action result for future reference
        self.state.action_history.append(action_name)
        self.state.action_results.append({
            'action': action_name,
            'improved': action_improved,
            'worsened': action_worsened,
            'sentiment_change': sentiment_change,
            'cooperation_change': cooperation_change
        })
        
        # =====================================================================
        # PRIMARY REWARDS (existing)
        # =====================================================================
        
        # PRIMARY: Commitment success
        if self.state.has_commitment_signal and self.state.cooperation > 0.7:
            reward += EnvironmentConfig.REWARD_COMMITMENT
        
        # MILESTONE: Shared situation
        if self.state.has_shared_situation and not self._milestone_shared_situation:
            reward += 1.0
            self._milestone_shared_situation = True
        
        # MILESTONE: Feels understood
        if self.state.feels_understood and not self._milestone_feels_understood:
            reward += 1.5
            self._milestone_feels_understood = True
        
        # MILESTONE: Discussing options
        if (self.state.mentioned_payment_plan and 
            self.state.cooperation > 0.5 and 
            not self._milestone_discussing_options):
            reward += 2.0
            self._milestone_discussing_options = True
        
        # =====================================================================
        # NEW PROGRESSIVE MILESTONES (step-by-step rewards)
        # =====================================================================
        
        # MILESTONE: First positive sentiment (from negative/neutral)
        if (self.state.sentiment > 0.1 and 
            prev_sentiment <= 0.1 and 
            not self._milestone_first_positive_sentiment):
            reward += 0.8  # Turning the conversation positive
            self._milestone_first_positive_sentiment = True
        
        # MILESTONE: Cooperation crossed 50% threshold
        if (self.state.cooperation > 0.5 and 
            not self._milestone_cooperation_above_50):
            reward += 1.0  # Getting debtor to cooperate
            self._milestone_cooperation_above_50 = True
        
        # MILESTONE: Debtor asked question (engaging) and we addressed it
        current_nlu = self.nlu_history[-1] if self.nlu_history else None
        if current_nlu and self._prev_question_count > 0:
            # Previous turn had questions, this turn showing engagement
            if current_nlu.intent in ['willing', 'explaining', 'committing']:
                if not self._milestone_question_answered:
                    reward += 0.5  # Successfully addressed their questions
                    self._milestone_question_answered = True
        
        # Track questions for next turn
        if current_nlu:
            self._prev_question_count = current_nlu.question_count
        
        # =====================================================================
        # EXPERT KNOWLEDGE REWARDS (new)
        # =====================================================================
        
        # Check if empathy was used before pressure
        used_empathy = any(a in EnvironmentConfig.EMPATHETIC_ACTIONS 
                          for a in self.state.action_history)
        is_pressure_action = action_name in EnvironmentConfig.PRESSURE_ACTIONS
        is_solution_action = action_name in EnvironmentConfig.SOLUTION_ACTIONS
        
        # Reward: Empathy before pressure (expert best practice)
        if is_pressure_action and self.state.turn == 1:
            # First action is pressure - bad!
            reward += EnvironmentConfig.EXPERT_PENALTIES.get('premature_hard_close', -3.0)
        elif is_pressure_action and used_empathy and self.state.turn >= 2:
            # Used empathy first, then pressure - good!
            if action_improved or self.state.cooperation > 0.5:
                reward += EnvironmentConfig.EXPERT_REWARDS.get('empathy_before_pressure', 2.0)
        
        # Reward: De-escalating hostile debtor
        if prev_sentiment < -0.5 and self.state.sentiment > prev_sentiment + 0.2:
            # Significant sentiment improvement from hostile state
            reward += EnvironmentConfig.EXPERT_REWARDS.get('de_escalate_hostility', 3.0)
        
        # Reward: Offering solution when debtor is willing
        if is_solution_action and self.state.intent in ['willing', 'explaining']:
            reward += EnvironmentConfig.EXPERT_REWARDS.get('offer_flexible_options', 2.0)
        
        # Reward: Asked about situation and debtor shared
        if action_name == 'ask_about_situation' and self.state.has_shared_situation:
            reward += EnvironmentConfig.EXPERT_REWARDS.get('acknowledge_situation', 2.0)
        
        # Reward: Recovery after negative turn (resilience)
        if len(self.state.action_results) >= 2:
            prev_result = self.state.action_results[-2]
            if prev_result.get('worsened', False) and action_improved:
                reward += EnvironmentConfig.EXPERT_REWARDS.get('recovered_from_negative', 2.5)
        
        # Penalty: Pressure on already hostile debtor (makes things worse)
        if is_pressure_action and prev_sentiment < -0.5:
            reward += EnvironmentConfig.EXPERT_PENALTIES.get('pressure_on_hostile', -3.0)
        
        # Penalty: Repeated failed strategy
        if len(self.state.action_results) >= 2:
            prev_result = self.state.action_results[-2]
            if (prev_result.get('action') == action_name and 
                prev_result.get('worsened', False)):
                reward += EnvironmentConfig.EXPERT_PENALTIES.get('repeated_failed_strategy', -2.0)
        
        # Penalty: Missed opportunity - debtor is willing but agent uses pressure
        if self.state.intent == 'willing' and is_pressure_action:
            reward += EnvironmentConfig.EXPERT_PENALTIES.get('missed_willing_opportunity', -1.5)
        
        # =====================================================================
        # FAILURE PENALTIES (existing)
        # =====================================================================
        
        if (terminated or truncated) and not (self.state.has_commitment_signal and self.state.cooperation > 0.7):
            if self.state.has_quit_signal or self.state.sentiment < -0.9:
                reward -= 5.0  # Debtor quit
            else:
                reward -= 3.0  # Timeout
        
        # =====================================================================
        # CONTINUOUS REWARDS (existing)
        # =====================================================================
        
        # Sentiment change
        reward += sentiment_change * EnvironmentConfig.REWARD_SENTIMENT_WEIGHT
        
        # Cooperation change
        reward += cooperation_change * EnvironmentConfig.REWARD_COOPERATION_WEIGHT
        
        # ENGAGEMENT: Based on intent
        if self.state.intent in ["committing", "willing"]:
            reward += 0.3  # Engaged positively
        elif self.state.intent in ["hostile", "refusing"]:
            reward -= 0.3  # Disengaged
        
        # HOSTILITY penalty
        if self.state.sentiment < -0.8:
            reward -= EnvironmentConfig.REWARD_HOSTILITY_PENALTY
        
        # TURN penalty
        reward -= EnvironmentConfig.REWARD_TURN_PENALTY
        
        return reward
    
    def _generate_agent_utterance(self, action_name: str) -> str:
        """Generate agent utterance using LLM"""
        history_text = self._get_conversation_history_text()
        
        return self.llm_client.generate_agent_utterance(
            strategy=action_name,
            conversation_history=history_text,
            turn=self.state.turn
        )
    
    def _generate_debtor_response(self, agent_utterance: str) -> str:
        """Generate debtor response using LLM with domain-randomized profile"""
        # Use domain randomizer to create prompt context
        debtor_context = self.domain_randomizer.to_prompt_context(self.profile)
        history_text = self._get_conversation_history_text()
        
        response_data = self.llm_client.generate_debtor_response(
            debtor_context=debtor_context,
            agent_utterance=agent_utterance,
            conversation_history=history_text
        )
        
        return response_data.get("response", "[Error]")
    
    def _generate_rule_based_response(self, action_name: str) -> str:
        """Generate rule-based response when no LLM (for testing/--no-llm mode)"""
        # Simple deterministic responses for testing
        responses = {
            "empathetic_listening": [
                "I appreciate you listening. I lost my job last month.",
                "Thanks for understanding. It's been really hard.",
                "Look, I'm trying my best here.",
            ],
            "ask_about_situation": [
                "I got laid off. It's been tough finding work.",
                "My car broke down and I had medical bills.",
                "Why do you need to know? Just leave me alone.",
            ],
            "firm_reminder": [
                "I know I owe the money. Stop calling me!",
                "Fine, I understand. What are my options?",
                "This is harassment!",
            ],
            "offer_payment_plan": [
                "Maybe I could do $50 a month...",
                "I can't afford anything right now.",
                "Okay, that might work. Let me think about it.",
            ],
            "propose_settlement": [
                "70%? I'll need to talk to my spouse.",
                "That's still too much. I can't do it.",
                "Okay, I think I can manage that. Let's do it.",
            ],
            "hard_close": [
                "Fine! I'll pay $100 today just stop calling!",
                "Go ahead and sue me, I have nothing!",
                "Okay okay, I'll make a payment. What do I do?",
            ]
        }
        
        import random
        options = responses.get(action_name, ["I'm not sure what to say."])
        return random.choice(options)
    
    def _get_conversation_history_text(self) -> str:
        """Get formatted conversation history"""
        if not self.conversation_history:
            return "(First turn)"
        
        lines = []
        for turn in self.conversation_history:
            lines.append(f"Agent: {turn['agent_utterance']}")
            lines.append(f"Debtor: {turn['debtor_response']}")
        
        return "\n".join(lines)
    
    def _render_turn(
        self,
        action_name: str,
        agent_utterance: str,
        debtor_response: str,
        nlu_features: NLUFeatures,
        reward: float
    ):
        """Render turn to console"""
        print(f"--- Turn {self.state.turn} ---")
        print(f"Strategy: {action_name}")
        print(f"Agent: {agent_utterance}")
        print(f"Debtor: {debtor_response}")
        print(f"NLU: intent={nlu_features.intent}, sent={nlu_features.sentiment:.2f}, coop={nlu_features.cooperation:.2f}")
        print(f"Reward: {reward:.2f}")
        print()
    
    def _render_episode_end(self, terminated: bool, truncated: bool):
        """Render episode summary"""
        print(f"\n{'='*70}")
        success = self.state.has_commitment_signal and self.state.cooperation > 0.7
        if success:
            print("✓ SUCCESS - Commitment achieved!")
        elif truncated:
            print("⚠ TRUNCATED - Max turns")
        else:
            print("✗ FAILED - Debtor quit")
        
        print(f"\nEpisode Stats:")
        print(f"  Turns: {self.state.turn}")
        print(f"  Total Reward: {self.episode_reward:.2f}")
        print(f"  Final Sentiment: {self.state.sentiment:.2f}")
        print(f"  Final Cooperation: {self.state.cooperation:.2f}")
        rate = 100 * self.successful_episodes / max(1, self.episodes_completed)
        print(f"  Success Rate: {self.successful_episodes}/{self.episodes_completed} ({rate:.1f}%)")
        print(f"{'='*70}\n")
    
    def get_success_rate(self) -> float:
        """Get success rate"""
        if self.episodes_completed == 0:
            return 0.0
        return self.successful_episodes / self.episodes_completed
    
    def close(self):
        """Clean up"""
        pass


# Test
if __name__ == "__main__":
    print("Testing NLU Environment (no LLM)")
    print("=" * 70)
    
    env = NLUDebtCollectionEnv(llm_client=None, render_mode="human")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation: {obs}")
    
    # Run a few random steps
    for _ in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
    
    env.close()
    print("\nTest complete!")
