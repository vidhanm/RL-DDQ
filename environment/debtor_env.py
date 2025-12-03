"""
Debt Collection Environment
Gymnasium-compatible environment for debt collection conversations
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any

from config import EnvironmentConfig
from environment.debtor_persona import (
    DebtorPersona,
    create_random_debtor,
    get_action_effectiveness
)
from utils.state_encoder import create_state_dict, get_encoder


class DebtCollectionEnv(gym.Env):
    """
    Debt Collection Conversation Environment

    Agent chooses high-level strategies, LLM generates actual utterances.
    Debtor (LLM-based) responds, environment updates state and calculates reward.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, llm_client=None, render_mode: Optional[str] = None):
        """
        Initialize environment

        Args:
            llm_client: LLM client for generating utterances (will be set later)
            render_mode: Rendering mode ("human" for text output)
        """
        super().__init__()

        self.llm_client = llm_client
        self.render_mode = render_mode

        # Action space: discrete actions (strategies)
        self.action_space = spaces.Discrete(EnvironmentConfig.NUM_ACTIONS)

        # Observation space: continuous vector
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(EnvironmentConfig.STATE_DIM,),
            dtype=np.float32
        )

        # State encoder
        self.encoder = get_encoder()

        # Episode state
        self.debtor: Optional[DebtorPersona] = None
        self.current_turn = 0
        self.conversation_history = []
        self.mentioned_payment_plan = False
        self.mentioned_consequences = False
        self.episode_reward = 0.0

        # Milestone tracking (for one-time rewards)
        self._milestone_shared_situation = False
        self._milestone_feels_understood = False
        self._milestone_discussing_options = False

        # Statistics
        self.episodes_completed = 0
        self.successful_episodes = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment for new episode

        Args:
            seed: Random seed
            options: Optional reset options (can specify persona_type)

        Returns:
            (observation, info) tuple
        """
        super().reset(seed=seed)

        # Create new debtor
        if options and "persona_type" in options:
            persona_type = options["persona_type"]
            self.debtor = DebtorPersona(persona_type)
        else:
            self.debtor = create_random_debtor()

        # Reset episode state
        self.current_turn = 0
        self.conversation_history = []
        self.mentioned_payment_plan = False
        self.mentioned_consequences = False
        self.episode_reward = 0.0

        # Reset milestone tracking
        self._milestone_shared_situation = False
        self._milestone_feels_understood = False
        self._milestone_discussing_options = False

        # Create initial state
        state_dict = self._get_state_dict()
        observation = self._encode_state(state_dict)

        info = {
            "persona": self.debtor.persona_type,
            "initial_sentiment": self.debtor.sentiment,
            "initial_cooperation": self.debtor.cooperation
        }

        if self.render_mode == "human":
            print(f"\n{'='*70}")
            print(f"NEW CONVERSATION - Debtor Persona: {self.debtor.persona_type.upper()}")
            print(f"{'='*70}\n")

        return observation, info

    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step of the environment

        Args:
            action: Action index (0 to NUM_ACTIONS-1)

        Returns:
            (observation, reward, terminated, truncated, info) tuple
        """
        if self.debtor is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action_name = EnvironmentConfig.ACTIONS[action]
        self.current_turn += 1

        # Update flags based on action
        if action_name == "offer_payment_plan":
            self.mentioned_payment_plan = True
        elif action_name in ["firm_reminder", "hard_close"]:
            self.mentioned_consequences = True

        # Get action effectiveness for this persona
        effectiveness = get_action_effectiveness(
            self.debtor.persona_type,
            action_name
        )

        # Generate agent utterance (if LLM client available)
        if self.llm_client:
            agent_utterance = self._generate_agent_utterance(action_name)
        else:
            agent_utterance = f"[{action_name}]"  # Placeholder if no LLM

        # Update debtor state based on action
        self.debtor.update_from_interaction(action_name, effectiveness)

        # Generate debtor response (if LLM client available)
        if self.llm_client:
            debtor_response = self._generate_debtor_response(agent_utterance)
        else:
            debtor_response = f"[Response to {action_name}]"

        # Update conversation history
        self.conversation_history.append({
            "turn": self.current_turn,
            "action": action_name,
            "agent_utterance": agent_utterance,
            "debtor_response": debtor_response
        })

        # Check for payment commitment
        self.debtor.check_commitment()

        # Check termination conditions (before reward calculation)
        terminated = False
        truncated = False

        if self.debtor.has_committed_to_pay:
            terminated = True  # Success!
        elif self.debtor.should_quit():
            terminated = True  # Debtor quit
        elif self.current_turn >= EnvironmentConfig.MAX_TURNS:
            truncated = True  # Hit turn limit

        # Calculate reward (now with termination info for failure penalties)
        reward = self._calculate_reward(action_name, terminated, truncated)
        self.episode_reward += reward

        # Get next state
        next_state_dict = self._get_state_dict()
        observation = self._encode_state(next_state_dict)

        # Info
        info = {
            "action_name": action_name,
            "effectiveness": effectiveness,
            "sentiment": self.debtor.sentiment,
            "cooperation": self.debtor.cooperation,
            "has_committed": self.debtor.has_committed_to_pay,
            "turn": self.current_turn,
            "episode_reward": self.episode_reward
        }

        # Render if requested
        if self.render_mode == "human":
            self._render_turn(action_name, agent_utterance, debtor_response, reward)

        # Update statistics if episode done
        if terminated or truncated:
            self.episodes_completed += 1
            if self.debtor.has_committed_to_pay:
                self.successful_episodes += 1

            if self.render_mode == "human":
                self._render_episode_end(terminated, truncated)

        return observation, reward, terminated, truncated, info

    def _get_state_dict(self) -> Dict:
        """Get current state as dictionary"""
        debtor_state = self.debtor.to_dict()

        # Get last action
        last_action = None
        if len(self.conversation_history) > 0:
            last_turn = self.conversation_history[-1]
            last_action = list(EnvironmentConfig.ACTIONS.keys())[
                list(EnvironmentConfig.ACTIONS.values()).index(last_turn["action"])
            ]

        # Get conversation summary (last 2 turns)
        summary = ""
        if len(self.conversation_history) > 0:
            recent_turns = self.conversation_history[-2:]
            summary = "\n".join([
                f"Agent: {turn['agent_utterance']}\nDebtor: {turn['debtor_response']}"
                for turn in recent_turns
            ])

        return create_state_dict(
            turn=self.current_turn,
            debtor_state=debtor_state,
            agent_last_action=last_action,
            mentioned_payment_plan=self.mentioned_payment_plan,
            mentioned_consequences=self.mentioned_consequences,
            conversation_summary=summary
        )

    def _encode_state(self, state_dict: Dict) -> np.ndarray:
        """Encode state dictionary to numpy array"""
        state_tensor = self.encoder.encode(state_dict)
        return state_tensor.numpy()

    def _calculate_reward(self, action_name: str, terminated: bool = False, truncated: bool = False) -> float:
        """
        Calculate reward for current transition

        Args:
            action_name: Name of action taken
            terminated: Whether episode ended (success or quit)
            truncated: Whether episode hit turn limit

        Returns:
            Reward value
        """
        reward = 0.0

        # Get previous state for comparison
        if len(self.debtor.sentiment_history) >= 2:
            prev_sentiment = self.debtor.sentiment_history[-2]
            prev_cooperation = self.debtor.cooperation_history[-2]
        else:
            prev_sentiment = self.debtor.sentiment_history[0]
            prev_cooperation = self.debtor.cooperation_history[0]

        # PRIMARY: Payment commitment
        if self.debtor.has_committed_to_pay and len(self.conversation_history) == self.current_turn:
            # Just committed this turn
            reward += EnvironmentConfig.REWARD_COMMITMENT

        # ============================================================
        # MILESTONE REWARDS (one-time bonuses for progress indicators)
        # Step 5 of CRITICAL_FIXES - densify sparse reward signal
        # ============================================================
        
        # Milestone: Debtor shared their situation (+1.0)
        if self.debtor.has_shared_situation and not self._milestone_shared_situation:
            reward += 1.0
            self._milestone_shared_situation = True
        
        # Milestone: Debtor feels understood (+1.5)
        if self.debtor.feels_understood and not self._milestone_feels_understood:
            reward += 1.5
            self._milestone_feels_understood = True
        
        # Milestone: Discussing payment options (+2.0)
        # Triggered when agent has offered plan AND debtor cooperation is rising
        if self.mentioned_payment_plan and self.debtor.cooperation > 0.5 and not self._milestone_discussing_options:
            reward += 2.0
            self._milestone_discussing_options = True

        # ============================================================
        # FAILURE PENALTIES (discourage bad outcomes)
        # ============================================================
        
        # Penalty: Episode ended without commitment
        if (terminated or truncated) and not self.debtor.has_committed_to_pay:
            if self.debtor.should_quit():
                # Debtor hung up / quit - harsh penalty
                reward -= 5.0
            else:
                # Hit turn limit without success - moderate penalty
                reward -= 3.0

        # ============================================================
        # CONTINUOUS REWARDS (existing)
        # ============================================================

        # SENTIMENT change
        sentiment_change = self.debtor.sentiment - prev_sentiment
        reward += sentiment_change * EnvironmentConfig.REWARD_SENTIMENT_WEIGHT

        # COOPERATION change
        cooperation_change = self.debtor.cooperation - prev_cooperation
        reward += cooperation_change * EnvironmentConfig.REWARD_COOPERATION_WEIGHT

        # ENGAGEMENT bonus
        if self.debtor.engagement > 0.6:
            reward += EnvironmentConfig.REWARD_ENGAGEMENT_BONUS
        elif self.debtor.engagement < 0.3:
            reward -= EnvironmentConfig.REWARD_ENGAGEMENT_BONUS

        # HOSTILITY penalty
        if self.debtor.sentiment < -0.8:
            reward -= EnvironmentConfig.REWARD_HOSTILITY_PENALTY

        # TURN penalty (encourage efficiency)
        reward -= EnvironmentConfig.REWARD_TURN_PENALTY

        return reward

    def _generate_agent_utterance(self, action_name: str) -> str:
        """Generate agent utterance using LLM"""
        if self.llm_client is None:
            return f"[{action_name}]"

        # Get conversation history summary
        history_text = self._get_conversation_history_text()

        # Generate utterance
        utterance = self.llm_client.generate_agent_utterance(
            strategy=action_name,
            conversation_history=history_text,
            turn=self.current_turn
        )

        return utterance

    def _generate_debtor_response(self, agent_utterance: str) -> str:
        """Generate debtor response using LLM"""
        if self.llm_client is None:
            return "[Debtor response]"

        # Get debtor context
        debtor_context = self.debtor.get_prompt_context()

        # Get conversation history
        history_text = self._get_conversation_history_text()

        # Generate response
        response_data = self.llm_client.generate_debtor_response(
            debtor_context=debtor_context,
            agent_utterance=agent_utterance,
            conversation_history=history_text
        )

        # NOTE: shared_situation and feels_understood are now updated deterministically
        # in debtor.update_from_interaction() to decouple reward from LLM randomness.
        # LLM response is only used for the text response itself.

        return response_data.get("response", "[Error generating response]")

    def _get_conversation_history_text(self) -> str:
        """Get formatted conversation history for LLM prompts"""
        if not self.conversation_history:
            return "(First turn of conversation)"

        lines = []
        for turn in self.conversation_history:
            lines.append(f"Turn {turn['turn']} - Agent: {turn['agent_utterance']}")
            lines.append(f"Turn {turn['turn']} - Debtor: {turn['debtor_response']}")

        return "\n".join(lines)

    def _render_turn(self, action_name: str, agent_utterance: str, debtor_response: str, reward: float):
        """Render single turn to console"""
        print(f"--- Turn {self.current_turn} ---")
        print(f"Strategy: {action_name}")
        print(f"Agent: {agent_utterance}")
        print(f"Debtor: {debtor_response}")
        print(f"Reward: {reward:.2f} | Sentiment: {self.debtor.sentiment:.2f} | Cooperation: {self.debtor.cooperation:.2f}")
        print()

    def _render_episode_end(self, terminated: bool, truncated: bool):
        """Render episode summary"""
        print(f"\n{'='*70}")
        if self.debtor.has_committed_to_pay:
            print("✓ SUCCESS - Debtor committed to payment!")
        elif truncated:
            print("⚠ TRUNCATED - Reached maximum turns")
        else:
            print("✗ FAILED - Debtor quit conversation")

        print(f"\nEpisode Stats:")
        print(f"  Turns: {self.current_turn}")
        print(f"  Total Reward: {self.episode_reward:.2f}")
        print(f"  Final Sentiment: {self.debtor.sentiment:.2f}")
        print(f"  Final Cooperation: {self.debtor.cooperation:.2f}")
        print(f"  Success Rate: {self.successful_episodes}/{self.episodes_completed} ({100*self.successful_episodes/max(1,self.episodes_completed):.1f}%)")
        print(f"{'='*70}\n")

    def get_success_rate(self) -> float:
        """Get current success rate"""
        if self.episodes_completed == 0:
            return 0.0
        return self.successful_episodes / self.episodes_completed

    def close(self):
        """Clean up environment"""
        pass
