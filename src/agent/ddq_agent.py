"""
DDQ Agent (Dyna-style Data-efficient Q-learning)
Extends DQN with world model and imagination
"""

import torch
import numpy as np
from typing import Tuple, List, Optional

from .dqn_agent import DQNAgent
from .world_model import create_world_model, WorldModelTrainer
from .advanced_world_models import EnhancedEnsembleWorldModel
from src.config import DDQConfig, EnvironmentConfig, DeviceConfig


class DDQAgent(DQNAgent):
    """DDQ Agent with world model and imagination"""

    def __init__(
        self,
        state_dim: int = EnvironmentConfig.STATE_DIM,
        action_dim: int = EnvironmentConfig.NUM_ACTIONS,
        K: int = DDQConfig.K,
        real_ratio: float = DDQConfig.REAL_RATIO,
        world_model_lr: float = DDQConfig.WORLD_MODEL_LEARNING_RATE,
        world_model_epochs: int = DDQConfig.WORLD_MODEL_EPOCHS,
        imagination_horizon: int = DDQConfig.IMAGINATION_HORIZON,
        use_ensemble: bool = True,
        uncertainty_threshold: float = 0.5,
        num_ensemble_models: int = 5,
        **kwargs
    ):
        """
        Initialize DDQ Agent

        Args:
            state_dim: State dimension
            action_dim: Number of actions
            K: Imagination factor (synthetic experiences per real experience)
            real_ratio: Ratio of real to imagined experiences in training
            world_model_lr: World model learning rate
            world_model_epochs: Epochs for world model training
            imagination_horizon: Steps to imagine forward
            use_ensemble: If True, use ensemble world model with uncertainty filtering
            uncertainty_threshold: Discard imagined samples with disagreement > threshold
            num_ensemble_models: Number of models in ensemble
            **kwargs: Additional args for DQN agent
        """
        # Initialize DQN base
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)

        # DDQ hyperparameters
        self.K = K
        self.real_ratio = real_ratio
        self.world_model_epochs = world_model_epochs
        self.imagination_horizon = imagination_horizon
        self.use_ensemble = use_ensemble
        self.uncertainty_threshold = uncertainty_threshold

        # Create world model (ensemble or single)
        if use_ensemble:
            self.world_model = EnhancedEnsembleWorldModel(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=DDQConfig.WORLD_MODEL_HIDDEN_DIM,
                num_models=num_ensemble_models,
                device=self.device
            )
            self.world_model_trainer = None  # Ensemble has its own train_step
        else:
            self.world_model = create_world_model(
                state_dim=state_dim,
                action_dim=action_dim,
                device=self.device
            )
            self.world_model_trainer = WorldModelTrainer(
                world_model=self.world_model,
                learning_rate=world_model_lr,
                device=self.device
            )

        # Imagined experience buffer (temporary storage)
        self.imagined_experiences = []

        # Statistics
        self.world_model_training_steps = 0
        self.total_imagined_experiences = 0
        self.filtered_experiences = 0  # Experiences discarded due to high uncertainty

    def train_world_model(self) -> dict:
        """
        Train world model on replay buffer

        Returns:
            Training statistics
        """
        if not self.replay_buffer.is_ready(DDQConfig.MIN_WORLD_MODEL_BUFFER):
            return {'trained': False}

        if self.use_ensemble:
            # Train ensemble directly
            stats = self._train_ensemble_on_buffer()
        else:
            # Train single world model
            stats = self.world_model_trainer.train_on_buffer(
                replay_buffer=self.replay_buffer,
                batch_size=self.batch_size,
                num_epochs=self.world_model_epochs
            )

        self.world_model_training_steps += stats.get('num_batches', 1)
        stats['trained'] = True

        return stats

    def _train_ensemble_on_buffer(self) -> dict:
        """Train ensemble world model on replay buffer"""
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.world_model_epochs):
            # Sample batch (handle PER vs uniform buffer)
            if self.use_prioritized_replay:
                states, actions, rewards, next_states, dones, _, _ = \
                    self.replay_buffer.sample(self.batch_size, beta=self.per_beta)
            else:
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample(self.batch_size)
            
            # Convert to tensors (handle if already tensors)
            if isinstance(states, torch.Tensor):
                states_t = states.to(self.device)
                next_states_t = next_states.to(self.device)
                rewards_t = rewards.to(self.device)
                actions_np = actions.cpu().numpy()
            else:
                states_t = torch.FloatTensor(states).to(self.device)
                next_states_t = torch.FloatTensor(next_states).to(self.device)
                rewards_t = torch.FloatTensor(rewards).to(self.device)
                actions_np = actions
            
            # One-hot encode actions
            actions_t = torch.zeros(len(actions_np), self.action_dim).to(self.device)
            for i, a in enumerate(actions_np):
                actions_t[i, int(a)] = 1.0
            
            # Train step
            stats = self.world_model.train_step(states_t, actions_t, next_states_t, rewards_t)
            total_loss += stats['total_loss']
            num_batches += 1
        
        return {
            'total_loss': total_loss / max(num_batches, 1),
            'num_batches': num_batches
        }

    def generate_imagined_experiences(self) -> int:
        """
        Generate imagined experiences using world model

        Returns:
            Number of imagined experiences generated
        """
        if not self.replay_buffer.is_ready(DDQConfig.MIN_WORLD_MODEL_BUFFER):
            return 0

        self.imagined_experiences = []

        # Sample real states to start imagination from
        num_real_experiences = min(len(self.replay_buffer), self.batch_size)
        start_states = self.replay_buffer.sample_states(num_real_experiences)

        # Generate K imagined experiences per real state
        for _ in range(self.K):
            for start_state in start_states:
                # Imagine from this state
                imagined_trajectory = self._imagine_trajectory(
                    start_state,
                    horizon=self.imagination_horizon
                )

                # Add to imagined buffer
                self.imagined_experiences.extend(imagined_trajectory)

        self.total_imagined_experiences += len(self.imagined_experiences)

        return len(self.imagined_experiences)

    def _imagine_trajectory(self, start_state: torch.Tensor, horizon: int) -> List[Tuple]:
        """
        Imagine a trajectory starting from state

        Args:
            start_state: Starting state [state_dim]
            horizon: Number of steps to imagine

        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        trajectory = []
        current_state = start_state.to(self.device)

        for _ in range(horizon):
            # Use policy-based action selection (not random!)
            # This generates more relevant trajectories the agent might actually take
            action_idx = self._select_imagination_action(current_state)

            # Convert action to one-hot
            action_one_hot = torch.zeros(self.action_dim).to(self.device)
            action_one_hot[action_idx] = 1.0

            # Predict next state and reward using world model
            if self.use_ensemble:
                # Use ensemble with uncertainty
                next_state, reward, disagreement = self.world_model.predict_with_uncertainty(
                    current_state.unsqueeze(0),
                    action_one_hot.unsqueeze(0)
                )
                disagreement = disagreement.item()
                
                # Filter by uncertainty threshold
                if disagreement > self.uncertainty_threshold:
                    self.filtered_experiences += 1
                    break  # Stop trajectory if uncertain
            else:
                next_state, reward = self.world_model.predict(
                    current_state.unsqueeze(0),
                    action_one_hot.unsqueeze(0)
                )

            # Remove batch dimension
            next_state = next_state.squeeze(0)
            reward = reward.item() if isinstance(reward, torch.Tensor) else reward

            # Store experience (imagined experiences are never "done")
            experience = (
                current_state.cpu().numpy(),
                action_idx,
                reward,
                next_state.cpu().numpy(),
                False  # Imagined experiences don't terminate
            )

            trajectory.append(experience)

            # Continue from predicted next state
            current_state = next_state

        return trajectory

    def _select_imagination_action(self, state: torch.Tensor) -> int:
        """
        Select action for imagination using epsilon-greedy policy
        
        This generates more relevant trajectories than random action selection
        """
        # Use higher epsilon for imagination exploration (more diverse)
        imagination_epsilon = max(0.3, self.epsilon)  # At least 30% random
        
        if np.random.random() < imagination_epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                state_t = state.unsqueeze(0) if state.dim() == 1 else state
                q_values = self.policy_net(state_t)  # Use policy_net (from DQNAgent)
                return q_values.argmax(dim=-1).item()
    
    def select_action_with_lookahead(
        self, 
        state: np.ndarray, 
        lookahead_depth: int = 2,
        discount: float = 0.99
    ) -> int:
        """
        Select action using multi-step look-ahead with world model.
        
        For each possible action, simulate future trajectory and 
        pick action with best cumulative discounted value.
        
        Args:
            state: Current state
            lookahead_depth: Number of future steps to simulate (default 2)
            discount: Discount factor for future rewards
            
        Returns:
            Best action based on look-ahead
        """
        if not self.use_ensemble or not hasattr(self.world_model, 'predict_with_uncertainty'):
            # Fall back to standard selection if no world model
            return self.select_action(state, explore=False)
        
        state_tensor = torch.FloatTensor(state).to(self.device)
        best_action = 0
        best_value = float('-inf')
        
        # Evaluate each possible action
        for action_idx in range(self.action_dim):
            cumulative_value = 0.0
            current_state = state_tensor.clone()
            
            # Simulate trajectory
            for step in range(lookahead_depth):
                # First step uses candidate action, subsequent use greedy policy
                if step == 0:
                    a = action_idx
                else:
                    with torch.no_grad():
                        q_vals = self.policy_net(current_state.unsqueeze(0))
                        a = q_vals.argmax(dim=-1).item()
                
                # One-hot encode action
                action_one_hot = torch.zeros(self.action_dim).to(self.device)
                action_one_hot[a] = 1.0
                
                # Predict next state and reward
                next_state, reward, disagreement = self.world_model.predict_with_uncertainty(
                    current_state.unsqueeze(0),
                    action_one_hot.unsqueeze(0)
                )
                
                # If high uncertainty, penalize this trajectory
                if disagreement.item() > self.uncertainty_threshold:
                    cumulative_value -= 1.0  # Penalty for uncertain predictions
                    break
                
                # Add discounted reward
                reward_val = reward.item() if isinstance(reward, torch.Tensor) else reward
                cumulative_value += (discount ** step) * reward_val
                
                # Get Q-value of next state (terminal value)
                with torch.no_grad():
                    next_q = self.policy_net(next_state).max().item()
                    cumulative_value += (discount ** (step + 1)) * next_q * 0.1  # Small weight
                
                current_state = next_state.squeeze(0)
            
            # Track best action
            if cumulative_value > best_value:
                best_value = cumulative_value
                best_action = action_idx
        
        return best_action

    def train_step(self) -> float:
        """
        DDQ training step: train DQN on mix of real + imagined experiences

        Returns:
            Loss value
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return 0.0

        # Calculate batch sizes
        real_batch_size = int(self.batch_size * self.real_ratio)
        imagined_batch_size = self.batch_size - real_batch_size

        # Sample real experiences (handle PER vs uniform buffer)
        if self.use_prioritized_replay:
            states_real, actions_real, rewards_real, next_states_real, dones_real, weights, indices = \
                self.replay_buffer.sample(real_batch_size, beta=self.per_beta)
            weights = weights.to(self.device)
        else:
            states_real, actions_real, rewards_real, next_states_real, dones_real = \
                self.replay_buffer.sample(real_batch_size)
            weights = None
            indices = None

        # Sample imagined experiences (if available)
        if len(self.imagined_experiences) >= imagined_batch_size:
            imagined_sample = np.random.choice(
                len(self.imagined_experiences),
                size=imagined_batch_size,
                replace=False
            )

            imagined_batch = [self.imagined_experiences[i] for i in imagined_sample]
            states_img, actions_img, rewards_img, next_states_img, dones_img = zip(*imagined_batch)

            # Convert to tensors
            states_img = torch.FloatTensor(np.array(states_img))
            actions_img = torch.LongTensor(actions_img)
            rewards_img = torch.FloatTensor(rewards_img)
            next_states_img = torch.FloatTensor(np.array(next_states_img))
            dones_img = torch.FloatTensor(dones_img)

            # Combine real and imagined
            states = torch.cat([states_real, states_img])
            actions = torch.cat([actions_real, actions_img])
            rewards = torch.cat([rewards_real, rewards_img])
            next_states = torch.cat([next_states_real, next_states_img])
            dones = torch.cat([dones_real, dones_img])
        else:
            # Not enough imagined experiences, use only real
            states = states_real
            actions = actions_real
            rewards = rewards_real
            next_states = next_states_real
            dones = dones_real

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using Double DQN
        with torch.no_grad():
            # Double DQN: Use policy net to SELECT action, target net to EVALUATE
            best_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute TD errors (per-sample)
        td_errors = current_q_values - target_q_values
        losses = self.criterion(current_q_values, target_q_values)

        # Apply importance sampling weights for PER (only for real samples)
        if self.use_prioritized_replay and weights is not None:
            # Only weight the real samples portion of the loss
            real_losses = losses[:len(weights)]
            weighted_real_losses = real_losses * weights
            
            if len(losses) > len(weights):
                # Mix weighted real + unweighted imagined
                imagined_losses = losses[len(weights):]
                loss = (weighted_real_losses.sum() + imagined_losses.sum()) / len(losses)
            else:
                loss = weighted_real_losses.mean()
            
            # Update priorities for real samples
            real_td_errors = td_errors[:len(weights)]
            self.replay_buffer.update_priorities(indices, real_td_errors.abs().detach().cpu().numpy())
        else:
            loss = losses.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update statistics
        self.steps_trained += 1
        self.total_loss += loss.item()

        return loss.item()

    def episode_done(self, episode: int, total_episodes: int = 500):
        """
        Called at end of episode - train world model and generate imagined experiences

        Args:
            episode: Episode number
            total_episodes: Total episodes for annealing schedules
        """
        # Train world model periodically
        if episode % 5 == 0:  # Train world model every 5 episodes
            wm_stats = self.train_world_model()

            if wm_stats.get('trained', False):
                # Generate imagined experiences after world model training
                num_imagined = self.generate_imagined_experiences()

        # Call parent episode_done (updates epsilon, target network, PER beta, etc.)
        super().episode_done(episode, total_episodes)

    def get_statistics(self) -> dict:
        """Get training statistics including DDQ-specific metrics"""
        stats = super().get_statistics()

        # Add DDQ-specific stats
        stats.update({
            'world_model_steps': self.world_model_training_steps,
            'total_imagined_experiences': self.total_imagined_experiences,
            'current_imagined_buffer_size': len(self.imagined_experiences),
            'imagination_factor_K': self.K,
            'real_ratio': self.real_ratio
        })

        # Add world model stats (handle ensemble vs single model)
        if self.world_model_trainer is not None:
            wm_stats = self.world_model_trainer.get_statistics()
            stats.update({
                'world_model_avg_loss': wm_stats.get('avg_total_loss', 0),
                'world_model_state_loss': wm_stats.get('avg_state_loss', 0),
                'world_model_reward_loss': wm_stats.get('avg_reward_loss', 0),
            })
        else:
            # Ensemble model - get stats from world model directly if available
            stats.update({
                'world_model_avg_loss': 0,
                'world_model_state_loss': 0,
                'world_model_reward_loss': 0,
            })

        return stats

    def print_statistics(self):
        """Print training statistics"""
        print(f"\nDDQ Agent Statistics:")
        stats = self.get_statistics()
        print(f"  Episodes trained: {stats['episodes_trained']}")
        print(f"  DQN steps trained: {stats['steps_trained']}")
        print(f"  World model steps: {stats['world_model_steps']}")
        print(f"  Total imagined experiences: {stats['total_imagined_experiences']}")
        print(f"  Current imagined buffer: {stats['current_imagined_buffer_size']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  DQN avg loss: {stats['avg_loss']:.4f}")
        print(f"  World model avg loss: {stats['world_model_avg_loss']:.4f}")
        print(f"    - State loss: {stats['world_model_state_loss']:.4f}")
        print(f"    - Reward loss: {stats['world_model_reward_loss']:.4f}")

    def save(self, filepath: str):
        """Save agent including world model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'world_model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episodes_trained': self.episodes_trained,
            'steps_trained': self.steps_trained,
            'world_model_training_steps': self.world_model_training_steps,
            'total_imagined_experiences': self.total_imagined_experiences,
        }, filepath)
        print(f"DDQ Agent saved to {filepath}")

    def load(self, filepath: str):
        """Load agent including world model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episodes_trained = checkpoint['episodes_trained']
        self.steps_trained = checkpoint['steps_trained']
        self.world_model_training_steps = checkpoint.get('world_model_training_steps', 0)
        self.total_imagined_experiences = checkpoint.get('total_imagined_experiences', 0)
        print(f"DDQ Agent loaded from {filepath}")
