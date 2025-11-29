"""
DDQ Agent (Dyna-style Data-efficient Q-learning)
Extends DQN with world model and imagination
"""

import torch
import numpy as np
from typing import Tuple, List

from agent.dqn_agent import DQNAgent
from agent.world_model import create_world_model, WorldModelTrainer
from config import DDQConfig, EnvironmentConfig, DeviceConfig


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
            **kwargs: Additional args for DQN agent
        """
        # Initialize DQN base
        super().__init__(state_dim=state_dim, action_dim=action_dim, **kwargs)

        # DDQ hyperparameters
        self.K = K
        self.real_ratio = real_ratio
        self.world_model_epochs = world_model_epochs
        self.imagination_horizon = imagination_horizon

        # Create world model
        self.world_model = create_world_model(
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )

        # World model trainer
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

    def train_world_model(self) -> dict:
        """
        Train world model on replay buffer

        Returns:
            Training statistics
        """
        if not self.replay_buffer.is_ready(DDQConfig.MIN_WORLD_MODEL_BUFFER):
            return {'trained': False}

        # Train world model on replay buffer
        stats = self.world_model_trainer.train_on_buffer(
            replay_buffer=self.replay_buffer,
            batch_size=self.batch_size,
            num_epochs=self.world_model_epochs
        )

        self.world_model_training_steps += stats['num_batches']
        stats['trained'] = True

        return stats

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
            # Sample random action (exploration in imagination)
            action_idx = torch.randint(0, self.action_dim, (1,)).item()

            # Convert action to one-hot
            action_one_hot = torch.zeros(self.action_dim).to(self.device)
            action_one_hot[action_idx] = 1.0

            # Predict next state and reward using world model
            next_state, reward = self.world_model.predict(
                current_state.unsqueeze(0),
                action_one_hot.unsqueeze(0)
            )

            # Remove batch dimension
            next_state = next_state.squeeze(0)
            reward = reward.item()

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

        # Sample real experiences
        states_real, actions_real, rewards_real, next_states_real, dones_real = \
            self.replay_buffer.sample(real_batch_size)

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

        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update statistics
        self.steps_trained += 1
        self.total_loss += loss.item()

        return loss.item()

    def episode_done(self, episode: int):
        """
        Called at end of episode - train world model and generate imagined experiences

        Args:
            episode: Episode number
        """
        # Train world model periodically
        if episode % 5 == 0:  # Train world model every 5 episodes
            wm_stats = self.train_world_model()

            if wm_stats.get('trained', False):
                # Generate imagined experiences after world model training
                num_imagined = self.generate_imagined_experiences()

        # Call parent episode_done (updates epsilon, target network, etc.)
        super().episode_done(episode)

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

        # Add world model stats
        wm_stats = self.world_model_trainer.get_statistics()
        stats.update({
            'world_model_avg_loss': wm_stats.get('avg_total_loss', 0),
            'world_model_state_loss': wm_stats.get('avg_state_loss', 0),
            'world_model_reward_loss': wm_stats.get('avg_reward_loss', 0),
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
