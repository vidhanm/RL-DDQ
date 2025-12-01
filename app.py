"""
Web Interface for Debt Collection Agent Demo
Interactive Gradio app for demonstrating trained DQN/DDQ agents

Usage:
    python app.py                    # Launch locally
    python app.py --share             # Launch with public URL
    python app.py --model ddq         # Use DDQ model
"""

import os
import sys
import argparse
import numpy as np
import torch
import gradio as gr
from typing import Optional, Tuple, List, Dict
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.debtor_env import DebtCollectionEnv
from environment.debtor_persona import DebtorPersona, create_random_debtor
from agent.dqn_agent import DQNAgent
from agent.ddq_agent import DDQAgent
from llm.nvidia_client import NVIDIAClient
from config import EnvironmentConfig, DeviceConfig

# Action names for display
ACTION_NAMES = [
    "😌 Empathetic Listening",
    "❓ Ask About Situation",
    "📋 Firm Reminder",
    "💳 Offer Payment Plan",
    "🤝 Propose Settlement",
    "⚠️ Hard Close"
]

ACTION_DESCRIPTIONS = {
    0: "Show understanding and empathy for the debtor's situation",
    1: "Ask questions to understand their circumstances better",
    2: "Remind them firmly about the debt and consequences",
    3: "Offer a structured payment plan option",
    4: "Propose a settlement for reduced amount",
    5: "Final push with urgency and consequences"
}

PERSONA_EMOJIS = {
    "angry": "😠",
    "cooperative": "😊",
    "sad": "😢",
    "avoidant": "😶"
}


class DemoApp:
    """Main demo application"""
    
    def __init__(self, model_type: str = "dqn", checkpoint_dir: str = "checkpoints"):
        self.model_type = model_type
        self.checkpoint_dir = checkpoint_dir
        self.agent = None
        self.env = None
        self.llm_client = None
        self.current_state = None
        self.conversation_history = []
        self.is_episode_done = False
        self.current_persona = None
        
        # Try to initialize LLM client
        try:
            self.llm_client = NVIDIAClient()
            print("[OK] LLM client initialized")
        except Exception as e:
            print(f"[WARN] LLM client failed: {e}")
            print("  Demo will run without actual LLM responses")
    
    def load_model(self, model_type: str) -> str:
        """Load a trained model"""
        self.model_type = model_type.lower()
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_final.pt")
        
        if not os.path.exists(checkpoint_path):
            # Try episode checkpoint
            checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.model_type}_episode_100.pt")
        
        if not os.path.exists(checkpoint_path):
            return f"❌ No checkpoint found for {model_type.upper()}"
        
        try:
            if self.model_type == "ddq":
                self.agent = DDQAgent(
                    state_dim=EnvironmentConfig.STATE_DIM,
                    action_dim=EnvironmentConfig.NUM_ACTIONS,
                    device=DeviceConfig.DEVICE
                )
            else:
                self.agent = DQNAgent(
                    state_dim=EnvironmentConfig.STATE_DIM,
                    action_dim=EnvironmentConfig.NUM_ACTIONS,
                    device=DeviceConfig.DEVICE
                )
            
            self.agent.load(checkpoint_path)
            return f"✅ Loaded {model_type.upper()} model from {os.path.basename(checkpoint_path)}"
        except Exception as e:
            return f"❌ Failed to load model: {e}"
    
    def start_conversation(self, persona_choice: str) -> Tuple[str, str, str, str]:
        """Start a new conversation with selected persona"""
        # Create environment
        self.env = DebtCollectionEnv(llm_client=self.llm_client, render_mode=None)
        
        # Set persona
        if persona_choice == "Random":
            self.current_state, info = self.env.reset()
            self.current_persona = self.env.debtor.persona_type
        else:
            persona_type = persona_choice.lower()
            self.current_state, info = self.env.reset(options={"persona": persona_type})
            self.current_persona = persona_type
        
        self.conversation_history = []
        self.is_episode_done = False
        
        # Get initial state display
        state_display = self._format_state()
        q_values_display = self._format_q_values()
        
        emoji = PERSONA_EMOJIS.get(self.current_persona, "👤")
        status = f"🎬 New conversation started with {emoji} {self.current_persona.title()} debtor"
        
        return status, state_display, q_values_display, ""
    
    def take_action(self, action_choice: str) -> Tuple[List, str, str, str]:
        """Agent takes an action, get debtor response"""
        if self.env is None or self.agent is None:
            return self.conversation_history, "❌ Please load a model and start a conversation first", "", ""
        
        if self.is_episode_done:
            return self.conversation_history, "⏹️ Episode ended. Start a new conversation.", self._format_state(), self._format_q_values()
        
        # Parse action from choice string
        action = None
        for i, name in enumerate(ACTION_NAMES):
            if name in action_choice:
                action = i
                break
        
        if action is None:
            # Use agent's recommended action
            action = self.agent.select_action(self.current_state, explore=False)
        
        # Take step in environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Update conversation history for display
        if self.env.conversation_history:
            last_turn = self.env.conversation_history[-1]
            agent_msg = last_turn.get('agent_utterance', f"[Action: {ACTION_NAMES[action]}]")
            debtor_msg = last_turn.get('debtor_response', "[No response]")
            
            self.conversation_history.append((agent_msg, debtor_msg))
        
        # Update state
        self.current_state = next_state
        self.is_episode_done = done
        
        # Format displays
        state_display = self._format_state()
        q_values_display = self._format_q_values()
        
        # Status message
        if done:
            if info.get('has_committed', False):
                status = f"✅ SUCCESS! Debtor committed to payment. Reward: {reward:.2f}"
            else:
                status = f"❌ Episode ended without commitment. Reward: {reward:.2f}"
        else:
            status = f"Turn {self.env.current_turn}/{EnvironmentConfig.MAX_TURNS} | Reward: {reward:.2f}"
        
        return self.conversation_history, status, state_display, q_values_display
    
    def auto_play(self) -> Tuple[List, str, str, str]:
        """Let agent play automatically until episode ends"""
        if self.env is None or self.agent is None:
            return [], "❌ Please load a model and start a conversation first", "", ""
        
        while not self.is_episode_done:
            # Get agent's action
            action = self.agent.select_action(self.current_state, explore=False)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Update conversation
            if self.env.conversation_history:
                last_turn = self.env.conversation_history[-1]
                agent_msg = last_turn.get('agent_utterance', f"[Action: {ACTION_NAMES[action]}]")
                debtor_msg = last_turn.get('debtor_response', "[No response]")
                self.conversation_history.append((agent_msg, debtor_msg))
            
            self.current_state = next_state
            self.is_episode_done = done
        
        # Final status
        if info.get('has_committed', False):
            status = f"✅ SUCCESS! Debtor committed to payment after {self.env.current_turn} turns"
        else:
            status = f"❌ Failed to get commitment after {self.env.current_turn} turns"
        
        return self.conversation_history, status, self._format_state(), self._format_q_values()
    
    def _format_state(self) -> str:
        """Format current state for display"""
        if self.current_state is None or self.env is None:
            return "No active conversation"
        
        state = self.env.state
        lines = [
            "### 📊 Current State",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Turn | {state.get('turn', 0)}/{EnvironmentConfig.MAX_TURNS} |",
            f"| Sentiment | {state.get('debtor_sentiment', 0):.2f} |",
            f"| Cooperation | {state.get('debtor_cooperation', 0):.2f} |",
            f"| Engagement | {state.get('debtor_engagement', 0):.2f} |",
            f"| Mentioned Payment Plan | {'✓' if state.get('mentioned_payment_plan', False) else '✗'} |",
            f"| Shared Situation | {'✓' if state.get('debtor_shared_situation', False) else '✗'} |",
            f"| Has Committed | {'✅' if state.get('has_committed', False) else '❌'} |",
        ]
        return "\n".join(lines)
    
    def _format_q_values(self) -> str:
        """Format Q-values for display"""
        if self.agent is None or self.current_state is None:
            return "No model loaded"
        
        # Get Q-values from agent
        state_tensor = torch.FloatTensor(self.current_state).to(self.agent.device)
        with torch.no_grad():
            q_values = self.agent.policy_net(state_tensor).cpu().numpy()
        
        best_action = np.argmax(q_values)
        
        lines = [
            "### 🎯 Q-Values (Agent's Evaluation)",
            "",
            "| Action | Q-Value | |",
            "|--------|---------|---|",
        ]
        
        for i, (name, q) in enumerate(zip(ACTION_NAMES, q_values)):
            marker = "👈 **BEST**" if i == best_action else ""
            bar_len = int(max(0, (q + 5) * 3))  # Scale for visualization
            bar = "█" * bar_len
            lines.append(f"| {name} | {q:.3f} | {bar} {marker} |")
        
        return "\n".join(lines)


def create_demo_interface(app: DemoApp) -> gr.Blocks:
    """Create the Gradio interface"""
    
    with gr.Blocks(
        title="DDQ Debt Collection Agent Demo",
        theme=gr.themes.Soft(),
        css="""
        .container { max-width: 1200px; margin: auto; }
        .chatbot { min-height: 400px; }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🤖 DDQ Debt Collection Agent Demo
        
        Interactive demonstration of a Reinforcement Learning agent trained to handle debt collection conversations.
        The agent uses **DDQ (DQN with World Model)** for sample-efficient learning.
        
        ---
        """)
        
        with gr.Tabs():
            # Tab 1: Live Conversation
            with gr.Tab("💬 Live Conversation"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Controls
                        with gr.Row():
                            model_dropdown = gr.Dropdown(
                                choices=["DQN", "DDQ"],
                                value="DQN",
                                label="Select Model"
                            )
                            load_btn = gr.Button("📥 Load Model", variant="primary")
                            load_status = gr.Textbox(label="Status", interactive=False)
                        
                        with gr.Row():
                            persona_dropdown = gr.Dropdown(
                                choices=["Random", "Angry", "Cooperative", "Sad", "Avoidant"],
                                value="Random",
                                label="Debtor Persona"
                            )
                            start_btn = gr.Button("🎬 Start Conversation", variant="primary")
                        
                        # Chat display
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=400,
                            show_label=True
                        )
                        
                        # Action buttons
                        with gr.Row():
                            action_dropdown = gr.Dropdown(
                                choices=ACTION_NAMES,
                                value=ACTION_NAMES[0],
                                label="Select Action (or let agent choose)"
                            )
                            action_btn = gr.Button("▶️ Take Action")
                            auto_btn = gr.Button("🤖 Auto-Play", variant="secondary")
                        
                        episode_status = gr.Textbox(label="Episode Status", interactive=False)
                    
                    with gr.Column(scale=1):
                        # State and Q-values display
                        state_display = gr.Markdown(value="### 📊 Current State\n\nStart a conversation to see state")
                        q_values_display = gr.Markdown(value="### 🎯 Q-Values\n\nLoad a model to see Q-values")
                
                # Event handlers
                load_btn.click(
                    fn=app.load_model,
                    inputs=[model_dropdown],
                    outputs=[load_status]
                )
                
                start_btn.click(
                    fn=app.start_conversation,
                    inputs=[persona_dropdown],
                    outputs=[episode_status, state_display, q_values_display, chatbot]
                ).then(
                    fn=lambda: [],
                    outputs=[chatbot]
                )
                
                action_btn.click(
                    fn=app.take_action,
                    inputs=[action_dropdown],
                    outputs=[chatbot, episode_status, state_display, q_values_display]
                )
                
                auto_btn.click(
                    fn=app.auto_play,
                    outputs=[chatbot, episode_status, state_display, q_values_display]
                )
            
            # Tab 2: Training Analysis
            with gr.Tab("📈 Training Analysis"):
                gr.Markdown("""
                ### Training Visualizations
                
                View the training progress and comparison between DQN and DDQ.
                """)
                
                with gr.Row():
                    if os.path.exists("figures/overview_all_runs.png"):
                        gr.Image("figures/overview_all_runs.png", label="Training Overview")
                    else:
                        gr.Markdown("*Run `python visualize.py` to generate training plots*")
                
                # Show available training history
                gr.Markdown("### Available Training Runs")
                
                history_files = []
                if os.path.exists("checkpoints"):
                    history_files = [f for f in os.listdir("checkpoints") if f.endswith(".json")]
                
                if history_files:
                    history_md = "\n".join([f"- {f}" for f in history_files])
                    gr.Markdown(history_md)
                else:
                    gr.Markdown("*No training history files found*")
            
            # Tab 3: About
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""
                ## About This Project
                
                This demo showcases a **Reinforcement Learning agent** trained to handle debt collection 
                conversations using the **DDQ (Dyna-Q with Deep Q-Network)** algorithm.
                
                ### 🎯 Key Features
                
                - **DQN Baseline**: Standard Deep Q-Network for comparison
                - **DDQ with World Model**: Sample-efficient learning using imagined experiences
                - **LLM-based Debtor Simulation**: Realistic conversation with GPT-powered personas
                - **Multiple Debtor Personas**: Angry, Cooperative, Sad, and Avoidant types
                
                ### 🔧 How It Works
                
                1. **Agent** selects a high-level strategy (action)
                2. **LLM** generates natural language utterance based on strategy
                3. **Debtor LLM** responds based on persona and conversation history
                4. **Environment** updates state and calculates reward
                5. **Agent** learns from experiences (real + imagined with world model)
                
                ### 📊 Action Space
                
                | Action | Description |
                |--------|-------------|
                | Empathetic Listening | Show understanding and empathy |
                | Ask About Situation | Gather information about circumstances |
                | Firm Reminder | Remind about debt and consequences |
                | Offer Payment Plan | Propose structured payment option |
                | Propose Settlement | Offer reduced settlement amount |
                | Hard Close | Final push with urgency |
                
                ### 🏆 Success Criteria
                
                The agent succeeds when the debtor commits to a payment plan or settlement.
                
                ---
                
                **Built with**: PyTorch, Gymnasium, Gradio, NVIDIA NIM API
                """)
        
        gr.Markdown("""
        ---
        *DDQ Debt Collection Agent Demo | Built for demonstrating RL in conversational AI*
        """)
    
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch Debt Collection Agent Demo")
    parser.add_argument('--model', type=str, default='dqn', choices=['dqn', 'ddq'],
                        help='Default model to load')
    parser.add_argument('--share', action='store_true',
                        help='Create public shareable link')
    parser.add_argument('--port', type=int, default=7860,
                        help='Port to run on')
    
    args = parser.parse_args()
    
    print("="*60)
    print("DDQ Debt Collection Agent Demo")
    print("="*60)
    
    # Create app
    app = DemoApp(model_type=args.model)
    
    # Load default model
    status = app.load_model(args.model)
    print(status)
    
    # Create and launch interface
    demo = create_demo_interface(app)
    
    print(f"\nLaunching demo on port {args.port}...")
    if args.share:
        print("Creating public shareable link...")
    
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()
