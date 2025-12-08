"""
Self-Play Router
WebSocket endpoint for adversarial self-play training with live updates
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import os
import json
import asyncio
import threading
from datetime import datetime
from typing import Optional, Dict, Any

import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

from web.backend.dependencies import get_llm_client

from src.environment.selfplay_env import SelfPlayEnv
from src.agent.ddq_agent import DDQAgent
from src.agent.adversarial_agent import AdversarialDebtorAgent, create_adversarial_agent
from src.agent.opponent_pool import DualPoolManager
from src.config import EnvironmentConfig, SelfPlayConfig, RLConfig

router = APIRouter(prefix="/api/selfplay", tags=["selfplay"])

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", "selfplay")


# ============== Self-Play Training State ==============

class SelfPlayManager:
    """Manages self-play training state and WebSocket connections"""
    
    def __init__(self):
        self.is_training = False
        self.should_stop = False
        self.current_generation = 0
        self.total_generations = 0
        self.current_episode = 0
        self.episodes_per_gen = 50
        self.collector_wins = 0
        self.adversary_wins = 0
        self.websocket: Optional[WebSocket] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
    
    def get_status(self) -> Dict[str, Any]:
        total = self.collector_wins + self.adversary_wins
        return {
            "is_training": self.is_training,
            "current_generation": self.current_generation,
            "total_generations": self.total_generations,
            "current_episode": self.current_episode,
            "collector_win_rate": self.collector_wins / total if total > 0 else 0,
            "adversary_win_rate": self.adversary_wins / total if total > 0 else 0,
        }
    
    async def send_message(self, message: dict):
        """Send message to WebSocket client"""
        if self.websocket:
            try:
                await self.websocket.send_json(message)
            except Exception as e:
                print(f"WebSocket send error: {e}")
    
    def send_message_sync(self, message: dict):
        """Send message from sync context"""
        if self.websocket and self.loop:
            asyncio.run_coroutine_threadsafe(
                self.send_message(message), 
                self.loop
            )


selfplay_manager = SelfPlayManager()


@router.get("/status")
async def get_selfplay_status():
    """Get current self-play training status"""
    return selfplay_manager.get_status()


@router.post("/stop")
async def stop_selfplay():
    """Request to stop self-play training"""
    if not selfplay_manager.is_training:
        raise HTTPException(status_code=400, detail="No training in progress")
    
    selfplay_manager.should_stop = True
    return {"message": "Stop requested"}


def run_selfplay_training(
    generations: int,
    episodes_per_gen: int,
    use_llm: bool,
    zero_sum: bool
):
    """Run self-play training in background thread"""
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        
        print("\n" + "="*70)
        print("🎮 SELF-PLAY TRAINING STARTED")
        print(f"   Generations: {generations}")
        print(f"   Episodes/Gen: {episodes_per_gen}")
        print(f"   Use LLM: {use_llm}")
        print(f"   Zero Sum: {zero_sum}")
        print("="*70 + "\n")
        
        # Get LLM client if needed
        llm_client = None
        if use_llm:
            try:
                llm_client = get_llm_client()
                print("✓ LLM client initialized")
            except Exception as e:
                print(f"⚠ LLM client failed: {e}")
        
        # Create environment
        env = SelfPlayEnv(
            llm_client=llm_client,
            render_mode=None,
            use_llm_for_collector=use_llm,
            use_llm_for_adversary=use_llm
        )
        print("✓ Self-play environment created")
        
        # Create agents
        collector = DDQAgent(
            state_dim=EnvironmentConfig.NLU_STATE_DIM,
            action_dim=EnvironmentConfig.NUM_ACTIONS,
        )
        print("✓ Collector agent created")
        
        adversary = create_adversarial_agent()
        print("✓ Adversary agent created")
        
        # Initialize pool manager
        pool_manager = DualPoolManager(
            pool_dir=os.path.join(CHECKPOINT_DIR, "opponent_pool"),
            max_size=SelfPlayConfig.OPPONENT_POOL_SIZE
        )
        print("✓ Pool manager initialized\n")
        
        selfplay_manager.collector_wins = 0
        selfplay_manager.adversary_wins = 0
        
        # Training loop
        for gen in range(generations):
            if selfplay_manager.should_stop:
                print("\n⚠ Training stopped by user")
                selfplay_manager.send_message_sync({
                    "type": "stopped",
                    "message": f"Training stopped at generation {gen}"
                })
                break
            
            print(f"\n{'='*50}")
            print(f"📊 GENERATION {gen + 1}/{generations}")
            print(f"{'='*50}")
            
            selfplay_manager.current_generation = gen + 1
            gen_collector_wins = 0
            gen_adversary_wins = 0
            gen_collector_reward = 0
            gen_adversary_reward = 0
            
            # Episode loop for this generation
            for ep in range(episodes_per_gen):
                if selfplay_manager.should_stop:
                    break
                
                selfplay_manager.current_episode = ep + 1
                
                # Run episode
                obs, info = env.reset()
                initial_msg = info.get("initial_message", "Hello?")
                print(f"\n  Episode {ep + 1}/{episodes_per_gen}")
                print(f"    Debtor opens: \"{initial_msg}\"")
                
                done = False
                ep_c_reward = 0
                ep_a_reward = 0
                turn = 0
                
                while not done:
                    turn += 1
                    c_action = collector.select_action(obs)
                    a_action = adversary.select_strategy(obs)
                    
                    c_action_name = EnvironmentConfig.ACTIONS.get(c_action, "unknown")
                    a_action_name = SelfPlayConfig.ADVERSARY_ACTIONS.get(a_action, "unknown")
                    
                    next_obs, c_reward, a_reward, terminated, truncated, info = env.step(
                        c_action, a_action
                    )
                    
                    # Print turn details
                    if env.state.utterances:
                        last = env.state.utterances[-1]
                        print(f"    Turn {turn}: C:[{c_action_name}] vs A:[{a_action_name}]")
                        if use_llm:
                            print(f"      Collector: {last.get('collector_utterance', '')[:60]}...")
                            print(f"      Adversary: {last.get('adversary_response', '')[:60]}...")
                    
                    # Store experiences
                    collector.store_experience(obs, c_action, c_reward, next_obs, terminated or truncated)
                    adversary.store_experience(obs, a_action, a_reward, next_obs, terminated or truncated)
                    
                    ep_c_reward += c_reward
                    ep_a_reward += a_reward
                    done = terminated or truncated
                    obs = next_obs
                    
                    # Send battle update for every turn
                    if env.state.utterances:
                        last = env.state.utterances[-1]
                        adversary_resp = last.get("adversary_response", "")
                        
                        # Clean up adversary response if it's JSON
                        if adversary_resp.startswith("{") and "response" in adversary_resp:
                            try:
                                import json
                                parsed = json.loads(adversary_resp)
                                adversary_resp = parsed.get("response", adversary_resp)
                            except:
                                pass
                        
                        selfplay_manager.send_message_sync({
                            "type": "battle",
                            "generation": gen + 1,
                            "episode": ep + 1,
                            "collector_strategy": last.get("collector_action", ""),
                            "collector_utterance": last.get("collector_utterance", ""),
                            "adversary_strategy": last.get("adversary_action", ""),
                            "adversary_response": adversary_resp
                        })
                
                # Track outcome
                outcome = info.get("outcome", "draw")
                outcome_icon = "🏆" if outcome == "collector_win" else "🛡️" if outcome == "adversary_win" else "🤝"
                print(f"    {outcome_icon} Result: {outcome} | C:{ep_c_reward:+.1f} A:{ep_a_reward:+.1f}")
                
                if outcome == "collector_win":
                    gen_collector_wins += 1
                    selfplay_manager.collector_wins += 1
                elif outcome == "adversary_win":
                    gen_adversary_wins += 1
                    selfplay_manager.adversary_wins += 1
                
                gen_collector_reward += ep_c_reward
                gen_adversary_reward += ep_a_reward
                
                # End episode for adversary stats
                adversary.episode_end(collector_succeeded=(outcome == "collector_win"))
                
                # Train agents
                if collector.replay_buffer.is_ready(RLConfig.BATCH_SIZE):
                    collector.train_step()
                if adversary.replay_buffer.is_ready(RLConfig.BATCH_SIZE):
                    adversary.train_step()
                
                # Send episode update
                selfplay_manager.send_message_sync({
                    "type": "episode",
                    "generation": gen + 1,
                    "episode": ep + 1
                })
            
            # Generation complete - calculate stats
            c_win_rate = gen_collector_wins / episodes_per_gen
            a_win_rate = gen_adversary_wins / episodes_per_gen
            avg_c_reward = gen_collector_reward / episodes_per_gen
            avg_a_reward = gen_adversary_reward / episodes_per_gen
            
            print(f"\n  📈 Gen {gen + 1} Summary:")
            print(f"     Collector Win Rate: {c_win_rate*100:.1f}%")
            print(f"     Adversary Win Rate: {a_win_rate*100:.1f}%")
            print(f"     Avg Collector Reward: {avg_c_reward:.2f}")
            print(f"     Avg Adversary Reward: {avg_a_reward:.2f}")
            
            # Get strategy distributions
            c_strategy_dist = {}
            a_strategy_dist = adversary.get_strategy_distribution()
            
            # Send generation update
            selfplay_manager.send_message_sync({
                "type": "generation",
                "generation": gen + 1,
                "collector_win_rate": round(c_win_rate, 3),
                "adversary_win_rate": round(a_win_rate, 3),
                "avg_collector_reward": round(avg_c_reward, 2),
                "avg_adversary_reward": round(avg_a_reward, 2),
                "collector_strategy_dist": c_strategy_dist,
                "adversary_strategy_dist": a_strategy_dist
            })
            
            # Save to pools
            pool_manager.add_collector(collector, gen, c_win_rate)
            pool_manager.add_adversary(adversary, gen, a_win_rate)
        
        # Save final checkpoints
        collector.save(os.path.join(CHECKPOINT_DIR, "collector_final.pt"))
        adversary.save(os.path.join(CHECKPOINT_DIR, "adversary_final.pt"))
        print(f"\n✓ Checkpoints saved to {CHECKPOINT_DIR}")
        
        # Training complete
        total = selfplay_manager.collector_wins + selfplay_manager.adversary_wins
        final_c_rate = selfplay_manager.collector_wins / total if total > 0 else 0
        final_a_rate = selfplay_manager.adversary_wins / total if total > 0 else 0
        
        print("\n" + "="*70)
        print("🎉 SELF-PLAY TRAINING COMPLETE!")
        print(f"   Final Collector Win Rate: {final_c_rate*100:.1f}%")
        print(f"   Final Adversary Win Rate: {final_a_rate*100:.1f}%")
        print("="*70 + "\n")
        
        selfplay_manager.send_message_sync({
            "type": "complete",
            "generations": selfplay_manager.current_generation,
            "collector_win_rate": round(final_c_rate, 3),
            "adversary_win_rate": round(final_a_rate, 3),
            "message": "Self-play training complete!"
        })
        
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR: {e}")
        traceback.print_exc()
        selfplay_manager.send_message_sync({
            "type": "error",
            "message": str(e)
        })
    finally:
        selfplay_manager.is_training = False
        selfplay_manager.should_stop = False


@router.websocket("/ws")
async def selfplay_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for live self-play training updates
    
    Client sends: {"action": "start", "generations": 10, "episodes_per_gen": 50, ...}
    Server sends: {"type": "generation", "generation": 1, "collector_win_rate": 0.4, ...}
    """
    await websocket.accept()
    selfplay_manager.websocket = websocket
    selfplay_manager.loop = asyncio.get_event_loop()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if data.get("action") == "start":
                if selfplay_manager.is_training:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Training already in progress"
                    })
                    continue
                
                # Start training in background thread
                selfplay_manager.is_training = True
                selfplay_manager.should_stop = False
                selfplay_manager.total_generations = data.get("generations", 10)
                selfplay_manager.episodes_per_gen = data.get("episodes_per_gen", 50)
                selfplay_manager.current_generation = 0
                selfplay_manager.current_episode = 0
                
                await websocket.send_json({
                    "type": "started",
                    "message": f"Starting self-play for {selfplay_manager.total_generations} generations"
                })
                
                # Run training in thread
                thread = threading.Thread(
                    target=run_selfplay_training,
                    args=(
                        selfplay_manager.total_generations,
                        selfplay_manager.episodes_per_gen,
                        data.get("use_llm", False),
                        data.get("zero_sum", True)
                    )
                )
                thread.start()
            
            elif data.get("action") == "stop":
                if selfplay_manager.is_training:
                    selfplay_manager.should_stop = True
                    await websocket.send_json({
                        "type": "stopping",
                        "message": "Stop requested, finishing current episode..."
                    })
            
            elif data.get("action") == "status":
                await websocket.send_json({
                    "type": "status",
                    **selfplay_manager.get_status()
                })
    
    except WebSocketDisconnect:
        print("Self-play WebSocket disconnected")
    finally:
        selfplay_manager.websocket = None
