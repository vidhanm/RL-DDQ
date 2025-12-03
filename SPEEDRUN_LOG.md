# Speedrun Log: Phase 5-6 Development

> **Created**: December 1, 2025  
> **Branch**: `speedrun-phase5-6`  
> **Purpose**: Track all decisions, implementations, and reasoning during rapid development

---

## 📋 Overview

This document tracks all development decisions and implementations during the Phase 5-6 speedrun. Each section documents:
- **What** we're building
- **Why** we made specific decisions
- **How** it integrates with existing code

---

## 🎯 Goals Summary

### Phase 5: Enhancement & Evaluation
| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Visualizations (learning curves, heatmaps, Q-value) | HIGH | ✅ DONE | Created comprehensive visualize.py + analysis/ package |
| Ablation Study Setup (K=2 vs K=5 vs K=10) | HIGH | ✅ DONE | Framework ready, needs DDQ runs with different K |
| Comparison Tools (DQN vs DDQ) | HIGH | ✅ DONE | Side-by-side analysis with metrics |
| Example Conversation Recording | MEDIUM | ⏳ TODO | Shows agent in action |

### Phase 6: Optional Enhancements
| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Voice Integration (LiveKit) | HIGH | ✅ DONE | STT/TTS + LiveKit real-time voice |
| Web Interface | HIGH | ✅ DONE | Gradio app.py with live conversation |
| Advanced World Model Architectures | MEDIUM | ✅ DONE | 4 variants: Probabilistic, LSTM, Transformer, Ensemble |
| Multi-step Planning | MEDIUM | ✅ DONE | 4 planners: Rollout, Tree, MPC/CEM, Uncertainty |
| Curriculum Learning | LOW | ✅ DONE | Adaptive 4-stage progression with regression |

---

## 📝 Multi-Step Plan

### Step 1: Visualization & Analysis Tools ✅ COMPLETE
**Goal**: Build comprehensive tools to analyze training runs

**Sub-tasks**:
1. ✅ Create `visualize.py` - Main visualization script
   - ✅ Learning curves (reward, success rate, loss over episodes)
   - ✅ Moving averages (10-episode, 50-episode windows)
   - ✅ Side-by-side DQN vs DDQ comparison
   
2. ✅ Create action heatmaps
   - ✅ Action distribution per persona type
   - ✅ Action frequency over training (exploration → exploitation)
   
3. ✅ Q-value analysis
   - ✅ Q-value distribution visualization
   - ✅ State-action value heatmaps

**Files created**:
- ✅ `visualize.py` - Main visualization script
- ✅ `analysis/` folder for analysis utilities
  - ✅ `analysis/__init__.py`
  - ✅ `analysis/history_loader.py`
  - ✅ `analysis/metrics.py`
  - ✅ `analysis/plot_utils.py`

---

### Step 2: Ablation Study Framework ✅ COMPLETE
**Goal**: Easy way to run and compare experiments with different hyperparameters

**Sub-tasks**:
1. ✅ Ablation plotting in `visualize.py --ablation`
2. ✅ Support different K values (imagination factor)
3. ✅ Auto-generate comparison reports

**Experiments to support** (framework ready, needs runs):
- ⏳ K=2 vs K=5 vs K=10 (imagination depth) - needs training runs
- ⏳ Different real_ratio values (0.5, 0.75, 0.9) - needs training runs
- ⏳ Learning rate variations - needs training runs

---

### Step 3: Advanced World Model Architectures ✅ COMPLETE
**Status**: Done
**Goal**: Improve world model prediction accuracy

**Architectures Implemented** (in `agent/advanced_world_models.py`):

1. **ProbabilisticWorldModel** ✅
   - Outputs Gaussian distributions (mean, log_variance) instead of point estimates
   - Enables uncertainty-aware planning via variance
   - Uses reparameterization trick for sampling
   - Negative log-likelihood loss for training

2. **RecurrentWorldModel (LSTM)** ✅
   - 2-layer LSTM with dropout for temporal patterns
   - Hidden state persistence across predictions
   - `predict_sequence()` for multi-step rollouts
   - Good for capturing conversation dynamics

3. **TransformerWorldModel** ✅
   - Self-attention over state-action history
   - Positional encoding for sequence order
   - Causal masking (can't see future)
   - `predict_with_history()` for context-aware predictions

4. **EnhancedEnsembleWorldModel** ✅
   - 5 diverse models with different initializations
   - Disagreement-based uncertainty (std across models)
   - Thompson sampling for exploration
   - Independent optimizers for diversity

**Factory Function**: `create_advanced_world_model(model_type, ...)`
- Supports: 'probabilistic', 'recurrent'/'lstm', 'transformer', 'ensemble'

**Utility Class**: `WorldModelComparison`
- Compare multiple model architectures on same data
- Pretty-print comparison results

**Files Modified**:
- Created: `agent/advanced_world_models.py` (~580 lines)
- Updated: `agent/__init__.py` (new exports)

---

### Step 4: Multi-step Planning ✅ COMPLETE
**Status**: Done
**Goal**: Plan multiple steps ahead instead of just 1-step imagination

**Planners Implemented** (in `agent/multistep_planning.py`):

1. **RolloutPlanner** ✅
   - Monte Carlo rollouts from each action
   - Configurable horizon and number of rollouts
   - Multiple rollout policies: random, epsilon-greedy, greedy
   - Uses Q-network for terminal value estimation

2. **TreeSearchPlanner** ✅
   - Best-first or depth-first tree expansion
   - Expands all actions per node
   - Value backup from leaves to root
   - Configurable max depth and expansions

3. **MPCPlanner (Model Predictive Control)** ✅
   - Random shooting: sample many sequences, pick best
   - Cross-entropy method (CEM): iteratively refine action distribution
   - Configurable elite selection and CEM iterations
   - Optimizes action sequence over horizon

4. **UncertaintyAwarePlanner** ✅
   - Works with ensemble or probabilistic world models
   - Penalizes actions with high uncertainty
   - Stops rollout if uncertainty exceeds threshold
   - Balances value and confidence

**Factory Function**: `create_planner(planner_type, ...)`
- Supports: 'rollout', 'tree', 'mpc'/'cem', 'uncertainty'

**Utility Class**: `PlannerComparison`
- Compare planners on same states
- Measure planning time and Q-network agreement
- Pretty-print comparison tables

**Files Created**:
- `agent/multistep_planning.py` (~580 lines)
- Updated: `agent/__init__.py` (new exports)

---

### Step 5: Web Interface ✅ COMPLETE
**Status**: Done
**Goal**: Interactive demo for presentations

**Tech stack**: Gradio (fastest to implement, good for ML demos)

**Features built**:
- ✅ Load trained model (DQN or DDQ)
- ✅ Run live conversation with persona selection
- ✅ Show agent's Q-values for each action
- ✅ Display current state visualization
- ✅ Show conversation history
- ✅ Auto-play mode (let agent run full episode)
- ✅ Training analysis tab with plots
- ✅ About tab with project description

**Files created**:
- `app.py` - Main Gradio application

**Usage**:
```bash
python app.py              # Launch locally on port 7860
python app.py --share      # Create public shareable link
python app.py --model ddq  # Load DDQ model by default
```

---

### Step 6: Voice Integration (LiveKit) ✅ COMPLETE
**Status**: Done
**Goal**: Real-time voice conversations with the agent

**Architecture**:
```
User Voice → STT → Text → Agent → Response Text → TTS → Audio
```

**Components Built** (in `voice_integration.py`):

1. **STT Handlers** ✅
   - `WhisperSTTHandler` - Local Whisper model for transcription
   - `DeepgramSTTHandler` - Cloud-based streaming STT
   - Supports streaming and batch transcription

2. **TTS Handlers** ✅
   - `LocalTTSHandler` - pyttsx3 local TTS
   - `ElevenLabsTTSHandler` - Cloud-based high-quality TTS
   - Streaming audio output support

3. **VoiceAgent** ✅
   - Main orchestrator for voice conversations
   - Integrates STT → RL Agent → TTS pipeline
   - State management (IDLE, LISTENING, PROCESSING, SPEAKING)
   - Conversation history tracking

4. **LiveKitVoiceRoom** ✅
   - WebRTC room connection via LiveKit
   - Audio track subscription and publishing
   - Silence detection for end-of-speech
   - Token generation for authentication

**Configuration**: `VoiceConfig` dataclass with:
- LiveKit server settings
- STT/TTS provider selection
- Audio parameters (sample rate, channels)
- Timeout configurations

**Files Created**:
- `voice_integration.py` (~650 lines)
- Updated: `requirements.txt` (voice dependencies)

**Usage**:
```python
from voice_integration import VoiceAgent, VoiceConfig, LiveKitVoiceRoom

# Create voice agent
config = VoiceConfig(stt_provider="whisper", tts_provider="local")
voice_agent = VoiceAgent(rl_agent, state_encoder, env, config)

# Process audio
response = await voice_agent.process_speech(audio_bytes)

# Or use LiveKit for real-time
room = LiveKitVoiceRoom(voice_agent, config)
await room.connect("my-room", "agent")
```

**Install Voice Dependencies**:
```bash
pip install livekit livekit-agents openai-whisper pyttsx3 aiohttp
pip install deepgram-sdk  # Optional: cloud STT
pip install elevenlabs    # Optional: cloud TTS
```

---

### Step 7: Curriculum Learning ✅ COMPLETE
**Status**: Done
**Goal**: Train agent progressively on harder scenarios

**Curriculum Stages** (in `curriculum_learning.py`):

| Stage | Personas | Min Episodes | Success Threshold |
|-------|----------|--------------|-------------------|
| Stage 1 | Cooperative only | 30 | 70% |
| Stage 2 | + Sad/Overwhelmed | 50 | 60% |
| Stage 3 | + Avoidant | 75 | 55% |
| Stage 4 | + Angry (full) | 100 | 50% |

**Components Built**:

1. **CurriculumScheduler** ✅
   - Controls persona distribution per stage
   - Auto-advances when success threshold met
   - Tracks per-stage history and statistics
   - `sample_persona()` for training loop

2. **AdaptiveCurriculum** ✅
   - Extends CurriculumScheduler
   - Dynamically adjusts persona weights based on performance
   - Regression to easier stage if struggling
   - Per-persona success tracking

3. **CurriculumTrainer** ✅
   - Wraps training loop with curriculum
   - Automatic persona selection
   - Checkpoint saving (curriculum state + logs)
   - Progress printing and callbacks

4. **Visualization** ✅
   - `plot_curriculum_progress()` function
   - Rewards by stage (color-coded)
   - Success rate bar chart per stage
   - Stage progression timeline

**Files Created**:
- `curriculum_learning.py` (~520 lines)

**Usage**:
```python
from curriculum_learning import AdaptiveCurriculum, CurriculumTrainer

# Create curriculum
curriculum = AdaptiveCurriculum(start_stage=CurriculumStage.STAGE_1)

# Create trainer
trainer = CurriculumTrainer(agent, env, curriculum)

# Train with curriculum
summary = trainer.train(num_episodes=500, print_every=20)

# Or manually use curriculum
persona = curriculum.sample_persona()
# ... run episode ...
curriculum.record_episode(success=True)
```

---

## 📊 Decision Log

### Decision 1: Visualization Architecture
**Date**: 2025-12-01  
**Topic**: How to structure the visualization system  
**Options Considered**:
1. Single monolithic `visualize.py` - Simple but hard to maintain
2. Extend existing `evaluate.py` - Would bloat the file
3. Modular `analysis/` package + `visualize.py` CLI - Clean separation

**Decision**: Option 3 - Modular architecture  
**Reasoning**: 
- Reusable components for notebooks and scripts
- Clean separation of concerns (loading, metrics, plotting)
- Easy to test individual components
- Follows Python best practices

**Impact**: Created 4 new files in modular structure

### Decision 2: History File Format
**Date**: 2025-12-01  
**Topic**: How to save and identify training runs  
**Options Considered**:
1. Keep overwriting `training_history.json` - Loses data
2. Append to single file - Complex parsing
3. Unique filenames with metadata - Best of both worlds

**Decision**: Option 3 - `{algorithm}_{episodes}ep_{timestamp}.json` with embedded metadata  
**Reasoning**:
- Each run is preserved
- Metadata embedded for self-documentation
- Easy to discover and compare runs
- Supports old format for backwards compatibility

**Impact**: Modified `train.py`, created history loader with format detection

### Decision 3: Plot Styling
**Date**: 2025-12-01  
**Topic**: Visual appearance of plots  
**Options Considered**:
1. Default matplotlib - Looks unprofessional
2. Seaborn defaults - Good but generic
3. Custom colorblind-friendly palette - Professional and accessible

**Decision**: Option 3 - Custom IBM colorblind-friendly palette  
**Reasoning**:
- Publication quality (300 DPI)
- Accessible to colorblind viewers
- Consistent branding across all plots
- Professional appearance for demos

**Impact**: Created `PlotStyle` class with consistent configuration

### Decision 4: Web Interface Framework
**Date**: 2025-12-01  
**Topic**: Which framework to use for the demo web interface  
**Options Considered**:
1. Gradio - Fastest to implement, good for ML demos, built-in sharing
2. Streamlit - More customizable, Python-native
3. FastAPI + React - Most professional, but too much work for speedrun

**Decision**: Gradio  
**Reasoning**:
- Fastest implementation (critical for speedrun)
- Built-in `share=True` for instant public URL
- Native support for ML components (chatbot, dataframes)
- No frontend code needed
- Easy to show Q-values and state visualizations

**Impact**: Created `app.py` with full demo interface

---

## 🔧 Implementation Log

### Implementation 1: Visualization & Analysis System
**Date**: 2025-12-01  
**Feature**: Comprehensive training visualization and analysis tools

**Files Created**:
- `analysis/__init__.py` - Package initialization
- `analysis/history_loader.py` - Load and parse training history files
- `analysis/metrics.py` - Calculate comparison metrics and statistics  
- `analysis/plot_utils.py` - Plotting utilities and consistent styling
- `visualize.py` - Main CLI script for generating visualizations

**Key Code Decisions**:
1. **TrainingRun dataclass**: Encapsulates all run data with computed properties (success_rate, avg_reward, smoothing methods)
2. **Dual format support**: Loader handles both old format (flat JSON) and new format (with metadata)
3. **MetricsCalculator**: Static methods for computing sample efficiency, learning speed, and statistical comparisons
4. **PlotStyle singleton**: Ensures consistent styling across all visualizations

**Features**:
- Auto-discover all `.json` files in checkpoints/
- Learning curves with smoothing and confidence bands
- DQN vs DDQ side-by-side comparison
- Ablation study support (compare different K values)
- Sample efficiency visualization (episodes to 50% success)
- Publication-ready plots (300 DPI, colorblind-friendly)

**CLI Usage**:
```bash
python visualize.py                    # Generate full report
python visualize.py --summary          # Print summary only
python visualize.py --compare          # Compare DQN vs DDQ
python visualize.py --ablation         # Compare K values
python visualize.py --show             # Show plots interactively
```

**Testing Notes**:
- Run `python visualize.py --summary` to verify loading works
- Need at least one training run file to generate plots
- Comparison requires both DQN and DDQ runs

### Implementation 2: Web Interface (Gradio)
**Date**: 2025-12-01  
**Feature**: Interactive demo application for presentations

**Files Created**:
- `app.py` - Main Gradio application (400+ lines)

**Key Code Decisions**:
1. **DemoApp class**: Encapsulates all state (model, environment, conversation)
2. **Three-tab layout**: Live Conversation, Training Analysis, About
3. **Real-time Q-value display**: Shows agent's decision reasoning
4. **Auto-play mode**: Let agent complete full episode automatically
5. **Persona selection**: Choose specific debtor type or random

**Features**:
- Load DQN or DDQ trained models
- Start conversations with different personas
- Manual action selection or agent auto-selection
- Real-time state and Q-value visualization
- Conversation history display
- Training analysis with saved plots
- Project documentation in About tab

**CLI Usage**:
```bash
python app.py              # Launch on localhost:7860
python app.py --share      # Create public URL
python app.py --model ddq  # Load DDQ by default
```

**Testing Notes**:
- Requires gradio>=4.0.0
- Works without LLM client (uses placeholder responses)
- Best tested with trained model checkpoints

---

## ⚠️ Known Issues & TODOs

- [ ] Issue 1
- [ ] Issue 2

---

## 📚 Resources & References

### LiveKit
- Documentation: https://docs.livekit.io/
- Python SDK: https://github.com/livekit/python-sdks

### Visualization
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- Plotly (interactive): https://plotly.com/python/

### World Models
- Dreamer V3: https://arxiv.org/abs/2301.04104
- MuZero: https://arxiv.org/abs/1911.08265

---

## 📅 Session History

### Session 1: December 1, 2025
**Duration**: Active  
**Focus**: Project setup, visualization system  
**Completed**:
- [x] Created SPEEDRUN_LOG.md
- [x] Renamed old training_history.json → dqn_100ep_old.json
- [x] Fixed train.py to save unique history files with metadata
- [x] Created speedrun-phase5-6 branch
- [x] Built comprehensive visualization system:
  - analysis/__init__.py
  - analysis/history_loader.py (TrainingRun dataclass, format detection)
  - analysis/metrics.py (comparison metrics, statistical tests)
  - analysis/plot_utils.py (consistent styling, colorblind-friendly)
  - visualize.py (CLI with multiple modes)

**Next Steps**:
- [ ] Test visualization with existing data
- [ ] Build conversation recording system
- [ ] Start Phase 6: Advanced world models OR web interface

---

*Last Updated: December 1, 2025*
