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
| Visualizations (learning curves, heatmaps, Q-value) | HIGH | ⏳ TODO | Critical for understanding agent behavior |
| Ablation Study Setup (K=2 vs K=5 vs K=10) | HIGH | ⏳ TODO | Proves DDQ value scientifically |
| Comparison Tools (DQN vs DDQ) | HIGH | ⏳ TODO | Need side-by-side analysis |
| Example Conversation Recording | MEDIUM | ⏳ TODO | Shows agent in action |

### Phase 6: Optional Enhancements
| Task | Priority | Status | Notes |
|------|----------|--------|-------|
| Voice Integration (LiveKit) | HIGH | ⏳ TODO | Real-time voice conversations |
| Web Interface | HIGH | ⏳ TODO | Demo-ready presentation |
| Advanced World Model Architectures | MEDIUM | ⏳ TODO | Transformer, ensemble variants |
| Multi-step Planning | MEDIUM | ⏳ TODO | Beyond 1-step imagination |
| Curriculum Learning | LOW | ⏳ TODO | Easy → Hard persona progression |

---

## 📝 Multi-Step Plan

### Step 1: Visualization & Analysis Tools
**Goal**: Build comprehensive tools to analyze training runs

**Sub-tasks**:
1. Create `visualize.py` - Main visualization script
   - Learning curves (reward, success rate, loss over episodes)
   - Moving averages (10-episode, 50-episode windows)
   - Side-by-side DQN vs DDQ comparison
   
2. Create action heatmaps
   - Action distribution per persona type
   - Action frequency over training (exploration → exploitation)
   
3. Q-value analysis
   - Q-value distribution visualization
   - State-action value heatmaps

**Files to create**:
- `visualize.py` - Main visualization script
- `analysis/` folder for analysis utilities

---

### Step 2: Ablation Study Framework
**Goal**: Easy way to run and compare experiments with different hyperparameters

**Sub-tasks**:
1. Create `run_ablation.py` - Script to run multiple experiments
2. Support different K values (imagination factor)
3. Auto-generate comparison reports

**Experiments to support**:
- K=2 vs K=5 vs K=10 (imagination depth)
- Different real_ratio values (0.5, 0.75, 0.9)
- Learning rate variations

---

### Step 3: Advanced World Model Architectures
**Goal**: Improve world model prediction accuracy

**Options to explore**:
1. **Ensemble World Model** - Multiple models, average predictions
2. **Transformer-based World Model** - Attention over state history
3. **Probabilistic World Model** - Output distributions, not point estimates
4. **Recurrent World Model** - LSTM/GRU for temporal patterns

**Decision needed**: Which architecture provides best trade-off of:
- Prediction accuracy
- Training speed
- Implementation complexity

---

### Step 4: Multi-step Planning
**Goal**: Plan multiple steps ahead instead of just 1-step imagination

**Approach options**:
1. **Tree Search** - Expand possible action sequences, evaluate leaves
2. **Model Predictive Control (MPC)** - Optimize action sequence over horizon
3. **Rollout Planning** - Simulate full episodes with world model

**Considerations**:
- Error accumulation over multiple steps
- Computational cost
- Integration with existing DDQ agent

---

### Step 5: Web Interface
**Goal**: Interactive demo for presentations

**Tech stack options**:
1. **Gradio** - Fastest to implement, good for ML demos
2. **Streamlit** - More customizable, Python-native
3. **FastAPI + React** - Most professional, more work

**Features needed**:
- Load trained model
- Run live conversation
- Show agent's Q-values and decision reasoning
- Visualize state changes in real-time

---

### Step 6: Voice Integration (LiveKit)
**Goal**: Real-time voice conversations with the agent

**Architecture**:
```
User Voice → STT → Text → Agent → Response Text → TTS → Audio
```

**LiveKit benefits**:
- Low latency
- WebRTC-based
- Good Python SDK

**Components needed**:
1. LiveKit room setup
2. STT integration (Whisper or LiveKit's STT)
3. TTS integration (ElevenLabs, OpenAI TTS, or LiveKit's TTS)
4. Audio streaming pipeline

---

### Step 7: Curriculum Learning
**Goal**: Train agent progressively on harder scenarios

**Curriculum design**:
1. **Stage 1**: Cooperative persona only (easiest)
2. **Stage 2**: Add Sad/Overwhelmed persona
3. **Stage 3**: Add Avoidant persona
4. **Stage 4**: Add Angry persona (hardest)

**Implementation**:
- Persona scheduler based on training progress
- Success rate thresholds to advance stages

---

## 📊 Decision Log

### Decision 1: [Template]
**Date**: YYYY-MM-DD  
**Topic**: [What decision was made]  
**Options Considered**:
1. Option A - Pros/Cons
2. Option B - Pros/Cons

**Decision**: [What we chose]  
**Reasoning**: [Why we chose it]  
**Impact**: [How it affects the project]

---

## 🔧 Implementation Log

### Implementation 1: [Template]
**Date**: YYYY-MM-DD  
**Feature**: [What was implemented]  
**Files Changed**:
- `file1.py` - Description of changes
- `file2.py` - Description of changes

**Key Code Decisions**:
- Decision 1: Why we did X instead of Y
- Decision 2: Why we structured it this way

**Testing Notes**:
- How to test this feature
- Known limitations

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
**Focus**: Project setup, planning  
**Completed**:
- [x] Created SPEEDRUN_LOG.md
- [x] Renamed old training_history.json → dqn_100ep_old.json
- [x] Fixed train.py to save unique history files with metadata
- [ ] Create speedrun-phase5-6 branch
- [ ] Start Step 1: Visualization tools

**Next Session Goals**:
- TBD based on progress

---

*Last Updated: December 1, 2025*
