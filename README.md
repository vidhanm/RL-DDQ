# Self-Improving Debt Collection AI Agent

A reinforcement learning system that learns optimal debt collection strategies through conversations, using DDQ (Dyna-style Data-efficient Q-learning) for sample-efficient learning.

## Overview

This project implements a **self-improving, self-modifying voice agent** for debt collection using:
- **Reinforcement Learning**: Agent learns which strategies work best
- **LLM Integration**: Natural language generation for realistic conversations
- **DDQ Algorithm**: 5-10x faster learning through world model imagination
- **NLU-Based State Extraction**: Deterministic behavioral signals from text
- **Domain Randomization**: Diverse debtor simulation for robust generalization
- **Expert Reward Shaping**: Encodes proven debt collection strategies into rewards
- **Multilingual Support**: Hindi, Hinglish, and English prompts for Indian market

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Domain          │    │ LLM Simulates   │    │ NLU Extracts    │         │
│  │ Randomization   │───▶│ Debtor Response │───▶│ State Features  │         │
│  │ (random profile)│    │ (realistic text)│    │ (deterministic) │         │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘         │
│                                                          │                  │
│                                                          ▼                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Q-Network       │◀───│ Experience      │◀───│ State + Reward  │         │
│  │ (learns policy) │    │ Replay Buffer   │    │ (stable signal) │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION PIPELINE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │ Real Debtor     │    │ Speech-to-Text  │    │ Same NLU        │         │
│  │ (unknown type)  │───▶│ Transcription   │───▶│ Extraction      │         │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘         │
│                                                          │                  │
│                                                          ▼                  │
│  ┌─────────────────┐                           ┌─────────────────┐         │
│  │ Agent Response  │◀──────────────────────────│ Same State      │         │
│  │ (learned policy)│                           │ Representation! │         │
│  └─────────────────┘                           └─────────────────┘         │
│                                                                             │
│  Key: Agent discovers debtor personality through conversation,              │
│       just like a human collector would!                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Architecture?

| Challenge | Solution |
|-----------|----------|
| Don't know debtor type before calling | Start with neutral state, adapt from responses |
| LLM outputs are inconsistent | Use deterministic NLU on LLM-generated text |
| 4 personas too narrow | Domain randomization creates millions of profiles |
| Hardcoded rules may not match reality | LLM has real knowledge of human behavior |
| Need same system for training & production | NLU gives identical state representation |

**See [PHASE7_NLU_ARCHITECTURE.md](PHASE7_NLU_ARCHITECTURE.md) for implementation details.**

## Key Features

- **Adaptive Learning**: Agent improves conversation strategies over time
- **Sample Efficiency**: DDQ generates synthetic training data, reducing LLM API costs
- **Realistic Simulation**: LLM-powered debtor responses with domain randomization
- **Robust Generalization**: NLU-based state works on unknown debtor types
- **Dual Approach**: Compare vanilla DQN vs. DDQ performance
- **Interpretable**: Visualize learned strategies and world model predictions
- **Expert Knowledge**: Reward function encodes debt collection best practices
- **Hindi/Hinglish Support**: Full multilingual prompts for Indian market

## Core Components

```
┌──────────────────────────────────────────────────────────────────┐
│                        CORE COMPONENTS                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Domain Randomizer ──▶ Samples diverse debtor profiles           │
│         │                                                        │
│         ▼                                                        │
│  LLM (NVIDIA/OpenAI) ──▶ Generates realistic conversations       │
│         │                                                        │
│         ▼                                                        │
│  NLU State Extractor ──▶ Deterministic behavioral features       │
│         │                                                        │
│         ▼                                                        │
│  DDQ Agent ──▶ Q-network + World Model imagination               │
│         │                                                        │
│         ▼                                                        │
│  Curriculum Learning ──▶ Easy-to-hard training progression       │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

See [WORKFLOW.md](WORKFLOW.md) for detailed system mechanics.

## Project Structure

```
RL DDQ/
├── environment/              # Debtor simulator (Gymnasium interface)
│   ├── debtor_env.py        # Legacy environment (persona-based)
│   ├── nlu_env.py           # NLU environment (Phase 7, recommended)
│   ├── debtor_persona.py    # Persona definitions (legacy)
│   └── domain_randomizer.py # Domain randomization (Phase 7)
├── agent/                   # RL agent components
│   ├── dqn.py              # DQN network
│   ├── dqn_agent.py        # DQN agent (Double DQN + PER)
│   ├── ddq_agent.py        # DDQ algorithm (world model imagination)
│   └── world_model.py      # World model for DDQ
├── nlu/                    # Natural Language Understanding (Phase 7)
│   └── state_extractor.py  # VADER sentiment + intent + cooperation
├── llm/                    # LLM integration
│   ├── nvidia_client.py    # NVIDIA NIM API wrapper
│   ├── openai_client.py    # OpenAI API wrapper
│   └── prompts.py          # Prompt templates
├── utils/                  # Utilities
│   ├── replay_buffer.py    # Experience replay (uniform + prioritized)
│   └── state_encoder.py    # State representation
├── train.py                # Training script (--legacy-env for old env)
├── evaluate.py             # Evaluation and demo
├── curriculum_learning.py  # Progressive difficulty training
├── config.py               # Hyperparameters
├── requirements.txt        # Dependencies
├── CRITICAL_FIXES.md       # 6 high-impact improvements (completed)
├── PHASE7_NLU_ARCHITECTURE.md  # NLU + Domain Randomization (completed)
├── WORKFLOW.md             # Detailed workflow documentation
└── README.md               # This file
```

## Installation

### Prerequisites
- Python 3.8+
- OpenAI API key

### Setup

```bash
# Clone/navigate to project
cd "RL DDQ"

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key
export OPENAI_API_KEY="your-key-here"  # Linux/Mac
set OPENAI_API_KEY=your-key-here       # Windows
```

## Quick Start

### Train Agent

```bash
# Train with DDQ (recommended)
python train.py --algorithm ddq --episodes 200

# Train with DQN only (baseline)
python train.py --algorithm dqn --episodes 200
```

### Evaluate/Demo

```bash
# Run evaluation and see live conversations
python evaluate.py --model checkpoints/ddq_best.pth --num_episodes 10
```

### Compare DQN vs DDQ

```bash
# Train both and generate comparison plots
python train.py --compare --episodes 200
```

## Configuration

Edit [config.py](config.py) to adjust:
- **RL Hyperparameters**: learning rate, epsilon, gamma, etc.
- **DDQ Settings**: imagination factor K, world model architecture
- **Environment**: number of personas, conversation length, reward weights
- **LLM**: model selection, temperature, max tokens

## How It Works

### 1. Conversation Flow

```
Agent chooses strategy → LLM generates utterance → Debtor (LLM) responds
→ Calculate reward → Update state → Repeat
```

### 2. Training (DDQ)

```
Collect real experiences → Train world model → Generate imagined experiences
→ Train DQN on real + imagined → Repeat
```

### 3. Key Innovation

**DQN**: 100 episodes = 800 experiences
**DDQ**: 100 episodes = 800 real + 4,000 imagined = 4,800 experiences

**Result**: 6x faster learning, same LLM cost!

## Action Space

The agent can choose from **9 high-level strategies**:

### Original Actions
1. **Empathetic Listening**: Show understanding and compassion
2. **Ask About Situation**: Inquire about circumstances
3. **Firm Reminder**: Professional but assertive
4. **Offer Payment Plan**: Propose installment options
5. **Propose Settlement**: Offer reduced amount
6. **Hard Close**: Create urgency with consequences

### New Expert-Recommended Actions
7. **Acknowledge and Redirect**: When debtor vents or goes off-topic, acknowledge then guide back
8. **Validate Then Offer**: Deep emotional validation followed by solution presentation
9. **Gentle Urgency**: Create importance without threats ("protect your credit score")

## Language Support

The system supports three languages for the Indian market:

```python
from src.llm.prompts import set_language

set_language("english")   # Default
set_language("hindi")     # Pure Hindi (Devanagari script)
set_language("hinglish")  # Hindi-English mix (common in India)
```

All 9 action strategies have prompts in all 3 languages.

## Expert Reward Shaping

The reward function encodes proven debt collection best practices:

| Positive Rewards | Value | Reasoning |
|-----------------|-------|----------|
| Empathy before pressure | +2.0 | Builds trust |
| De-escalate hostility | +3.0 | Critical skill |
| Offer flexible options | +2.0 | Increases commitment |
| Recovery from negative | +2.5 | Shows resilience |

| Penalties | Value | Reasoning |
|-----------|-------|----------|
| Premature hard close | -3.0 | Damages trust |
| Pressure on hostile debtor | -3.0 | Escalates situation |
| Repeated failed strategy | -2.0 | Inflexibility = failure |

## Debtor Simulation

### Domain Randomization (Recommended - Phase 7)

Instead of 4 fixed personas, we sample continuous parameters:

| Parameter | Range | Example |
|-----------|-------|---------|
| Agreeableness | 0.0 - 1.0 | 0.3 (disagreeable) |
| Emotional Stability | 0.0 - 1.0 | 0.6 (somewhat stable) |
| Financial Stress | 0.0 - 1.0 | 0.8 (high stress) |
| Life Event | none, job_loss, medical, divorce | job_loss |
| Communication Style | terse ↔ verbose, evasive ↔ direct | direct, terse |

This creates **millions of unique debtor profiles** for robust training.

### Legacy Personas (Fallback)

- **Angry**: Defensive, easily frustrated, needs empathy first
- **Cooperative**: Willing to work together, responds well to plans
- **Sad**: Overwhelmed, emotional, needs understanding
- **Avoidant**: Tries to end conversation, needs engagement

## Performance Metrics

- **Success Rate**: % of conversations ending in payment commitment
- **Average Reward**: Mean reward per episode
- **Sentiment Improvement**: Average sentiment change during conversation
- **Efficiency**: Average turns to reach commitment
- **Sample Efficiency**: Learning speed (episodes to reach 70% success)

## Visualization

The project generates:
- Learning curves (DQN vs DDQ comparison)
- Success rate over episodes
- Heatmaps of Q-values per state/persona
- World model prediction accuracy
- Example conversation transcripts

## Development Roadmap

### Phase 1: Foundation ✅ **COMPLETE**
- [x] Project setup
- [x] Documentation (WORKFLOW.md, README.md, CONTEXT.md)
- [x] Environment implementation
- [x] LLM integration (OpenAI + NVIDIA NIM)

### Phase 2: Environment & LLM ✅ **COMPLETE**
- [x] DebtorEnv class (Gymnasium interface)
- [x] Persona definitions (4 personas)
- [x] Prompt templates
- [x] State encoding/decoding
- [x] Reward function

### Phase 3: Baseline DQN ✅ **COMPLETE**
- [x] DQN network (with Dueling variant)
- [x] Replay buffer (with prioritized variant)
- [x] Training loop
- [x] Epsilon-greedy exploration
- [x] Target network updates
- [x] Neptune.ai integration
- [x] Evaluation script with plotting

### Phase 4: World Model & DDQ ✅ **COMPLETE**
- [x] World model network architecture
- [x] World model training on real experiences
- [x] Imagination mechanism (K=5)
- [x] DDQ training loop (75% real + 25% imagined)
- [x] Ensemble and uncertainty estimation (optional)
- [x] Performance comparison tools

### Phase 5: Enhancement & Evaluation ⏳ **IN PROGRESS**
- [x] All 4 debtor personas
- [x] Persona-conditioned world model
- [x] Basic plotting capabilities
- [ ] Hyperparameter tuning (full training run)
- [ ] Comprehensive visualizations (learning curves, heatmaps, Q-value analysis)
- [ ] Record example conversations
- [ ] Ablation studies (K=2 vs K=5 vs K=10)
- [ ] Final documentation and demo preparation

### Phase 6: Expert Enhancements ✅ **COMPLETE**
- [x] Expert-knowledge reward shaping (7 rewards, 6 penalties)
- [x] Hindi/Hinglish language support
- [x] 3 new action strategies (9 total actions)
- [x] Action history tracking for context-aware rewards

### Phase 7: Optional Advanced Features
- [ ] Voice integration (TTS/STT)
- [ ] Web interface for live demos
- [ ] Adversarial self-play training
- [ ] Conversation phase detection
- [ ] Meta-learning for fast adaptation

## Known Limitations

1. **World Model Errors**: Predictions can be inaccurate, leading to suboptimal learning
   - *Mitigation*: Ensemble models, uncertainty estimation, limited planning horizon

2. **Distribution Shift**: World model trained on early (random) agent may not generalize
   - *Mitigation*: Continuous world model updates, prioritize recent experiences

3. **Gaming**: Agent might find adversarial inputs that fool world model
   - *Mitigation*: Disagreement penalty, regularization toward real experience

See [doc.md](doc.md) for comprehensive analysis of DDQ limitations and mitigations.

## Research Background

This project is inspired by:
- **Dyna-Q** (Sutton, 1990): Original planning + learning architecture
- **MuZero** (DeepMind, 2019): World model for game playing
- **Dreamer** (Hafner et al., 2020): Learning in imagination

---

## Current Status

**Project Status**: ✅ **Phase 6 Complete (Expert Enhancements)** → ⏳ **Phase 5 In Progress (Testing & Visualization)**

**Implementation Progress:**
- ✅ Phases 1-4: **100% Complete** (~4,000 lines of code)
- ⏳ Phase 5: **~30% Complete** (testing and visualization in progress)
- ✅ Phase 6: **100% Complete** (expert enhancements)

**Recent Additions (Dec 2025):**
- ✅ Expert reward shaping with 7 positive rewards and 6 penalties
- ✅ Hindi and Hinglish language support for Indian market
- ✅ 3 new action strategies: `acknowledge_and_redirect`, `validate_then_offer`, `gentle_urgency`
- ✅ Action history tracking for context-aware decision making

**Next Steps:**
1. Run full training (75-200 episodes) to validate performance
2. Compare DQN vs DDQ performance with visualizations
3. Tune expert reward weights based on training results
4. A/B test Hindi vs English conversations

**Last Updated**: 2025-12-04
