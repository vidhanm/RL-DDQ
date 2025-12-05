# Self-Improving Debt Collection AI Agent

A reinforcement learning voice agent that learns optimal debt collection strategies using **DDQ (Dyna-style Data-efficient Q-learning)** with LLM integration, adversarial self-play, and multilingual support.

## 🚀 Quick Start

```bash
# Install
pip install -r requirements.txt

# Set API key
export OPENAI_API_KEY="your-key"  # or set NVIDIA_API_KEY

# Train (DDQ recommended)
python scripts/train.py --algorithm ddq --episodes 100

# Or DQN baseline
python scripts/train.py --algorithm dqn --episodes 100

# Run web UI
python -m uvicorn web.backend.main:app --reload
# Open http://localhost:8000
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **DDQ Algorithm** | 5-10x faster learning via world model imagination |
| **9 Strategies** | Empathy, plans, settlements, gentle urgency, etc. |
| **Adversarial Training** | Collector vs resistant debtor self-play |
| **Multilingual** | English, Hindi, Hinglish support |
| **Domain Randomization** | Millions of unique debtor profiles |
| **Expert Rewards** | Encodes debt collection best practices |
| **Web Dashboard** | Real-time training visualization |

### 🎲 How Domain Randomization Works

Instead of training on 4 fixed debtor types, we **randomly generate** personality traits for each conversation:

```
Each debtor = random mix of:
├── Agreeableness (0-100%)     → How cooperative?
├── Emotional Stability (0-100%) → Calm or reactive?
├── Financial Stress (0-100%)    → How desperate?
└── Life Event (job loss, medical, divorce, none)
```

**Why?** The agent learns to handle *any* debtor, not just 4 scripted ones. Like training a driver on random roads instead of the same 4 routes.


## 🏗️ Architecture

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Debtor     │───▶│  NLU State   │───▶│  DDQ Agent   │
│ (LLM/Adver.) │    │  Extraction  │    │  (Q + World) │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
┌──────────────┐    ┌──────────────┐           ▼
│   Response   │◀───│  LLM Text    │◀───  Strategy
│   (Voice)    │    │  Generation  │      Selection
└──────────────┘    └──────────────┘
```

## 📁 Project Structure

```
RL DDQ/
├── src/
│   ├── agent/           # DDQ, DQN, Adversarial agents
│   ├── environment/     # NLU env, SelfPlay env
│   ├── llm/             # NVIDIA/OpenAI clients, prompts
│   ├── nlu/             # State extraction (sentiment, intent)
│   └── utils/           # Replay buffer, encoders
├── scripts/
│   ├── train_selfplay.py    # Adversarial training
│   └── evaluate.py          # Evaluation & demo
├── web/
│   ├── backend/         # FastAPI + WebSocket
│   └── frontend/        # Dashboard UI
├── docs/
│   └── RESEARCH_INSIGHTS.md  # 42 papers, 6 topics
└── data/                # Checkpoints, logs
```

## 🎮 Action Space (9 Strategies)

1. **Empathetic Listening** - Show understanding
2. **Ask About Situation** - Gather context  
3. **Firm Reminder** - Professional assertive
4. **Offer Payment Plan** - Installments
5. **Propose Settlement** - Reduced amount
6. **Hard Close** - Urgency with consequences
7. **Acknowledge & Redirect** - Handle venting
8. **Validate Then Offer** - Deep empathy → solution
9. **Gentle Urgency** - "Protect your credit score"

## ⚔️ Adversarial Self-Play

Train robust collectors against 7 adversary resistance strategies:

| Adversary | Tactic |
|-----------|--------|
| Aggressive | "Stop calling! This is harassment!" |
| Evasive | "Let me think about it..." |
| Emotional | "I can't take this anymore..." |
| Negotiate Hard | "90% off or nothing" |
| Partial Cooperate | Fake interest, no commitment |
| Stall | "Send documents first" |
| Dispute | "Prove this debt is mine" |

```bash
# Full adversarial training
python scripts/train_selfplay.py --generations 20 --episodes 100 --use-llm
```

## 📊 Web Dashboard

| Page | URL | Features |
|------|-----|----------|
| Home | `/` | Project overview |
| Training | `/train` | Train agents, view metrics |
| Evaluation | `/evaluate` | Test conversations |
| **Adversarial Arena** | `/adversarial` | Live self-play battles |

## 🔬 Research

Comprehensive research across 6 topics (42 papers):

1. Model-Based RL & Planning
2. Task-Oriented Dialogue RL  
3. Self-Improvement & Meta-Learning
4. Adversarial Training & Robustness
5. Efficient RL / Few-Shot Learning
6. Voice/Spoken Dialogue Systems

**See [docs/RESEARCH_INSIGHTS.md](docs/RESEARCH_INSIGHTS.md) for actionable insights.**

## 🛠️ Configuration

Edit `src/config.py`:

```python
# Key settings
STATE_DIM = 12           # NLU features
ACTION_DIM = 9           # Strategies
IMAGINATION_FACTOR = 5   # DDQ imagination multiplier
LANGUAGE = "english"     # or "hindi", "hinglish"
```

## 📈 Development Status

| Phase | Status |
|-------|--------|
| 1-4: Core DDQ | ✅ Complete |
| 5: Evaluation | ✅ Complete |
| 6: Expert Enhancements | ✅ Complete |
| 7: Adversarial Self-Play | ✅ Complete |
| **Research** | ✅ 42 papers reviewed |

## 📚 References

- **DDQ**: Peng et al. "Deep Dyna-Q" (2018)
- **DreamerV3**: Hafner et al. "Mastering Diverse Domains" (2023)
- **Research**: See [RESEARCH_INSIGHTS.md](docs/RESEARCH_INSIGHTS.md)

---

**Last Updated**: December 5, 2025 | **Status**: ✅ Phase 7 Complete
