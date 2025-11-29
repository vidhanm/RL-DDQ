# DDQ Debt Collection - Quick Concepts Guide

## Core Concepts

**Episode** = 1 full conversation (8-15 turns)
**Experience** = 1 turn within conversation = (state, action, reward, next_state)
**State** = 20 exact numbers (not ranges!) like [0.2, -0.3, 0.6, ...]

Example: 3 episodes = 21 experiences (3+3+15 turns)

---

## How Agent Decides Actions

```
Debtor speaks → Extract features → State vector (20 numbers) → Neural Net → Q-values → Pick best action
```

**Example:**
```
Debtor: "I lost my job! But maybe we can work it out..."
         ↓
State: [turn=0.2, sentiment=-0.3, cooperation=0.6, engagement=0.8, ...]
         ↓
DQN Neural Net processes
         ↓
Q-values: [5.2, 4.1, 2.3, 8.7, 6.4, -1.2]  ← payment_plan = highest!
         ↓
Action: "offer_payment_plan"
```

---

## Key Insight: State = Agent's "Memory"

State already contains debtor's response! Agent DOES respond to what debtor says.

```
Turn 1: State₀ [angry] → Agent: firm → Debtor: "Screw you!" → State₁ [VERY angry]
Turn 2: State₁ [VERY angry] → Agent: empathy → Debtor: "Fine..." → State₂ [less angry]
```

Agent chooses action based on State, which encodes debtor's last response.

---

## DDQ Imagination (How 500 → 3000 experiences)

**NOT duplicating calls!** Imagining "what if" alternatives:

```
Real experience: State₁ → chose "empathetic" → Reward +2

Imagination (K=5 per real state):
  State₁ → "firm_reminder" → World Model predicts → -3 reward
  State₁ → "payment_plan" → World Model predicts → +4 reward
  State₁ → "hard_close" → World Model predicts → -5 reward
  ... (5 imagined per real)

500 real × 5 = 2,500 imagined
Total: 3,000 experiences! 6x more training data, same LLM cost!
```

World Model = Neural net that learned debtor behavior patterns (NO LLM calls)

---

## LLM Usage

**LLM used ALWAYS during conversations (training AND testing):**
- Generate agent messages: "I understand this is difficult..."
- Generate debtor responses: "I lost my job..."

**LLM NOT used for:**
- Feature extraction (uses fast rules)
- DQN training (neural net only)
- Imagination (world model only)

---

## Feature Extraction: Hybrid Approach

**Your project uses BOTH smartly:**

**Method 1: Rule-Based (Fast, Free)** ✅ CURRENT
```python
if "lost job" in text: hardship = 1.0
if "hate" in text: sentiment -= 0.2
# Takes <1ms, costs $0
```

**Method 2: LLM-Based (Accurate, Slow)** 🔧 OPTIONAL
```python
LLM: "Analyze sentiment: -0.3"
LLM: "Cooperation level: 0.6"
# Takes 1-2s, costs $$
```

**Best Practice (What you do):**
- LLM for realistic conversation text ✅
- Rules for state updates ✅
- Best of both: quality + speed + low cost

---

## Training Flow Summary

```
EPISODE 1-100 (Real conversations with LLM)
  → Store 500 real experiences

EVERY 5 EPISODES (DDQ imagination)
  → Train World Model on 500 real
  → Generate 2,500 imagined (K=5)
  → Total: 3,000 experiences

DQN TRAINING
  Sample batch: 75% real (24) + 25% imagined (8) = 32 total
  Train neural net: Learn Q(state, action) = expected reward
  Agent gets smarter!
```

---

## Numbers = Exact Values, NOT Ranges

**NOT ranges:**
```
sentiment: "slightly negative" ❌
cooperation: "medium" ❌
```

**Exact numbers:**
```
sentiment: -0.347 ✅
cooperation: 0.612 ✅
```

Why? Neural networks need precise values to learn subtle patterns.

---

## Quick Facts

- **Need 500 experiences** before NN starts training
- **Episode lengths vary:** 3-15 turns (avg ~7)
- **Your 3 episodes:** 21 experiences = need ~71 more episodes to start learning
- **States:** 20 continuous numbers, updated every turn
- **Actions:** 6 high-level strategies (empathy, firm, payment_plan, etc.)
- **Personas:** 4 types (angry, cooperative, sad, avoidant)

---

## To See Real Learning

Run: `python train.py --algorithm ddq --episodes 75 --neptune`

Around episode 40-50:
- ✅ World model starts training
- ✅ Imagination generates synthetic data
- ✅ DQN loss appears (was 0.0000)
- ✅ Agent learns patterns!

---

**Summary:** RL agent learns WHICH action to choose. LLM generates the actual words. State = numbers encoding conversation. DDQ = 6x more training data via imagination. Your hybrid approach = industry best practice! 🎯
