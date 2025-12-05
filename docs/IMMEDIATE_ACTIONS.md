# Immediate Action Items

**Source**: [RESEARCH_INSIGHTS.md](./RESEARCH_INSIGHTS.md) (42 papers, 6 topics)  
**Created**: December 5, 2025  
**Last Updated**: December 5, 2025

---

## Quick Status

| Category | Total | Done | In Progress | TODO |
|----------|-------|------|-------------|------|
| Immediate | 18 | 0 | 0 | 18 |
| Medium-Term | 14 | 0 | 0 | 14 |
| Long-Term | 7 | 0 | 0 | 7 |

---

## 🔥 Immediate Actions (Do Now)

### 1. Multi-Step Look-Ahead
**Source**: Topic 1 (Model-Based RL)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Use world model to simulate 2-3 future turns before choosing action.

**Brief Plan**:
1. In `ddq_agent.py`, modify `select_action()`
2. For each possible action, use world model to predict next 2 states
3. Sum discounted Q-values across trajectory
4. Pick action with best cumulative value

---

### 2. Conversation Phase in State
**Source**: Topic 1 (Model-Based RL)  
**Status**: `[x] DONE`  
**Impact**: Medium | **Effort**: Low

**What**: Add explicit phase indicator (opening/discovery/negotiation/commitment) to state.

**Brief Plan**:
1. Add `conversation_phase` to `NLUFeatures` ✅
2. Classify based on turn number + detected intents ✅
3. 5 phases: opening, discovery, negotiation, commitment, hostile ✅

---

### 3. Commitment Probability Prediction
**Source**: Topic 1 (Model-Based RL)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Train world model to predict commitment probability, not just next state.

**Brief Plan**:
1. Add output head to `WorldModel`
2. Train on historical commitment outcomes
3. Use in action selection (prefer actions → higher commit prob)

---

### 4. Step-by-Step Rewards
**Source**: Topic 2 (Dialogue RL)  
**Status**: `[x] DONE`  
**Impact**: High | **Effort**: Low

**What**: Give progressive rewards for partial progress, not just final outcome.

**Brief Plan**:
1. In reward function, add: ✅
   - Sentiment improvement: +0.8 (first positive turn) ✅
   - Cooperation increase: +1.0 (crossed 50% threshold) ✅
   - Question answered: +0.5 ✅
2. Already had 3 milestones, added 3 more = 6 total ✅

---

### 5. Negotiation Signals in State
**Source**: Topic 2 (Dialogue RL)  
**Status**: `[ ] TODO`  
**Impact**: Medium | **Effort**: Low

**What**: Detect negotiation phase (offer made, counter-offer, close attempt).

**Brief Plan**:
1. Add `negotiation_stage` to NLU output
2. Track: `none → offer_made → counter → agreement`
3. Use as state feature

---

### 6. Hindsight Regeneration
**Source**: Topic 2 (Dialogue RL)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Learn from failed conversations by asking "what should we have done?".

**Brief Plan**:
1. Collect failed episodes
2. Use LLM to suggest: "At turn 3, empathy would have worked better"
3. Create synthetic positive examples
4. Add to replay buffer

---

### 7. Auto-Curriculum
**Source**: Topic 3 (Self-Improvement)  
**Status**: `[x] DONE`  
**Impact**: High | **Effort**: Medium

**What**: Automatically adjust debtor difficulty based on agent performance.

**Brief Plan**:
1. Track success rate per difficulty level ✅
2. If success > 80%, increase difficulty ✅
3. If success < 40%, decrease difficulty ✅
4. Added `DifficultyAutoCurriculum` class ✅

---

### 8. In-Context Adaptation
**Source**: Topic 3 (Self-Improvement)  
**Status**: `[ ] TODO`  
**Impact**: Medium | **Effort**: Low

**What**: Include recent successful strategies in LLM prompt for similar debtors.

**Brief Plan**:
1. Store successful (state, action) pairs by debtor profile
2. When generating response, include: "In similar situations, X worked well"
3. Let LLM use this context

---

### 9. RISE Self-Improvement Loop
**Source**: Topic 3 (Self-Improvement)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: High

**What**: Agent reviews own failures and systematically improves.

**Brief Plan**:
1. Collect failure cases
2. Agent analyzes: "Why did this fail?"
3. Generate improved strategies
4. Train on corrected data

---

### 10. Diverse Adversary Pool
**Source**: Topic 4 (Adversarial)  
**Status**: `[/] PARTIAL` (have 7 strategies, need more personas)  
**Impact**: High | **Effort**: Low

**What**: Maintain pool of diverse adversary agents (Hostile Harry, Evasive Eva, etc.).

**Brief Plan**:
1. Create named adversary profiles in `opponent_pool.py`
2. Each with distinct resistance style
3. Sample from pool during training

---

### 11. QARL Curriculum (Weak → Strong)
**Source**: Topic 4 (Adversarial)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Start with weak adversary, gradually increase strength.

**Brief Plan**:
1. Add `adversary_strength` parameter (0.0 → 1.0)
2. Low strength = easy resistance, high = sophisticated tactics
3. Increase over training generations

---

### 12. Opponent Modeling ⭐
**Source**: Topic 4 (Adversarial)  
**Status**: `[x] DONE`  
**Impact**: Very High | **Effort**: Medium

**What**: After 2-3 turns, infer debtor type and adapt strategy.

**Brief Plan**:
1. Create `DebtorTypeClassifier` ✅
2. Input: first 2-3 turns of conversation ✅
3. Output: predicted type (Hostile, Cooperative, Evasive, etc.) ✅
4. Add inferred type to state representation ✅
5. Agent learns type-specific strategies ✅

---

### 13. Semantic Caching for LLM 💰
**Source**: Topic 5 (Efficiency)  
**Status**: `[x] DONE`  
**Impact**: Very High (50-70% cost reduction) | **Effort**: Medium

**What**: Cache LLM responses by semantic similarity.

**Brief Plan**:
1. Create `SemanticCache` class ✅
2. Embed prompts, find similar cached responses ✅
3. If similarity > 0.85, return cached ✅
4. Huge cost savings during training ✅

---

### 14. Prioritized Replay (Commitment Focus)
**Source**: Topic 5 (Efficiency)  
**Status**: `[x] DONE`  
**Impact**: Medium | **Effort**: Low

**What**: Prioritize rare commitment moments in replay buffer.

**Brief Plan**:
1. Already have PER by TD-error ✅
2. Add bonus priority for: ✅
   - Commitment attempts (3x) ✅
   - Hostile situations (2x) ✅
   - Successful de-escalations (2.5x) ✅

---

### 15. Imagination Augmentation
**Source**: Topic 5 (Efficiency)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Generate counterfactual experiences from world model.

**Brief Plan**:
1. For each real experience:
   - "What if different action?"
   - "What if debtor was more hostile?"
2. Generate synthetic experiences
3. 1 real → 10-20 imagined experiences

---

### 16. Speech Emotion Recognition
**Source**: Topic 6 (Voice)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: High

**What**: Detect emotion from voice (anger, sadness) not just text.

**Brief Plan**:
1. Add audio feature extraction (pitch, energy, rate)
2. Train or use pretrained SER model
3. Fuse with text-based sentiment
4. Detect hidden anger, sarcasm

---

### 17. Smart Turn-Taking
**Source**: Topic 6 (Voice)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Know when debtor has finished speaking (VAD + semantic).

**Brief Plan**:
1. Voice Activity Detection for silence
2. Semantic completeness check
3. Avoid interrupting, avoid awkward pauses
4. Target 300-800ms response delay

---

### 18. Latency Optimization
**Source**: Topic 6 (Voice)  
**Status**: `[ ] TODO`  
**Impact**: High | **Effort**: Medium

**What**: Reduce response time to <500ms.

**Brief Plan**:
1. Streaming ASR (process chunks, don't wait)
2. Streaming TTS (start speaking while generating)
3. Use faster LLM for voice (Gemini Flash)
4. Semantic caching (see #13)

---

## ⏳ Medium-Term Actions (Do Soon)

| # | Action | Source | Status |
|---|--------|--------|--------|
| 1 | Prediction-reliability weighting | Topic 1 | `[ ] TODO` |
| 2 | Symlog reward normalization | Topic 1 | `[ ] TODO` |
| 3 | Future-oriented empathy reward | Topic 2 | `[ ] TODO` |
| 4 | Offline RL training pipeline | Topic 2 | `[ ] TODO` |
| 5 | Expert preferences for DPO | Topic 2 | `[ ] TODO` |
| 6 | Self-refine prompting | Topic 3 | `[ ] TODO` |
| 7 | EWC for continual learning | Topic 3 | `[ ] TODO` |
| 8 | SPIN-style training | Topic 3 | `[ ] TODO` |
| 9 | Balanced domain randomization | Topic 4 | `[ ] TODO` |
| 10 | Worst-case reward (WocaR) | Topic 4 | `[ ] TODO` |
| 11 | LLM-guided reward shaping | Topic 5 | `[ ] TODO` |
| 12 | Transfer learning | Topic 5 | `[ ] TODO` |
| 13 | State abstraction | Topic 5 | `[ ] TODO` |
| 14 | Emotional TTS | Topic 6 | `[ ] TODO` |

---

## 📅 Long-Term Actions (Do Later)

| # | Action | Status |
|---|--------|--------|
| 1 | Transformer-based world model (UniZero) | `[ ] TODO` |
| 2 | Graph-structured policy (GNN) | `[ ] TODO` |
| 3 | Full MAML online adaptation | `[ ] TODO` |
| 4 | Full NFSP Nash equilibrium | `[ ] TODO` |
| 5 | Full offline RL (CQL + MAML) | `[ ] TODO` |
| 6 | Full-duplex voice (barge-in) | `[ ] TODO` |
| 7 | End-to-end Audio-LLM | `[ ] TODO` |

---

## 📋 Implementation Log

| Date | Action | Status Change | Notes |
|------|--------|---------------|-------|
| 2025-12-05 | Created file | - | Extracted from RESEARCH_INSIGHTS.md |

---

## 🎯 Suggested Priority Order

Based on impact/effort ratio:

1. **#13 Semantic Caching** - Huge cost savings, moderate effort
2. **#12 Opponent Modeling** - High impact on strategy
3. **#4 Step-by-Step Rewards** - Quick win, low effort
4. **#14 Prioritized Replay** - Easy enhancement to existing PER
5. **#7 Auto-Curriculum** - Already have basic version, just make adaptive
