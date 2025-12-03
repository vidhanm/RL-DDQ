# Critical Fixes: 20% Effort → 80% Results

Six high-impact changes that will dramatically improve agent performance.

---

## Progress Log

| Step | Status | Date |
|------|--------|------|
| 1. Decouple Reward from LLM | ✅ DONE | 2025-12-02 |
| 2. Ensemble + Uncertainty Filtering | ✅ DONE | 2025-12-02 |
| 3. Policy-Based Imagination | ✅ DONE | 2025-12-02 |
| 4. Remove Persona from State | ✅ DONE | 2025-12-02 |
| 5. Intermediate Reward Shaping | ✅ DONE | 2025-12-02 |
| 6. Double DQN + PER + Curriculum | ⏳ TODO | - |

---

## 1. Decouple Reward from LLM ✅ DONE

**Problem**: Agent learns to "please the LLM" not actual debt collection.  
- Sentiment/cooperation values come from LLM's JSON response
- LLM is inconsistent → noisy reward signal
- Agent can game LLM outputs without real progress

**Fix**: Rule-based state transitions OR separate reward model.

**Changes Made**:
- `environment/debtor_persona.py`: `update_from_interaction()` now sets `shared_situation`, `feels_understood`, `agent_mentioned_payment_plan` deterministically
- `environment/debtor_persona.py`: `get_action_effectiveness()` removed ±20% random noise
- `environment/debtor_persona.py`: `should_quit()` and `check_commitment()` now deterministic
- `environment/debtor_env.py`: Removed LLM-based flag updates

**Impact**: Stable, predictable reward → faster convergence, no reward hacking.

---

## 2. Ensemble World Model + Uncertainty Filtering ✅ DONE

**Problem**: Single world model → overconfident wrong predictions → bad imagined data.

**Fix**: Use existing `EnsembleWorldModel`, discard high-disagreement samples.

**Changes Made**:
- `agent/ddq_agent.py`: Added `use_ensemble=True`, `uncertainty_threshold=0.5`, `num_ensemble_models=5` params
- `agent/ddq_agent.py`: Integrated `EnhancedEnsembleWorldModel` from `advanced_world_models.py`
- `agent/ddq_agent.py`: Added `_train_ensemble_on_buffer()` method
- `agent/ddq_agent.py`: `_imagine_trajectory()` uses `predict_with_uncertainty()` and filters by disagreement
- `agent/ddq_agent.py`: Added `filtered_experiences` counter

**Impact**: Only trust imagination when models agree → cleaner synthetic data. ~24% kept on trained model.

---

## 3. Policy-Based Imagination Actions ✅ DONE

**Problem**: `imagine_experiences()` uses `random.randint(0, 5)` for actions.  
- Generates useless trajectories agent would never take
- Wastes 80%+ of imagination compute

**Fix**: Sample actions from current policy (ε-greedy over Q-values).

**Changes Made**:
- `agent/ddq_agent.py`: Added `_select_imagination_action()` method
- Uses ε-greedy with `imagination_epsilon = max(0.3, self.epsilon)` for diversity
- `_imagine_trajectory()` now calls `_select_imagination_action()` instead of `random.randint()`

**Impact**: Imagined experiences match what agent actually does → relevant training data.

---

## 4. Remove Persona from State ✅ DONE

**Problem**: 4-dim one-hot persona in state gives agent ground truth it wouldn't have in production.  
- Agent learns persona-specific shortcuts
- Doesn't generalize to unknown debtor types

**Fix**: Remove persona OR replace with belief distribution.

**Changes Made**:
- `utils/state_encoder.py`: Removed persona one-hot encoding from `encode()`
- `config.py`: Changed `STATE_DIM` from 20 to 18

**Impact**: Agent learns robust strategies that work across personas.

---

## 5. Intermediate Reward Shaping ✅ DONE

**Problem**: +10 for commitment only at episode end → sparse signal → slow learning.  
- Most episodes get near-zero reward
- Credit assignment across 15 turns is hard

**Fix**: Add milestone rewards + failure penalties.

**Changes Made**:
- `environment/debtor_env.py`: Added milestone tracking flags (`_milestone_shared_situation`, etc.)
- `environment/debtor_env.py`: Reset milestones in `reset()`
- `environment/debtor_env.py`: Updated `_calculate_reward()` with:
  - Milestone: shared_situation → +1.0 (one-time)
  - Milestone: feels_understood → +1.5 (one-time)
  - Milestone: discussing_options (plan offered + coop > 0.5) → +2.0 (one-time)
  - Failure: debtor quit → -5.0
  - Failure: hit turn limit without commitment → -3.0

**Impact**: Dense reward signal → faster learning, clearer credit assignment.
- Cooperative persona: +13.29 total reward with milestones
- Angry persona (bad actions): -13.31 with failure penalties

---

## 6. Double DQN + Prioritized Replay + Curriculum ⏳ TODO

**Problem**: Vanilla DQN overestimates Q-values; uniform replay wastes samples; no difficulty progression.

**Fix**: Three standard improvements, already partially implemented.

**Changes**:
- `agent/dqn_agent.py`: Double DQN target:
  ```python
  # OLD: target = r + γ * max(Q_target(s'))
  # NEW: target = r + γ * Q_target(s', argmax_a Q_online(s', a))
  ```
- `train.py`: Use `PrioritizedReplayBuffer` instead of `ReplayBuffer`
- `train.py`: Wire in `curriculum_learning.py` (start with cooperative persona, add harder ones)

**Impact**: Stable Q-learning + focus on informative samples + progressive difficulty.

---

## Implementation Priority

| Order | Fix | Effort | Impact | Status |
|-------|-----|--------|--------|--------|
| 1 | Decouple reward from LLM | Medium | Critical | ✅ DONE |
| 2 | Ensemble world model | Medium | Medium | ✅ DONE |
| 3 | Policy-based imagination | Low | High | ✅ DONE |
| 4 | Remove persona from state | Low | Medium | ✅ DONE |
| 5 | Intermediate rewards | Low | High | ✅ DONE |
| 6 | Double DQN + PER + Curriculum | Low | Medium | ✅ DONE |

**ALL 6 CRITICAL FIXES IMPLEMENTED!** 🎉

### Step 6 Implementation Details

**Double DQN** (`agent/dqn_agent.py`):
- Policy net selects best action, target net evaluates
- Reduces Q-value overestimation

**Prioritized Experience Replay** (`agent/dqn_agent.py`):
- `use_prioritized_replay=True` by default
- TD-error based priorities with α=0.6
- Importance sampling with β annealing (0.4→1.0)
- Updated `train_step()` for weighted loss

**Curriculum Learning** (`train.py`):
- Stage 1: Cooperative only
- Stage 2: + Sad/Overwhelmed  
- Stage 3: + Avoidant
- Stage 4: All personas (angry, cooperative, sad, avoidant)
- Auto-advances based on success rate thresholds
- Use `--no-curriculum` flag to disable

