# Phase 7: LLM + NLU Architecture

**Goal**: Replace hardcoded persona rules with realistic LLM-simulated debtors + deterministic NLU state extraction.

**Why**: Current persona rules are made-up. LLM has real knowledge of human behavior. NLU gives stable state extraction.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1: Domain Randomization (Sample Debtor Profile)                │   │
│  │  ├── personality: agreeableness=0.3, stability=0.6, assertiveness=0.7│   │
│  │  ├── situation: debt=$8500, overdue=90, job_loss=True                │   │
│  │  └── style: verbosity=0.4, directness=0.6                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 2: Agent Selects Action                                        │   │
│  │  State = [account_info, behavioral_signals (initially neutral)]      │   │
│  │  Action = Q-network picks: "empathetic_listening"                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 3: LLM Generates Agent Utterance                               │   │
│  │  "I understand this might be a difficult time. Could you tell me     │   │
│  │   more about your current situation?"                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 4: LLM Simulates Debtor Response (using profile)               │   │
│  │  Prompt: "You are a debtor with [profile]. Respond to agent."        │   │
│  │  Output: "Look, I lost my job 2 months ago. I want to pay but        │   │
│  │          I literally have no income right now."                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 5: NLU Pipeline (DETERMINISTIC)                                │   │
│  │  ├── VADER Sentiment: -0.25 (slightly negative but not hostile)      │   │
│  │  ├── Intent Classifier: "explaining_situation"                       │   │
│  │  ├── Signals: shared_hardship=True, mentioned_payment=False          │   │
│  │  └── Cooperation Score: 0.5 (willing to talk)                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 6: State Update & Reward                                       │   │
│  │  New State = [account_info, NLU_features]                            │   │
│  │  Reward = milestone_rewards(shared_situation=True) = +1.0            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                      ↓                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 7: Store Experience & Train                                    │   │
│  │  (state, action, reward, next_state) → Replay Buffer → Q-Learning   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Production Flow (Same State Representation!)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PRODUCTION ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TURN 0 (Before Call):                                                      │
│  ├── Load account from CRM: debt=$8500, overdue=90, calls=0                │
│  ├── Behavioral signals: neutral defaults (sentiment=0, coop=0.5)          │
│  └── Agent picks opening action based on account info                       │
│                                                                             │
│  TURN 1+:                                                                   │
│  ├── Real debtor speaks → Speech-to-Text                                   │
│  ├── NLU extracts: sentiment, cooperation, signals                          │
│  ├── State = [account_info, NLU_features] ← SAME AS TRAINING!              │
│  └── Agent picks next action using learned Q-values                         │
│                                                                             │
│  Key: Agent discovers who they're talking to through conversation           │
│       Just like a human debt collector would!                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Works for Unknown Personas

| Training | Production |
|----------|------------|
| Random profile → LLM generates text | Unknown person speaks |
| NLU extracts features from text | NLU extracts features from speech |
| Agent sees: [sentiment, coop, signals] | Agent sees: [sentiment, coop, signals] |
| **Same state representation!** | **Same state representation!** |

The agent learns to respond to **behavioral patterns**, not persona labels.
Domain randomization ensures training covers wide range of possible behaviors.

---

## Implementation Steps

### Progress Tracker

| Step | Description | Files | Status |
|------|-------------|-------|--------|
| 7.1 | Install NLU dependencies | requirements.txt | ✅ DONE |
| 7.2 | Create NLU state extractor | nlu/state_extractor.py (NEW) | ✅ DONE |
| 7.3 | Create domain randomizer | environment/domain_randomizer.py (NEW) | ✅ DONE |
| 7.4 | Update LLM prompts | llm/prompts.py | ✅ DONE (already parameterized) |
| 7.5 | Create NLU environment | environment/nlu_env.py (NEW) | ✅ DONE |
| 7.6 | Update state encoder | utils/state_encoder.py | ✅ SKIPPED (NLU env handles encoding) |
| 7.7 | Update config | config.py | ✅ DONE |
| 7.8 | Test NLU extraction | test script | ✅ DONE |
| 7.9 | Test full training loop | train.py | ✅ DONE |
| 7.10 | Update curriculum for domain randomization | curriculum_learning.py | ✅ DONE (difficulty sampling in train.py) |

### All Steps Complete! 🎉

---

## Step 7.1: Install NLU Dependencies

**File**: `requirements.txt`

Add:
```
vaderSentiment>=3.3.2
```

**Why VADER**:
- No API calls (free, fast)
- Deterministic (same text → same score)
- Good for conversational text
- No model download needed

---

## Step 7.2: Create NLU State Extractor

**File**: `nlu/state_extractor.py` (NEW)

```python
"""
NLU-based state extraction from debtor text responses.
Deterministic: same text → same features.
"""

class DebtorResponseAnalyzer:
    """Extract behavioral features from debtor text"""
    
    def analyze(self, text: str) -> dict:
        return {
            'sentiment': self._extract_sentiment(text),
            'intent': self._classify_intent(text),
            'cooperation_score': self._estimate_cooperation(text),
            'signals': self._extract_signals(text)
        }
    
    def _extract_sentiment(self, text: str) -> float:
        """VADER sentiment: -1.0 to 1.0"""
        # Uses vaderSentiment
        
    def _classify_intent(self, text: str) -> str:
        """Classify debtor intent: refusing, explaining, willing, committing, hostile, avoidant"""
        # Keyword-based classification
        
    def _estimate_cooperation(self, text: str) -> float:
        """0.0 (refusing) to 1.0 (fully cooperative)"""
        # Based on intent + sentiment + keywords
        
    def _extract_signals(self, text: str) -> dict:
        """Extract boolean signals"""
        # shared_situation, commitment_language, quit_signals, etc.
```

---

## Step 7.3: Create Domain Randomizer

**File**: `environment/domain_randomizer.py` (NEW)

```python
"""
Domain randomization for diverse debtor simulation.
Replaces 4 discrete personas with continuous parameter space.
"""

@dataclass
class DebtorProfile:
    # Personality (Big Five inspired)
    agreeableness: float      # 0-1
    emotional_stability: float # 0-1
    assertiveness: float       # 0-1
    
    # Situation
    debt_amount: float         # $500 - $50,000
    days_overdue: int          # 30 - 365
    financial_stress: float    # 0-1
    life_event: str            # none, job_loss, medical, divorce
    call_history: int          # 0, 1, 2+ previous calls
    
    # Communication style
    verbosity: float           # 0-1 (terse to verbose)
    directness: float          # 0-1 (evasive to direct)

class DomainRandomizer:
    """Sample random debtor profiles for training diversity"""
    
    def sample(self) -> DebtorProfile:
        """Sample a random debtor profile"""
        
    def to_prompt_context(self, profile: DebtorProfile) -> str:
        """Convert profile to LLM prompt context"""
```

---

## Step 7.4: Update LLM Prompts

**File**: `llm/prompts.py`

**Change**: Replace discrete persona prompts with parameterized prompts.

Before:
```python
"You are an ANGRY debtor who is hostile and resistant..."
```

After:
```python
def build_debtor_prompt(profile: DebtorProfile) -> str:
    return f"""You are a person with:
- ${profile.debt_amount:.0f} debt, {profile.days_overdue} days overdue
- Financial stress level: {profile.financial_stress:.0%}
- Life situation: {profile.life_event or 'none'}
- Personality: {'agreeable' if profile.agreeableness > 0.5 else 'disagreeable'}, 
  {'calm' if profile.emotional_stability > 0.5 else 'emotional'}
- Communication: {'direct' if profile.directness > 0.5 else 'evasive'}

Respond naturally to the debt collector. Stay in character.
"""
```

---

## Step 7.5: Integrate NLU into Environment

**File**: `environment/debtor_env.py`

**Changes**:
1. Import `DebtorResponseAnalyzer` and `DomainRandomizer`
2. In `reset()`: Sample new debtor profile
3. In `step()`: Use NLU to extract state from LLM response
4. Remove old persona-based state updates

Key change in `_generate_debtor_response()`:
```python
# OLD: Parse LLM JSON for sentiment/cooperation (unreliable)
# NEW: 
response_text = self.llm_client.generate_debtor_response(...)
nlu_features = self.nlu_analyzer.analyze(response_text)
self._update_state_from_nlu(nlu_features)
```

---

## Step 7.6: Update State Encoder

**File**: `utils/state_encoder.py`

**Changes**:
- Add account context features (debt_amount, days_overdue, call_count)
- NLU features replace persona-based features
- Keep state_dim at 18 or adjust slightly

---

## Step 7.7: Update Config

**File**: `config.py`

**Changes**:
- Add NLU config section
- Add domain randomization bounds
- Update STATE_DIM if needed

---

## Effort Estimate

| Component | New Files | Modified Files | Lines of Code | Complexity |
|-----------|-----------|----------------|---------------|------------|
| NLU Extractor | 1 | 0 | ~150 | Medium |
| Domain Randomizer | 1 | 0 | ~100 | Low |
| Prompt Updates | 0 | 1 | ~50 | Low |
| Environment Integration | 0 | 1 | ~100 | Medium |
| State Encoder | 0 | 1 | ~30 | Low |
| Config | 0 | 1 | ~20 | Low |
| **Total** | **2 new** | **4 modified** | **~450** | **Medium** |

**Time estimate**: 2-3 hours of focused work

---

## Order of Implementation

```
7.1 Dependencies → 7.2 NLU Extractor → 7.3 Domain Randomizer → 
7.4 Prompts → 7.5 Environment → 7.6 State Encoder → 7.7 Config →
7.8 Test NLU → 7.9 Test Training → 7.10 Update Curriculum
```

Each step is independently testable. We can verify each component works before moving to the next.

---

## Success Criteria

After implementation:
- [ ] Training runs with LLM-simulated debtors
- [ ] NLU extracts consistent features from debtor text
- [ ] Domain randomization creates diverse debtor profiles
- [ ] Agent learns policies that generalize across profiles
- [ ] Same state representation works for production

---

## Current Status

**Phase 7 Start Date**: 2025-12-02
**Current Step**: 7.1 (Dependencies)

Let's begin! 🚀
