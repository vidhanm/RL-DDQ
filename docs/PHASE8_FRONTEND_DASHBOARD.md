# Phase 8: Frontend Dashboard (Train + Evaluate + Test)

**Goal**: Build a multi-page web dashboard to train, evaluate, and test the DDQ agent via browser.

**Why**: CLI is fine for development, but a visual interface makes it easier to monitor training, compare results, and eventually do live voice testing.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRONTEND DASHBOARD                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                │
│  │   TRAIN PAGE   │  │ EVALUATE PAGE  │  │   TEST PAGE    │                │
│  ├────────────────┤  ├────────────────┤  ├────────────────┤                │
│  │ • Algorithm    │  │ • Checkpoint   │  │ Coming Soon:   │                │
│  │ • Episodes     │  │ • Num Episodes │  │ LiveKit Voice  │                │
│  │ • Use LLM      │  │ • Run Eval     │  │ Integration    │                │
│  │ • Start/Stop   │  │                │  │                │                │
│  ├────────────────┤  ├────────────────┤  └────────────────┘                │
│  │ LIVE UPDATES:  │  │ RESULTS:       │                                    │
│  │ • Episode #    │  │ • Success Rate │                                    │
│  │ • Reward Chart │  │ • Avg Reward   │                                    │
│  │ • Live Dialog  │  │ • Conversations│                                    │
│  └────────────────┘  └────────────────┘                                    │
│                                                                             │
│         ↓ WebSocket              ↓ REST API                                │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      FASTAPI BACKEND                                  │  │
│  │  /api/training/start (WebSocket)  |  /api/evaluate/run (POST)        │  │
│  │  /api/training/stop               |  /api/models/list                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Progress Tracker

| Step | Description | Files | Status |
|------|-------------|-------|--------|
| 8.0 | Delete old checkpoints & commit Phase 7 | checkpoints/ | ✅ DONE |
| 8.1 | Update backend dependencies for NLU | backend/dependencies.py | ✅ DONE |
| 8.2 | Update conversation router for NLU | backend/routers/conversation.py | ✅ DONE |
| 8.3 | Update schemas for NLU state | backend/schemas/models.py | ✅ DONE |
| 8.4 | Create evaluate router | backend/routers/evaluate.py (NEW) | ✅ DONE |
| 8.5 | Add training WebSocket | backend/routers/training.py | ✅ DONE |
| 8.6 | Update main.py | backend/main.py | ✅ DONE |
| 8.7 | Create frontend page structure | frontend/*.html | ✅ DONE |
| 8.8 | Build Train page | frontend/train.html, js/train.js | ✅ DONE |
| 8.9 | Build Evaluate page | frontend/evaluate.html, js/evaluate.js | ✅ DONE |
| 8.10 | Create Test placeholder | frontend/test.html | ✅ DONE |
| 8.11 | Update evaluate.py CLI | evaluate.py | ✅ DONE |
| 8.12 | Test full flow | - | ⬜ TODO |

---

## Step 8.0: Cleanup & Commit Phase 7

**Actions:**
1. Delete old 18-dim checkpoints (incompatible with NLU)
2. Commit Phase 7 work

**Files to delete:**
- `checkpoints/dqn_*.pt` (old 18-dim)
- `checkpoints/ddq_*.pt` (old 18-dim)
- `checkpoints/*.json` (old format)

**Commit message:** `feat: complete Phase 7 NLU architecture`

---

## Step 8.1: Update Backend Dependencies

**File:** `backend/dependencies.py`

**Changes:**
- Replace `EnvironmentConfig.STATE_DIM` → `EnvironmentConfig.NLU_STATE_DIM`
- Update both `load_model()` and `preload_models()` functions

**Commit message:** `fix: backend uses NLU_STATE_DIM (19)`

---

## Step 8.2: Update Conversation Router

**File:** `backend/routers/conversation.py`

**Changes:**
- Import `NLUDebtCollectionEnv` instead of `DebtCollectionEnv`
- Update `format_state()` for NLU fields
- Remove persona selection (domain randomization handles diversity)

**Commit message:** `feat: conversation router uses NLU environment`

---

## Step 8.3: Update Schemas

**File:** `backend/schemas/models.py`

**Changes:**
- Update `StateDisplay` with NLU fields (intent, signals)
- Add `StartTrainingRequest` schema
- Add `TrainingProgressUpdate` schema for WebSocket
- Add `EvaluateRequest` and `EvaluateResponse` schemas

**Commit message:** `feat: add NLU state + training schemas`

---

## Step 8.4: Create Evaluate Router

**File:** `backend/routers/evaluate.py` (NEW)

**Endpoints:**
- `POST /api/evaluate/run` - Run evaluation on checkpoint
- `GET /api/evaluate/checkpoints` - List available checkpoints

**Commit message:** `feat: add evaluate router`

---

## Step 8.5: Add Training WebSocket

**File:** `backend/routers/training.py`

**Changes:**
- Add `WebSocket /api/training/ws` for live progress
- Add `POST /api/training/start` to trigger training
- Add `POST /api/training/stop` to cancel training

**WebSocket messages:**
```json
{"type": "episode", "episode": 1, "reward": 5.2, "success": true}
{"type": "dialogue", "agent": "...", "debtor": "..."}
{"type": "complete", "total_episodes": 100, "success_rate": 0.65}
```

**Commit message:** `feat: training WebSocket for live updates`

---

## Step 8.6: Update main.py

**File:** `backend/main.py`

**Changes:**
- Import and register `evaluate_router`
- Add routes for new pages (`/train`, `/evaluate`, `/test`)
- Update startup to use NLU_STATE_DIM

**Commit message:** `feat: register evaluate router + page routes`

---

## Step 8.7: Frontend Page Structure

**Files:**
- `frontend/index.html` - Dashboard with navigation
- `frontend/train.html` - Training page
- `frontend/evaluate.html` - Evaluation page  
- `frontend/test.html` - Test placeholder
- `frontend/css/styles.css` - Update with nav styles

**Commit message:** `feat: multi-page frontend structure`

---

## Step 8.8: Build Train Page

**Files:**
- `frontend/train.html` - Form + progress display
- `frontend/js/train.js` - WebSocket logic

**Features:**
- Algorithm selector (DQN/DDQ)
- Episode count input
- Use LLM checkbox
- Start/Stop buttons
- Live episode counter
- Reward chart (Chart.js)
- Live dialogue feed

**Commit message:** `feat: train page with live updates`

---

## Step 8.9: Build Evaluate Page

**Files:**
- `frontend/evaluate.html` - Form + results
- `frontend/js/evaluate.js` - API calls

**Features:**
- Checkpoint selector dropdown
- Episode count input
- Run Evaluation button
- Results display (success rate, avg reward)
- Sample conversations viewer

**Commit message:** `feat: evaluate page`

---

## Step 8.10: Test Placeholder

**File:** `frontend/test.html`

**Content:**
- Simple "Coming Soon" message
- LiveKit Voice Integration teaser
- Link back to dashboard

**Commit message:** `feat: test page placeholder`

---

## Step 8.11: Update evaluate.py CLI

**File:** `evaluate.py`

**Changes:**
- Use `NLUDebtCollectionEnv` instead of `DebtCollectionEnv`
- Use `NLU_STATE_DIM` for agent creation
- Remove DQN vs DDQ comparison (just evaluate single checkpoint)
- Add `--algorithm` flag

**Commit message:** `fix: evaluate.py uses NLU environment`

---

## Step 8.12: Test Full Flow

**Tests:**
1. Start server: `python run_server.py`
2. Open `http://localhost:8000`
3. Navigate to Train page → Start 10-episode training
4. Navigate to Evaluate page → Run evaluation on checkpoint
5. Verify Test page shows placeholder

**Commit message:** `test: verify Phase 8 complete`

---

## Real-Time Updates: Resource Cost

**Q: Does real-time updating take a lot of resources?**

**A: No, minimal!**
- WebSocket connection = ~100 bytes overhead
- Each update message = ~200-500 bytes JSON
- Network: ~1KB per episode (negligible)
- The LLM API calls are 99% of compute cost
- Browser rendering is trivial for this data volume

**Throttling option:** If needed, can update every N episodes instead of every episode.

---

## Current Status

**Phase 8 Start Date:** 2025-12-03
**Current Step:** 8.0 (Cleanup)

Let's begin! 🚀
