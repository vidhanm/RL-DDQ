# Project Context & Planning

## Project Overview

**Goal**: Build a self-improving debt collection AI agent using reinforcement learning (DDQ algorithm) for a job assignment.

**Timeline**: Multiple weeks available

**Target Company**: Debt collection AI specialization

**Deliverable**: Working demo showing agent learning optimal conversation strategies

---

## Key Decisions Made

### ✅ Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **RL Algorithm** | DDQ (DQN + World Model) | Sample efficiency, demonstrates advanced RL understanding |
| **Environment** | Custom Gymnasium interface | Full control, no framework overhead |
| **LLM Provider** | OpenAI GPT | User has API key, reliable responses |
| **Debtor Simulation** | LLM-based (not rule-based) | More realistic, flexible personas |
| **Voice Integration** | Text-first, voice later | Faster development, can add TTS/STT later |
| **Framework Choice** | No ConvLab | Too complex for timeline, wrong domain fit |
| **Baseline Comparison** | DQN vs DDQ | Shows DDQ's value scientifically |

### ✅ Design Decisions

| Aspect | Design | Reason |
|--------|--------|--------|
| **Action Space** | 6 high-level strategies | Manageable for RL, clear for evaluation |
| **State Representation** | 20-dim numerical vector | Captures conversation state compactly |
| **Debtor Personas** | 4 types (angry/cooperative/sad/avoidant) | Covers main negotiation scenarios |
| **Episode Length** | 8-15 turns max | Realistic conversation, manageable horizon |
| **Imagination Factor** | K=5 | Balance between data augmentation and error accumulation |
| **LLM Usage** | Only during execution, not training | Cost efficiency |

---

## Research & Understanding

### Questions Explored

1. **What is DDQ and why use it?**
   - ✅ Understood: Sample-efficient RL using world model imagination
   - ✅ Benefit: 5-10x more training data without extra LLM costs

2. **How does conversation flow work?**
   - ✅ Understood: Agent strategy → LLM utterance → Debtor LLM response → State update → Reward
   - ✅ Key: RL operates on numerical states, not text

3. **When is LLM called vs not called?**
   - ✅ LLM used: During conversations (agent + debtor utterances)
   - ✅ LLM NOT used: During training (world model prediction is neural net)

4. **What does world model predict?**
   - ✅ Predicts: Next state vector + reward (numbers)
   - ✅ Does NOT predict: Actual utterance text
   - ✅ Why: RL only needs state transitions, not words

5. **How does imagination generate experiences?**
   - ✅ NOT copying: Takes real state, tries different actions
   - ✅ IS exploring: "What if I had chosen action B instead of A?"
   - ✅ Chains: Can imagine multi-step scenarios

6. **Are world model predictions reliable?**
   - ✅ Honest answer: Often wrong, but better than nothing
   - ✅ Mitigations: Mix real+imagined, ensemble, limit horizon
   - ✅ Research: Proven to work despite imperfections (MuZero, Dreamer)

### ConvLab Analysis

**Why NOT using ConvLab:**
- Wrong domain (task-oriented dialogue vs negotiation)
- Massive complexity overhead (NLU, DST, NLG components)
- 1-2 week learning curve
- Hides RL implementation details
- Rule-based user simulators (we want LLM-based)

**When ConvLab WOULD be good:**
- Production task-oriented chatbot (hotel, restaurant booking)
- Multi-domain dialogue management
- Have months to learn framework

---

## Development Plan

### Phase 1: Foundation

**Status**: ✅ **COMPLETE**

**Completed**:
- [x] Architecture planning
- [x] Workflow documentation (WORKFLOW.md)
- [x] Project README (README.md)
- [x] Context tracking (CONTEXT.md)
- [x] Project structure (folders created)
- [x] requirements.txt
- [x] config.py
- [x] LLM integration (OpenAI client)
- [x] Debtor environment (Gymnasium interface)

---

### Phase 2: Environment & LLM

**Status**: ✅ **COMPLETE**

**Completed**:
- [x] Implement `DebtorEnv` class (Gymnasium interface)
- [x] Implement persona definitions (4 personas: angry, cooperative, sad, avoidant)
- [x] OpenAI API integration + NVIDIA NIM client
- [x] Prompt templates for agent + debtor
- [x] State encoding/decoding utilities
- [x] Reward function implementation
- [x] Test: Run random agent, verify conversations work

**Success Criteria**: ✅ **MET**
- ✅ Can run complete conversations with random actions
- ✅ Debtor responses are realistic and persona-consistent
- ✅ State and reward calculations are correct

---

### Phase 3: Baseline DQN

**Status**: ✅ **COMPLETE**

**Completed**:
- [x] DQN network architecture (with Dueling DQN variant)
- [x] Replay buffer implementation (with prioritized replay variant)
- [x] Training loop (vanilla DQN, no world model)
- [x] Epsilon-greedy exploration
- [x] Target network updates
- [x] Logging and checkpointing (Neptune.ai integration)
- [x] Evaluation script with plotting

**Success Criteria**: ✅ **READY FOR VALIDATION**
- ⏳ DQN agent learns to improve over episodes (needs full training run)
- ⏳ Success rate increases from ~10% to ~50%+ (needs full training run)
- ✅ Can visualize learning curve (plotting implemented)

---

### Phase 4: World Model & DDQ

**Status**: ✅ **COMPLETE**

**Completed**:
- [x] World model network architecture
- [x] World model training on real experiences
- [x] Imagination mechanism (generate synthetic data, K=5)
- [x] DDQ training loop (real + imagined)
- [x] Mix ratio mitigation (75% real, 25% imagined)
- [x] Ensemble and uncertainty estimation (implemented, optional)
- [x] Performance comparison tools (DQN vs DDQ in evaluate.py)

**Success Criteria**: ✅ **READY FOR VALIDATION**
- ⏳ World model predictions are reasonably accurate (needs full training run)
- ⏳ DDQ learns faster than DQN (needs comparison run)
- ✅ Can demonstrate 5-10x data augmentation (imagination mechanism working)

---

### Phase 5: Enhancement & Evaluation

**Status**: ⏳ **IN PROGRESS**

**Completed**:
- [x] Add all 4 debtor personas (angry, cooperative, sad, avoidant)
- [x] Persona-conditioned world model (state includes persona info)
- [x] Basic plotting capabilities (learning curves in evaluate.py)

**In Progress**:
- [ ] Hyperparameter tuning (testing with 3 episodes, need full run)
- [ ] Generate comprehensive visualizations (learning curves, heatmaps, Q-value analysis)
- [ ] Record example conversations
- [ ] Ablation studies (K=2 vs K=5 vs K=10)
- [ ] Final documentation and demo preparation

**Success Criteria**: ⏳ **PENDING VALIDATION**
- ⏳ Agent handles all persona types effectively (need full training)
- ⏳ Clear evidence of DDQ's advantages (need DQN vs DDQ comparison)
- ⏳ Professional demo ready for interview (need visualizations & documentation)

---

### Phase 6: Optional Enhancements

**If time allows**:
- [ ] Voice integration (TTS for agent, STT for debtor)
- [ ] Web interface for live demos
- [ ] Advanced world model architectures
- [ ] Multi-step planning (beyond 1-step)
- [ ] Curriculum learning (easy → hard personas)

---

## Technical Specifications

### Action Space (6 actions)

```python
0: empathetic_listening    # "I understand this is difficult..."
1: ask_about_situation     # "Can you tell me what happened?"
2: firm_reminder           # "This account is 90 days overdue..."
3: offer_payment_plan      # "We can set up monthly payments..."
4: propose_settlement      # "We can settle for 70% today..."
5: hard_close              # "Without payment, this escalates to legal..."
```

### State Representation (20 dimensions)

```python
[
    turn_normalized,           # 0-1
    debtor_sentiment,          # -1 to 1
    debtor_cooperation,        # 0 to 1
    debtor_engagement,         # 0 to 1
    mentioned_payment_plan,    # 0 or 1
    mentioned_consequences,    # 0 or 1
    debtor_shared_situation,   # 0 or 1
    sentiment_trend,           # -1 to 1
    cooperation_trend,         # -1 to 1
    last_action_0,            # One-hot encoded
    last_action_1,            # (6 dimensions total)
    last_action_2,
    last_action_3,
    last_action_4,
    last_action_5,
    # Additional features as needed
]
```

### Debtor Personas (4 types)

1. **Angry Persona**
   - Initial sentiment: -0.4 to -0.6
   - Cooperative: 0.1 to 0.3
   - Triggers: Pressure, consequences
   - Responds to: Empathy, understanding

2. **Cooperative Persona**
   - Initial sentiment: 0.2 to 0.4
   - Cooperative: 0.6 to 0.8
   - Triggers: Complications, unrealistic demands
   - Responds to: Payment plans, clear paths

3. **Sad/Overwhelmed Persona**
   - Initial sentiment: -0.2 to 0.0
   - Cooperative: 0.4 to 0.6
   - Triggers: Pressure, judgment
   - Responds to: Empathy, flexible options

4. **Avoidant Persona**
   - Initial sentiment: -0.1 to 0.1
   - Cooperative: 0.2 to 0.4
   - Triggers: Long conversations, pressure
   - Responds to: Quick solutions, urgency

### Hyperparameters (Initial)

```python
# RL Parameters
learning_rate = 0.0001
gamma = 0.95              # Discount factor
epsilon_start = 1.0       # Exploration
epsilon_end = 0.05
epsilon_decay = 0.995
batch_size = 32
replay_buffer_size = 10000
target_update_freq = 10   # Episodes

# DDQ Parameters
K = 5                     # Imagination factor
world_model_lr = 0.001
world_model_epochs = 5
min_buffer_size = 500     # Before using world model
real_ratio = 0.75         # 75% real, 25% imagined

# Environment
max_turns = 15
num_personas = 4

# Training
num_episodes = 500
eval_freq = 10           # Evaluate every N episodes
save_freq = 50           # Save checkpoint every N episodes
```

---

## LLM Cost Estimation

### Per Episode

```
1 episode = ~8 turns average
1 turn = 2 LLM calls (agent + debtor)
1 episode = 16 LLM calls

Cost per call (GPT-4):
- Input: ~200 tokens = $0.001
- Output: ~100 tokens = $0.003
- Total per call: ~$0.004

1 episode = 16 * $0.004 = $0.064
```

### Training Budget

```
DQN baseline: 200 episodes
200 * $0.064 = $12.80

DDQ: 200 episodes
200 * $0.064 = $12.80 (same cost!)

Both algorithms: $25.60 total

With testing/development: ~$50-100 budget
```

### Cost Optimization

- Use GPT-3.5-turbo for development ($0.0005 per call) = 10x cheaper
- Switch to GPT-4 for final training and demo
- Cache similar prompts where possible

---

## Risk Assessment & Mitigations

### Risk 1: World Model Learns Poorly
**Impact**: DDQ performs worse than DQN
**Mitigation**:
- Pre-train on rule-based simulator
- Start with simple personas
- Monitor world model accuracy
- Fallback: Use DQN baseline

### Risk 2: LLM Responses Inconsistent
**Impact**: Training signal is noisy
**Mitigation**:
- Lower temperature for more consistent responses
- Detailed persona prompts
- Test multiple prompts, pick best
- Accept some noise (RL is robust to it)

### Risk 3: Agent Learns Unethical Strategies
**Impact**: Demo shows manipulative behavior
**Mitigation**:
- Reward function includes sentiment/cooperation
- Penalize overly aggressive actions
- Human review of learned strategies
- Add ethical constraints if needed

### Risk 4: Timeline Slippage
**Impact**: Not ready for interview
**Mitigation**:
- Build DQN baseline first (functional demo)
- DDQ is enhancement, not requirement
- Document what works + plan for rest
- Focus on understanding over perfection

---

## Questions for Future

- [ ] Should we add emotion detection from debtor responses?
- [ ] Multi-objective reward (payment AND satisfaction)?
- [ ] Transfer learning from one persona to another?
- [ ] Real-world data integration (if company provides)?
- [ ] Explainability: Why did agent choose this action?

---

## Meeting Notes & Decisions

### Session 1 (2025-11-29)

**Discussed**:
- Project scope and goals
- DDQ vs DQN comparison
- LLM integration approach
- ConvLab evaluation (decided against)
- Complete workflow explanation

**Decided**:
- Use DDQ for sample efficiency demo
- LLM-based debtor (not rule-based)
- Text-first (voice optional later)
- Build both DQN and DDQ for comparison
- Custom environment (no ConvLab)

**User Understanding Checkpoints**:
✅ Understands DDQ workflow (episode execution)
✅ Understands training flow (real + imagined)
✅ Understands when LLM is/isn't called
✅ Understands world model predictions (states, not text)
✅ Understands imagination = exploring alternatives (not copying)
✅ Understands world model can be wrong but still useful

**Next Steps**:
- Complete project setup (requirements, config)
- Begin environment implementation
- Test LLM integration with simple conversation

---

## Resources & References

### Papers
- Sutton (1990): "Integrated Architectures for Learning, Planning, and Reacting" (Dyna-Q)
- Schrittwieser et al. (2019): "MuZero: Mastering Atari, Go, Chess and Shogi"
- Hafner et al. (2020): "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer)

### Code References
- OpenAI Gymnasium: https://gymnasium.farama.org/
- PyTorch DQN Tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/ (reference only)

### Domain Knowledge
- Debt collection best practices
- Negotiation psychology
- Conversation design patterns

---

## Changelog

### 2025-11-29 (Session 1 - Planning & Setup)
- Project initialized
- Architecture designed
- WORKFLOW.md created (comprehensive flow documentation)
- README.md created (project overview)
- CONTEXT.md created (this file)
- Folders created: environment/, agent/, llm/, utils/
- requirements.txt created
- config.py created with all hyperparameters

### 2025-11-29 (Session 2 - Core Implementation)
**Environment & LLM Integration:**
- ✅ environment/debtor_persona.py - 4 debtor personas with behavior patterns
- ✅ environment/debtor_env.py - Full Gymnasium environment
- ✅ environment/__init__.py
- ✅ llm/openai_client.py - OpenAI API wrapper with retry logic
- ✅ llm/prompts.py - Prompt templates for agent + debtor
- ✅ llm/__init__.py
- ✅ utils/state_encoder.py - State encoding (dict → vector)
- ✅ utils/__init__.py
- ✅ test_env.py - Environment testing script

**RL Components:**
- ✅ utils/replay_buffer.py - Experience replay (with prioritized variant)
- ✅ agent/dqn.py - DQN network architecture (with Dueling variant)
- ✅ agent/dqn_agent.py - DQN agent with training logic
- ✅ agent/__init__.py
- ✅ train.py - Main training script

**Status**: 🎯 **DDQ Implementation Complete & Ready for Full Training!**

**Files Created**: 18 Python files total
**Lines of Code**: ~4,000 lines
**Next Steps**:
1. ✅ ~~Test environment (python test_env.py)~~ - DONE
2. ✅ ~~Train DQN baseline (python train.py)~~ - DONE (3 episodes tested)
3. ✅ ~~Build world model for DDQ~~ - DONE
4. ⏳ Run full training (75-200 episodes) to validate performance
5. ⏳ Compare DQN vs DDQ performance with visualizations
6. ⏳ Prepare demo and documentation

---

## Notes

- User has weeks available (not days) - adjust pace accordingly
- Focus on understanding AND implementation
- Build incrementally, test frequently
- Document decisions for interview discussion
- DDQ is ambitious but achievable with proper planning

**Current Status**: ✅ **Phase 4 Complete (DDQ Implementation)** → ⏳ **Phase 5 In Progress (Testing & Visualization)**

**Last Updated**: 2025-11-29 (Session 3)
