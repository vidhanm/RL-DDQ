# DDQ Debt Collection Agent - Complete Workflow Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Episode Execution Flow](#episode-execution-flow)
3. [Training Flow](#training-flow)
4. [LLM Usage](#llm-usage)
5. [World Model Mechanics](#world-model-mechanics)
6. [State Representation](#state-representation)
7. [Action Space](#action-space)
8. [Reward Function](#reward-function)

---

## System Overview

This system implements a **self-improving debt collection agent** using DDQ (Dyna-style Data-efficient Q-learning).

### Core Components:

```
┌─────────────────────────────────────────────┐
│  Debt Collector Agent (RL)                  │
│  - Chooses strategies using DQN             │
│  - Uses LLM to generate utterances          │
│  - Learns via DDQ (DQN + World Model)       │
└─────────────────────────────────────────────┘
                  ↕ (state, action, reward)
┌─────────────────────────────────────────────┐
│  Debtor Simulator (Environment)             │
│  - Simulates debtor with persona            │
│  - Uses LLM to generate responses           │
│  - Returns state and reward                 │
└─────────────────────────────────────────────┘
```

### Key Innovation: DDQ

- **DQN**: Learns which strategies work best
- **World Model**: Learns to predict debtor behavior
- **Imagination**: Generates synthetic training data from world model
- **Result**: 5-10x faster learning with same number of LLM calls

---

## Episode Execution Flow

### Phase 1: Conversation Setup

```python
# Step 1: Initialize new conversation
state, info = env.reset()

# What happens internally:
# - Random debtor persona selected (angry/sad/cooperative/avoidant)
# - Initial attributes set:
debtor = {
    "persona": "angry",           # Personality type
    "sentiment": -0.3,            # -1 (hostile) to +1 (friendly)
    "cooperation": 0.2,           # 0 (uncooperative) to 1 (very cooperative)
    "engagement": 0.5,            # How much they're participating
    "financial_stress": 0.8,      # Their actual ability to pay
    "has_committed": False,       # Have they agreed to pay?
    "conversation_history": []    # Dialogue history
}

# Initial state returned:
state = {
    "turn": 0,
    "debtor_sentiment": -0.3,
    "debtor_cooperation": 0.2,
    "debtor_engagement": 0.5,
    "conversation_summary": "",
    "agent_last_action": None
}
```

---

### Phase 2: Single Turn Execution

#### **Step 2.1: Agent Strategy Selection (RL)**

```python
# Agent observes current state
state_vector = encode_state(state)
# Converts state dict to numerical vector: [0.0, -0.3, 0.2, 0.5, ...]

# DQN network predicts Q-value for each action
q_values = dqn_network.forward(state_vector)
# Output: [2.3, 5.1, 1.8, 3.2, 4.0, 2.1]
#         [emp, ask, firm, plan, settle, close]

# ε-greedy action selection
if random.random() < epsilon:
    action = random.randint(0, 5)    # Explore
else:
    action = torch.argmax(q_values)  # Exploit best action

# Example: action = 0 (empathetic_listening)
```

**Available Actions:**
- 0: `empathetic_listening` - Show understanding and compassion
- 1: `ask_about_situation` - Inquire about their circumstances
- 2: `firm_reminder` - Professional but assertive
- 3: `offer_payment_plan` - Propose installment options
- 4: `propose_settlement` - Offer reduced amount
- 5: `hard_close` - Create urgency/consequences

#### **Step 2.2: Strategy → Natural Language (LLM Call #1)**

```python
# Convert high-level strategy to natural utterance
prompt = f"""You are a professional debt collection agent.

Strategy to execute: empathetic_listening
Conversation history:
{conversation_history}

Generate a natural, professional utterance that executes this strategy.
Be empathetic, clear, and goal-oriented.

Output ONLY the utterance, no explanation."""

# 🔴 LLM API CALL
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7
)

agent_utterance = response.choices[0].message.content
# Example: "Hi, this is Sarah from ABC Collections. I understand that
#           financial situations can be challenging. I'm here to work
#           with you to find a solution. Can we talk about this?"
```

#### **Step 2.3: Debtor Response Generation (LLM Call #2)**

```python
# Simulate debtor's response using LLM
debtor_prompt = f"""You are roleplaying a debtor with this profile:

PERSONA: {debtor.persona}
- Current sentiment: {debtor.sentiment} (-1=hostile, +1=friendly)
- Cooperation level: {debtor.cooperation} (0=uncooperative, 1=very cooperative)
- Financial stress: {debtor.financial_stress}
- Personality traits: {get_persona_traits(debtor.persona)}

CONVERSATION SO FAR:
{conversation_history}

THE AGENT JUST SAID:
"{agent_utterance}"

Respond naturally as this debtor would. Be realistic and consistent with your persona.

Output JSON:
{{
    "response": "your response as the debtor",
    "new_sentiment": -0.4,          // How you feel after agent's message
    "new_cooperation": 0.3,         // Updated willingness to cooperate
    "new_engagement": 0.7,          // How engaged you are
    "will_commit_to_pay": false,    // Ready to commit to payment?
    "reasoning": "brief explanation" // Why you responded this way
}}"""

# 🔴 LLM API CALL
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": debtor_prompt}],
    temperature=0.8
)

debtor_output = json.loads(response.choices[0].message.content)
# Example:
# {
#     "response": "Look, I know I owe money, but I'm barely getting by.
#                  I don't need another call reminding me of this.",
#     "new_sentiment": -0.4,
#     "new_cooperation": 0.3,
#     "new_engagement": 0.7,
#     "will_commit_to_pay": false,
#     "reasoning": "Angry persona feels defensive when contacted about debt"
# }
```

#### **Step 2.4: State Update**

```python
# Update debtor's internal state
debtor.sentiment = debtor_output["new_sentiment"]
debtor.cooperation = debtor_output["new_cooperation"]
debtor.engagement = debtor_output["new_engagement"]
debtor.has_committed = debtor_output["will_commit_to_pay"]

# Add to conversation history
conversation_history.append({
    "turn": current_turn,
    "agent_action": action_name,
    "agent_utterance": agent_utterance,
    "debtor_utterance": debtor_output["response"]
})

# Create next state
next_state = {
    "turn": current_turn + 1,
    "debtor_sentiment": debtor.sentiment,
    "debtor_cooperation": debtor.cooperation,
    "debtor_engagement": debtor.engagement,
    "conversation_summary": get_recent_summary(conversation_history),
    "agent_last_action": action
}
```

#### **Step 2.5: Reward Calculation**

```python
def calculate_reward(old_state, action, new_state, debtor):
    reward = 0.0

    # PRIMARY GOAL: Payment commitment (+10)
    if debtor.has_committed:
        reward += 10.0

    # Sentiment improvement/degradation
    sentiment_change = new_state["debtor_sentiment"] - old_state["debtor_sentiment"]
    reward += sentiment_change * 3.0  # Weight: 3x

    # Cooperation improvement
    coop_change = new_state["debtor_cooperation"] - old_state["debtor_cooperation"]
    reward += coop_change * 2.0  # Weight: 2x

    # Engagement (good to keep them talking)
    if new_state["debtor_engagement"] > 0.6:
        reward += 0.5

    # Penalty for making debtor very hostile (avoid burnout)
    if new_state["debtor_sentiment"] < -0.8:
        reward -= 3.0

    # Small penalty per turn (encourage efficiency)
    reward -= 0.1

    return reward

# Example calculation:
# sentiment: -0.3 → -0.4 (worse by 0.1) → -0.3 reward
# cooperation: 0.2 → 0.3 (better by 0.1) → +0.2 reward
# engagement: 0.7 (high) → +0.5 reward
# turn penalty: -0.1
# Total: -0.3 + 0.2 + 0.5 - 0.1 = +0.3 reward
```

#### **Step 2.6: Store Experience**

```python
# Convert states to numerical vectors
state_vector = encode_state(state)
next_state_vector = encode_state(next_state)

# Create experience tuple
experience = (state_vector, action, reward, next_state_vector, done)

# Add to replay buffer
replay_buffer.add(experience)

# Conversation continues until:
# - Debtor commits to payment (success)
# - Debtor hangs up / refuses to engage (failure)
# - Max turns reached (10-15 turns)
done = check_termination_condition(next_state, debtor)
```

---

### Phase 3: Episode Completion

```python
# After conversation ends
episode_summary = {
    "success": debtor.has_committed,
    "total_reward": sum(all_rewards),
    "num_turns": len(conversation_history),
    "final_sentiment": debtor.sentiment,
    "persona": debtor.persona
}

# Store for analysis
episode_history.append(episode_summary)
```

---

## Training Flow

Training happens periodically (e.g., every 10 episodes) or after buffer reaches threshold.

### Step 1: Sample Real Experiences

```python
# Sample batch from replay buffer
batch_size = 32
real_batch = replay_buffer.sample(batch_size)

# real_batch contains 32 experiences:
# [(s1, a1, r1, s'1, done1), (s2, a2, r2, s'2, done2), ...]
```

### Step 2: Train World Model

```python
# World model learns to predict: (state, action) → (next_state, reward)

world_model.train()
optimizer = torch.optim.Adam(world_model.parameters(), lr=0.001)

for epoch in range(world_model_epochs):
    for (s, a, r, s_next, done) in real_batch:
        # Forward pass
        predicted_next_state, predicted_reward = world_model(s, a)

        # Compute loss
        state_loss = F.mse_loss(predicted_next_state, s_next)
        reward_loss = F.mse_loss(predicted_reward, r)
        total_loss = state_loss + reward_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

# After training, world model has learned:
# "angry debtor + empathy → sentiment improves, reward ≈ +1.0"
# "angry debtor + hard_close → sentiment worsens, reward ≈ -2.0"
```

### Step 3: Generate Imagined Experiences (DDQ Magic)

```python
imagined_batch = []

# For each real experience, generate K imagined variations
K = 5  # Imagination factor

for (s, a, r, s_next, done) in real_batch:
    # Start from real state
    sim_state = s_next  # Or random state from buffer

    # Imagine K steps forward
    for k in range(K):
        # Choose random action to explore
        sim_action = random.randint(0, 5)

        # ❌ NO LLM CALL - World model predicts instantly
        sim_next_state, sim_reward = world_model.predict(sim_state, sim_action)

        # Store imagined experience
        imagined_experience = (sim_state, sim_action, sim_reward, sim_next_state, False)
        imagined_batch.append(imagined_experience)

        # Continue chain: next imagination starts from predicted state
        sim_state = sim_next_state

# Result:
# - 32 real experiences
# - 32 * 5 = 160 imagined experiences
# - Total: 192 training samples
```

**Key Insight:** World model runs instantly (neural network forward pass), generating 160 extra experiences with ZERO LLM API calls.

### Step 4: Train DQN on Real + Imagined

```python
# Combine real and imagined experiences
combined_batch = real_batch + imagined_batch  # 192 total

# Standard DQN training
dqn.train()
optimizer = torch.optim.Adam(dqn.parameters(), lr=0.0001)

for (s, a, r, s_next, done) in combined_batch:
    # Current Q-value
    q_values = dqn(s)
    current_q = q_values[a]

    # Target Q-value (Bellman equation)
    with torch.no_grad():
        if done:
            target_q = r
        else:
            next_q_values = target_dqn(s_next)  # Use target network
            target_q = r + gamma * torch.max(next_q_values)

    # Compute loss
    loss = F.mse_loss(current_q, target_q)

    # Update DQN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Update target network periodically
if training_step % target_update_freq == 0:
    target_dqn.load_state_dict(dqn.state_dict())
```

---

## LLM Usage Summary

### When LLM is Called:

| Phase | Call Type | Purpose | Frequency |
|-------|-----------|---------|-----------|
| **Episode Execution** | Agent utterance generation | Convert strategy → natural language | Every turn |
| **Episode Execution** | Debtor response generation | Simulate realistic debtor | Every turn |
| **Evaluation/Demo** | Display conversations | Show results to user | On demand |

### When LLM is NOT Called:

| Phase | Alternative | Why No LLM |
|-------|-------------|------------|
| **Training - World Model** | Neural network training | Learning state patterns, not text |
| **Training - Imagination** | World model prediction | Generating numerical state transitions |
| **Training - DQN** | Q-learning updates | Pure RL optimization |

### Cost Analysis:

```
1 Episode = 8 turns average
1 Turn = 2 LLM calls (agent + debtor)
1 Episode = 16 LLM calls

100 Episodes = 1,600 LLM calls

Training on 100 episodes:
- Without DDQ: 1,600 experiences, 1,600 LLM calls
- With DDQ (K=5): 1,600 + 8,000 imagined = 9,600 experiences, 1,600 LLM calls

6x more training data, same LLM cost!
```

---

## World Model Mechanics

### Architecture:

```python
class WorldModel(nn.Module):
    def __init__(self, state_dim=20, action_dim=6, hidden_dim=128):
        super().__init__()

        # Input: state vector + action one-hot
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Predict next state
        self.state_predictor = nn.Linear(hidden_dim, state_dim)

        # Predict reward
        self.reward_predictor = nn.Linear(hidden_dim, 1)

    def forward(self, state_vector, action_one_hot):
        # state_vector: [batch, 20]
        # action_one_hot: [batch, 6]

        x = torch.cat([state_vector, action_one_hot], dim=-1)
        hidden = self.encoder(x)

        next_state = self.state_predictor(hidden)
        reward = self.reward_predictor(hidden)

        return next_state, reward
```

### What World Model Learns:

After training on real experiences, world model captures patterns like:

```
Pattern 1: Empathy on Angry Debtor
Input:  state=[turn=2, sentiment=-0.6, coop=0.2, ...], action=empathetic
Output: next_state=[turn=3, sentiment=-0.3, coop=0.4, ...], reward=+1.2

Pattern 2: Hard Close on Angry Debtor
Input:  state=[turn=2, sentiment=-0.6, coop=0.2, ...], action=hard_close
Output: next_state=[turn=3, sentiment=-0.9, coop=0.1, ...], reward=-2.5

Pattern 3: Payment Plan on Cooperative Debtor
Input:  state=[turn=4, sentiment=0.3, coop=0.7, ...], action=payment_plan
Output: next_state=[turn=5, sentiment=0.5, coop=0.9, ...], reward=+3.0
```

---

## State Representation

State is a dictionary converted to numerical vector for neural networks.

### State Dictionary:

```python
state = {
    # Conversation metadata
    "turn": 3,                          # Current turn number

    # Debtor attributes
    "debtor_sentiment": -0.2,           # -1 (hostile) to +1 (friendly)
    "debtor_cooperation": 0.5,          # 0 to 1
    "debtor_engagement": 0.8,           # 0 to 1

    # Conversation flags
    "mentioned_payment_plan": True,     # Has agent mentioned payment plan?
    "mentioned_consequences": False,    # Has agent mentioned consequences?
    "debtor_shared_situation": True,    # Did debtor open up?

    # Trends
    "sentiment_trend": 0.15,            # Change in last 2 turns
    "cooperation_trend": 0.1,           # Change in last 2 turns

    # Last action
    "agent_last_action": 1,             # Which strategy was used

    # Conversation summary (for display, not encoding)
    "conversation_summary": "Agent showed empathy, debtor opened up about job loss..."
}
```

### Encoding to Vector:

```python
def encode_state(state_dict):
    vector = [
        state_dict["turn"] / 15.0,                      # Normalize
        state_dict["debtor_sentiment"],                 # Already -1 to 1
        state_dict["debtor_cooperation"],               # Already 0 to 1
        state_dict["debtor_engagement"],                # Already 0 to 1
        1.0 if state_dict["mentioned_payment_plan"] else 0.0,
        1.0 if state_dict["mentioned_consequences"] else 0.0,
        1.0 if state_dict["debtor_shared_situation"] else 0.0,
        state_dict["sentiment_trend"],
        state_dict["cooperation_trend"],
        # One-hot encode last action (6 dimensions)
        *one_hot(state_dict["agent_last_action"], num_actions=6)
    ]
    return torch.tensor(vector, dtype=torch.float32)
    # Shape: [20] - a 20-dimensional vector
```

---

## Action Space

```python
ACTION_SPACE = {
    0: {
        "name": "empathetic_listening",
        "description": "Show understanding and compassion",
        "example": "I understand this is a difficult situation..."
    },
    1: {
        "name": "ask_about_situation",
        "description": "Inquire about debtor's circumstances",
        "example": "Can you tell me what's been happening financially?"
    },
    2: {
        "name": "firm_reminder",
        "description": "Professional but assertive about debt",
        "example": "This account is now 90 days overdue and requires immediate attention."
    },
    3: {
        "name": "offer_payment_plan",
        "description": "Propose installment payment options",
        "example": "We can set up a monthly plan that fits your budget..."
    },
    4: {
        "name": "propose_settlement",
        "description": "Offer reduced total amount",
        "example": "If you can pay 70% today, we can settle this account."
    },
    5: {
        "name": "hard_close",
        "description": "Create urgency with consequences",
        "example": "Without payment by Friday, this will escalate to legal action."
    }
}
```

---

## Reward Function

```python
def calculate_reward(old_state, action, new_state, debtor, done):
    """
    Reward function balances multiple objectives:
    - Primary: Get payment commitment
    - Secondary: Improve sentiment and cooperation
    - Tertiary: Maintain engagement, avoid hostility
    """

    reward = 0.0

    # ========================================
    # PRIMARY OBJECTIVE: Payment Commitment
    # ========================================
    if debtor.has_committed and not debtor.was_committed_before:
        reward += 10.0  # Major success!

    # ========================================
    # SENTIMENT CHANGE
    # ========================================
    sentiment_delta = new_state["debtor_sentiment"] - old_state["debtor_sentiment"]
    reward += sentiment_delta * 3.0  # Weight: 3x

    # ========================================
    # COOPERATION CHANGE
    # ========================================
    coop_delta = new_state["debtor_cooperation"] - old_state["debtor_cooperation"]
    reward += coop_delta * 2.0  # Weight: 2x

    # ========================================
    # ENGAGEMENT
    # ========================================
    if new_state["debtor_engagement"] > 0.6:
        reward += 0.5  # Good - debtor is talking
    elif new_state["debtor_engagement"] < 0.3:
        reward -= 1.0  # Bad - losing them

    # ========================================
    # PENALTIES
    # ========================================

    # Avoid making debtor extremely hostile
    if new_state["debtor_sentiment"] < -0.8:
        reward -= 3.0  # Relationship damage

    # Encourage efficiency (don't drag conversation)
    reward -= 0.1  # Small penalty per turn

    # Penalty for ending without commitment
    if done and not debtor.has_committed:
        reward -= 5.0  # Failed conversation

    return reward
```

---

## Comparison: DQN vs DDQ

### DQN Only:

```
Episode 1-100: Collect 800 experiences (100 episodes * 8 turns avg)
Training: Learn from 800 real experiences
Performance after 100 episodes: Moderate
LLM Calls: 1,600
```

### DDQ (DQN + World Model):

```
Episode 1-100: Collect 800 experiences (100 episodes * 8 turns avg)
Training:
  - Learn world model from 800 real experiences
  - Generate 4,000 imagined experiences (K=5)
  - Train DQN on 800 + 4,000 = 4,800 total experiences
Performance after 100 episodes: Significantly better
LLM Calls: 1,600 (same as DQN!)
```

**Key Advantage:** 6x more training data with zero extra cost.

---

## Version History

- **v1.0** (2025-11-29): Initial workflow documentation
- Updates will be tracked here as code evolves

---

## References

- Sutton, R. S. (1990). Integrated architectures for learning, planning, and reacting based on approximating dynamic programming. *Dyna-Q*
- Schrittwieser et al. (2019). Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model. *MuZero*
- Hafner et al. (2020). Dream to Control: Learning Behaviors by Latent Imagination. *Dreamer*
