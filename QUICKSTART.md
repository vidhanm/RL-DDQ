# Quick Start Guide

## ✅ You Have Built

**Complete DQN baseline system with:**
- 🤖 4 debtor personas (angry, cooperative, sad, avoidant)
- 🎯 6 agent strategies (empathetic, ask, firm, payment_plan, settlement, hard_close)
- 🧠 DQN reinforcement learning agent
- 💬 OpenAI LLM integration for natural conversations
- 📊 Training and evaluation pipeline

**Total: 20 Python files, ~2,500 lines of code**

---

## 🚀 Step 1: Install Dependencies

```bash
# Make sure you're in the project directory
cd "c:\Users\Vidhan\Desktop\RL DDQ"

# Install required packages
pip install -r requirements.txt
```

**Dependencies:**
- PyTorch (for neural networks)
- Gymnasium (RL environment interface)
- OpenAI (LLM API)
- NumPy, Pandas, Matplotlib (standard ML libs)

---

## 🧪 Step 2: Test the Environment

### Option A: Test Without LLM (Fast)

```bash
python test_env.py
```

When prompted, type `n` to skip LLM test.

**What this does:**
- Creates 3 random debtor personas
- Runs conversations with random actions
- Shows you state encoding, rewards, and termination
- Verifies all imports work

### Option B: Test With LLM (Requires API Key, ~$0.05)

```bash
# Set your OpenAI API key first
export OPENAI_API_KEY="your-key-here"  # Linux/Mac
set OPENAI_API_KEY=your-key-here       # Windows

# Run test
python test_env.py
```

When prompted, type `y` to test with LLM.

**What this does:**
- Generates realistic agent utterances
- Simulates debtor responses with personas
- Shows complete natural conversations
- Costs about $0.05

---

## 🏋️ Step 3: Train DQN Agent

### Quick Test Training (No LLM)

```bash
python train.py --episodes 50 --no-llm
```

**What happens:**
- Trains for 50 episodes (5-10 minutes)
- Uses placeholder text instead of LLM
- Agent learns from persona behavior patterns
- Saves checkpoints to `checkpoints/`
- **Cost: $0 (no LLM calls)**

### Full Training (With LLM)

```bash
# Make sure API key is set
export OPENAI_API_KEY="your-key-here"

# Train for 200 episodes
python train.py --episodes 200
```

**What happens:**
- Generates real conversations with LLM
- Agent learns from natural language interactions
- Takes ~2-4 hours (depends on LLM API speed)
- Saves checkpoints every 50 episodes
- **Cost: ~$12-15 (200 episodes × 16 LLM calls × $0.004)**

### Training Options

```bash
# Show conversations during training
python train.py --episodes 100 --render

# Save to custom directory
python train.py --episodes 100 --save-dir my_checkpoints

# Quick test (10 episodes with LLM)
python train.py --episodes 10
```

---

## 📊 Step 4: Monitor Training

### During Training

You'll see progress like this:

```
Training: 100%|████████| 200/200 [2:30:00<00:00, 45.00s/it]

Episode 50/200
  Avg Reward (last 10): 12.34
  Success Rate (last 10): 40.0%
  Avg Loss: 0.0234
  Epsilon: 0.6065
  Buffer: 400/10000
  → Checkpoint saved: checkpoints/dqn_episode_50.pt
  → Evaluation Success Rate: 45.0%
```

### Key Metrics

- **Success Rate**: % of conversations ending in payment commitment
- **Avg Reward**: Higher is better (target: positive rewards)
- **Epsilon**: Exploration rate (should decrease over time)
- **Loss**: Training loss (should stabilize, not necessarily decrease to 0)

---

## 🎯 Expected Results

### After 50 Episodes
- Success rate: ~20-30%
- Agent learning basic patterns
- Random exploration still dominant

### After 100 Episodes
- Success rate: ~40-50%
- Agent choosing better strategies
- Learns persona-specific tactics

### After 200 Episodes
- Success rate: ~50-70%
- Consistent good performance
- DQN baseline complete ✅

---

## 📁 Output Files

After training, you'll have:

```
checkpoints/
├── dqn_episode_50.pt      # Checkpoint at episode 50
├── dqn_episode_100.pt     # Checkpoint at episode 100
├── dqn_episode_150.pt     # Checkpoint at episode 150
├── dqn_episode_200.pt     # Checkpoint at episode 200
├── dqn_final.pt           # Final trained model
└── training_history.json  # Rewards, successes, losses
```

---

## 🐛 Troubleshooting

### "ModuleNotFoundError"

```bash
# Make sure all packages installed
pip install -r requirements.txt
```

### "OpenAI API key not found"

```bash
# Set environment variable
export OPENAI_API_KEY="sk-..."  # Your actual key

# Or create .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### Training is slow

- **Use GPT-3.5-turbo** (10x faster, 10x cheaper)
  - Edit `config.py`: Set `USE_DEV_MODEL = True`
- **Reduce episodes**: Start with 50 instead of 200
- **Train without LLM first**: Use `--no-llm` flag

### "RuntimeError: CUDA out of memory"

```python
# Edit config.py
class DeviceConfig:
    DEVICE = "cpu"  # Force CPU instead of GPU
```

---

## ⏭️ Next Steps

### Once DQN Training Works:

**1. Analyze Results**
- Check `checkpoints/training_history.json`
- Plot learning curves
- Review successful conversations

**2. Build World Model (DDQ)**
- Create `agent/world_model.py`
- Add imagination mechanism
- Compare DQN vs DDQ

**3. Optimize Hyperparameters**
- Tune learning rate, epsilon decay
- Adjust reward weights
- Try different network architectures

**4. Add Visualizations**
- Plot learning curves
- Heatmaps of Q-values
- Conversation examples

**5. Prepare Demo**
- Best performing model
- Example conversations
- Performance comparison graphs

---

## 💡 Tips

### Cost Optimization
- **Development**: Use `--no-llm` or GPT-3.5-turbo
- **Final Training**: Use GPT-4 for best quality
- **Budget**: $50-100 should be plenty for full project

### Time Management
- **Environment test**: 5 minutes
- **Quick training (50 episodes, no LLM)**: 10 minutes
- **Full training (200 episodes, with LLM)**: 2-4 hours
- **Build world model**: 2-3 hours
- **Full DDQ training**: 2-4 hours

### Best Practices
- Test without LLM first (fast iteration)
- Start with small episode counts
- Monitor training - stop if something looks wrong
- Save checkpoints frequently
- Review conversations to understand agent behavior

---

## 📚 Documentation

- **[README.md](README.md)** - Project overview
- **[WORKFLOW.md](WORKFLOW.md)** - Detailed system workflow
- **[CONTEXT.md](CONTEXT.md)** - Project planning and decisions
- **[config.py](config.py)** - All hyperparameters

---

## ✨ You're Ready!

Run this to start:

```bash
# 1. Test environment (5 min)
python test_env.py

# 2. Quick training test (10 min)
python train.py --episodes 50 --no-llm

# 3. Full DQN training (2-4 hours)
python train.py --episodes 200
```

**Good luck! 🚀**
