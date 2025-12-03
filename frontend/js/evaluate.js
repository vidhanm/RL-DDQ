/**
 * Evaluation Page JavaScript
 * API calls for model evaluation
 */

// DOM Elements
const checkpointSelect = document.getElementById('checkpoint-select');
const evalEpisodes = document.getElementById('eval-episodes');
const evalUseLlm = document.getElementById('eval-use-llm');
const runEvalBtn = document.getElementById('run-eval-btn');
const statusBar = document.getElementById('status-bar');
const loadingOverlay = document.getElementById('loading-overlay');

// Results elements
const evalSuccessRate = document.getElementById('eval-success-rate');
const evalAvgReward = document.getElementById('eval-avg-reward');
const evalAvgLength = document.getElementById('eval-avg-length');
const evalNumEpisodes = document.getElementById('eval-num-episodes');
const evaluationSummary = document.getElementById('evaluation-summary');
const summaryText = document.getElementById('summary-text');
const sampleConversations = document.getElementById('sample-conversations');
const checkpointsTbody = document.getElementById('checkpoints-tbody');

// Conversation navigation
const convNav = document.querySelector('.conversation-nav');
const prevConvBtn = document.getElementById('prev-conv-btn');
const nextConvBtn = document.getElementById('next-conv-btn');
const convIndicator = document.getElementById('conv-indicator');

// State
let conversations = [];
let currentConvIndex = 0;

// Load checkpoints on page load
async function loadCheckpoints() {
    try {
        const response = await fetch('/api/evaluate/checkpoints');
        const data = await response.json();
        
        checkpointSelect.innerHTML = '';
        checkpointsTbody.innerHTML = '';
        
        if (data.checkpoints.length === 0) {
            checkpointSelect.innerHTML = '<option value="">No checkpoints found</option>';
            checkpointsTbody.innerHTML = '<tr><td colspan="3">No checkpoints found</td></tr>';
            return;
        }
        
        data.checkpoints.forEach(cp => {
            // Add to select
            const option = document.createElement('option');
            option.value = cp.name;
            option.textContent = `${cp.name} (${cp.algorithm.toUpperCase()})`;
            checkpointSelect.appendChild(option);
            
            // Add to table
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${cp.name}</td>
                <td><span class="badge ${cp.algorithm}">${cp.algorithm.toUpperCase()}</span></td>
                <td>
                    <button class="small" onclick="selectCheckpoint('${cp.name}')">Select</button>
                </td>
            `;
            checkpointsTbody.appendChild(row);
        });
        
        statusBar.textContent = `Found ${data.checkpoints.length} checkpoint(s)`;
    } catch (error) {
        console.error('Failed to load checkpoints:', error);
        statusBar.textContent = 'Failed to load checkpoints';
        statusBar.className = 'status-bar error';
    }
}

// Select checkpoint from table
function selectCheckpoint(name) {
    checkpointSelect.value = name;
    statusBar.textContent = `Selected: ${name}`;
}

// Run evaluation
async function runEvaluation() {
    const checkpoint = checkpointSelect.value;
    if (!checkpoint) {
        statusBar.textContent = 'Please select a checkpoint';
        statusBar.className = 'status-bar error';
        return;
    }
    
    // Determine algorithm from checkpoint name
    const algorithm = checkpoint.toLowerCase().includes('ddq') ? 'ddq' : 'dqn';
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    document.getElementById('loading-text').textContent = 'Running evaluation...';
    statusBar.textContent = 'Evaluating...';
    statusBar.className = 'status-bar';
    runEvalBtn.disabled = true;
    
    try {
        const response = await fetch('/api/evaluate/run', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                checkpoint: checkpoint,
                algorithm: algorithm,
                num_episodes: parseInt(evalEpisodes.value),
                use_llm: evalUseLlm.checked
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Evaluation failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
        statusBar.textContent = 'Evaluation complete!';
        statusBar.className = 'status-bar success';
    } catch (error) {
        console.error('Evaluation error:', error);
        statusBar.textContent = `Error: ${error.message}`;
        statusBar.className = 'status-bar error';
    } finally {
        loadingOverlay.style.display = 'none';
        runEvalBtn.disabled = false;
    }
}

// Display evaluation results
function displayResults(result) {
    // Update stats
    evalSuccessRate.textContent = `${(result.success_rate * 100).toFixed(1)}%`;
    evalSuccessRate.style.color = result.success_rate >= 0.5 ? '#22c55e' : '#ef4444';
    
    evalAvgReward.textContent = result.avg_reward.toFixed(2);
    evalAvgReward.style.color = result.avg_reward >= 0 ? '#22c55e' : '#ef4444';
    
    evalAvgLength.textContent = `${result.avg_length.toFixed(1)} turns`;
    evalNumEpisodes.textContent = result.num_episodes;
    
    // Summary
    evaluationSummary.style.display = 'block';
    const rating = result.success_rate >= 0.6 ? 'Good' : result.success_rate >= 0.4 ? 'Moderate' : 'Needs improvement';
    summaryText.textContent = `The model achieved a ${(result.success_rate * 100).toFixed(1)}% success rate over ${result.num_episodes} episodes. Performance rating: ${rating}.`;
    
    // Sample conversations
    conversations = result.sample_conversations || [];
    currentConvIndex = 0;
    
    if (conversations.length > 0) {
        convNav.style.display = 'flex';
        displayConversation(0);
        updateConvNav();
    } else {
        convNav.style.display = 'none';
        sampleConversations.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding-top: 50px;">No conversations recorded</p>';
    }
}

// Display a single conversation
function displayConversation(index) {
    const conv = conversations[index];
    sampleConversations.innerHTML = '';
    
    conv.forEach((turn, i) => {
        const turnDiv = document.createElement('div');
        turnDiv.className = 'dialogue-turn';
        turnDiv.innerHTML = `
            <div class="turn-number">Turn ${i + 1}</div>
            <div class="message agent">
                <strong>🤖 Agent:</strong> ${turn.agent_utterance || '[Action taken]'}
            </div>
            <div class="message debtor">
                <strong>👤 Debtor:</strong> ${turn.debtor_response || '[Response]'}
            </div>
        `;
        sampleConversations.appendChild(turnDiv);
    });
}

// Update conversation navigation
function updateConvNav() {
    convIndicator.textContent = `${currentConvIndex + 1} / ${conversations.length}`;
    prevConvBtn.disabled = currentConvIndex === 0;
    nextConvBtn.disabled = currentConvIndex === conversations.length - 1;
}

// Navigation handlers
prevConvBtn?.addEventListener('click', () => {
    if (currentConvIndex > 0) {
        currentConvIndex--;
        displayConversation(currentConvIndex);
        updateConvNav();
    }
});

nextConvBtn?.addEventListener('click', () => {
    if (currentConvIndex < conversations.length - 1) {
        currentConvIndex++;
        displayConversation(currentConvIndex);
        updateConvNav();
    }
});

// Event listeners
runEvalBtn.addEventListener('click', runEvaluation);

// Initialize
document.addEventListener('DOMContentLoaded', loadCheckpoints);

// Make selectCheckpoint available globally
window.selectCheckpoint = selectCheckpoint;

// Add some styles
const style = document.createElement('style');
style.textContent = `
    .dialogue-turn {
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    .turn-number {
        font-size: 0.75rem;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .message {
        padding: 8px 12px;
        border-radius: 8px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
    .message.agent {
        background: rgba(59, 130, 246, 0.1);
        border-left: 3px solid #3b82f6;
    }
    .message.debtor {
        background: rgba(139, 92, 246, 0.1);
        border-left: 3px solid #8b5cf6;
    }
    .status-bar.success {
        background: rgba(34, 197, 94, 0.2);
        border-color: #22c55e;
    }
    .status-bar.error {
        background: rgba(239, 68, 68, 0.2);
        border-color: #ef4444;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge.ddq {
        background: rgba(139, 92, 246, 0.2);
        color: #a78bfa;
    }
    .badge.dqn {
        background: rgba(59, 130, 246, 0.2);
        color: #60a5fa;
    }
    button.small {
        padding: 6px 12px;
        font-size: 0.8rem;
    }
`;
document.head.appendChild(style);
