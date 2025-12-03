/**
 * DDQ Agent Dashboard Application
 * Main application logic
 */

// State
let currentModel = null;
let isEpisodeDone = false;
let qValueChart = null;

// DOM Elements
const elements = {
    // Controls
    modelSelect: document.getElementById('model-select'),
    personaSelect: document.getElementById('persona-select'),
    loadBtn: document.getElementById('load-btn'),
    startBtn: document.getElementById('start-btn'),
    actionSelect: document.getElementById('action-select'),
    actionBtn: document.getElementById('action-btn'),
    autoBtn: document.getElementById('auto-btn'),
    
    // Displays
    statusBar: document.getElementById('status-bar'),
    chatContainer: document.getElementById('chat-container'),
    statePanel: document.getElementById('state-panel'),
    qvalueChart: document.getElementById('qvalue-chart'),
    episodeStatus: document.getElementById('episode-status'),
    
    // Loading overlay
    loadingOverlay: document.getElementById('loading-overlay'),
    loadingText: document.getElementById('loading-text')
};

// ============== UI Helpers ==============

function showLoading(message = 'Loading...') {
    elements.loadingOverlay.classList.remove('hidden');
    elements.loadingText.textContent = message;
}

function hideLoading() {
    elements.loadingOverlay.classList.add('hidden');
}

function showStatus(message, type = 'info') {
    elements.statusBar.textContent = message;
    elements.statusBar.className = `status-bar status-${type}`;
}

function setButtonsEnabled(enabled) {
    elements.actionBtn.disabled = !enabled;
    elements.autoBtn.disabled = !enabled;
    elements.actionSelect.disabled = !enabled;
}

// ============== Chat Display ==============

function clearChat() {
    elements.chatContainer.innerHTML = '';
}

function addChatMessage(agentMsg, debtorMsg) {
    const turnDiv = document.createElement('div');
    turnDiv.className = 'chat-turn';
    
    // Agent message
    const agentDiv = document.createElement('div');
    agentDiv.className = 'chat-message agent-message';
    agentDiv.innerHTML = `<strong>🤖 Agent:</strong> ${agentMsg}`;
    turnDiv.appendChild(agentDiv);
    
    // Debtor message
    const debtorDiv = document.createElement('div');
    debtorDiv.className = 'chat-message debtor-message';
    debtorDiv.innerHTML = `<strong>👤 Debtor:</strong> ${debtorMsg}`;
    turnDiv.appendChild(debtorDiv);
    
    elements.chatContainer.appendChild(turnDiv);
    elements.chatContainer.scrollTop = elements.chatContainer.scrollHeight;
}

function renderConversation(messages) {
    clearChat();
    messages.forEach(msg => {
        addChatMessage(msg.agent_utterance, msg.debtor_response);
    });
}

// ============== State Display ==============

function renderState(state) {
    const html = `
        <table class="state-table">
            <tr>
                <td>📊 Turn</td>
                <td><strong>${state.turn}</strong> / ${state.max_turns}</td>
            </tr>
            <tr>
                <td>😊 Sentiment</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${(state.sentiment + 1) * 50}%; background: ${state.sentiment > 0 ? '#22c55e' : '#ef4444'}"></div>
                    </div>
                    <span>${state.sentiment.toFixed(2)}</span>
                </td>
            </tr>
            <tr>
                <td>🤝 Cooperation</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${state.cooperation * 100}%; background: #3b82f6"></div>
                    </div>
                    <span>${state.cooperation.toFixed(2)}</span>
                </td>
            </tr>
            <tr>
                <td>💬 Engagement</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${state.engagement * 100}%; background: #8b5cf6"></div>
                    </div>
                    <span>${state.engagement.toFixed(2)}</span>
                </td>
            </tr>
            <tr>
                <td>💳 Payment Plan Mentioned</td>
                <td>${state.mentioned_payment_plan ? '✅' : '❌'}</td>
            </tr>
            <tr>
                <td>📖 Shared Situation</td>
                <td>${state.shared_situation ? '✅' : '❌'}</td>
            </tr>
            <tr>
                <td>✅ Has Committed</td>
                <td class="${state.has_committed ? 'success' : ''}">${state.has_committed ? '🎉 YES!' : '❌ No'}</td>
            </tr>
        </table>
    `;
    elements.statePanel.innerHTML = html;
}

// ============== Q-Value Chart ==============

function initQValueChart() {
    const ctx = elements.qvalueChart.getContext('2d');
    qValueChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Q-Value',
                data: [],
                backgroundColor: [],
                borderColor: [],
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `Q-Value: ${ctx.raw.toFixed(3)}`
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255,255,255,0.1)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#e2e8f0', font: { size: 11 } }
                }
            }
        }
    });
}

function updateQValueChart(qValues) {
    if (!qValueChart) return;
    
    const labels = qValues.map(q => q.action_name);
    const data = qValues.map(q => q.value);
    const colors = qValues.map(q => q.is_best ? '#22c55e' : '#3b82f6');
    
    qValueChart.data.labels = labels;
    qValueChart.data.datasets[0].data = data;
    qValueChart.data.datasets[0].backgroundColor = colors;
    qValueChart.data.datasets[0].borderColor = colors;
    qValueChart.update();
}

// ============== Episode Status ==============

function renderEpisodeStatus(status) {
    let className = 'episode-status';
    if (status.is_done) {
        className += status.success ? ' success' : ' failure';
    }
    
    elements.episodeStatus.className = className;
    elements.episodeStatus.textContent = status.message;
}

// ============== Event Handlers ==============

async function handleLoadModel() {
    const modelType = elements.modelSelect.value;
    showLoading(`Loading ${modelType.toUpperCase()} model...`);
    
    try {
        const response = await api.loadModel(modelType);
        currentModel = modelType;
        showStatus(response.message, 'success');
    } catch (error) {
        showStatus(`Failed to load model: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handleStartConversation() {
    if (!currentModel) {
        showStatus('Please load a model first', 'error');
        return;
    }
    
    const persona = elements.personaSelect.value;
    showLoading(`Starting conversation with ${persona} debtor...`);
    
    try {
        const response = await api.startConversation(persona);
        
        isEpisodeDone = false;
        clearChat();
        renderState(response.state);
        updateQValueChart(response.q_values);
        renderEpisodeStatus({ is_done: false, message: response.message });
        setButtonsEnabled(true);
        showStatus(response.message, 'success');
    } catch (error) {
        showStatus(`Failed to start conversation: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handleTakeAction() {
    if (isEpisodeDone) {
        showStatus('Episode ended. Start a new conversation.', 'warning');
        return;
    }
    
    const actionId = parseInt(elements.actionSelect.value);
    showLoading('Processing action...');
    
    try {
        const response = await api.takeAction(actionId);
        
        renderConversation(response.conversation);
        renderState(response.state);
        updateQValueChart(response.q_values);
        renderEpisodeStatus(response.status);
        
        isEpisodeDone = response.status.is_done;
        if (isEpisodeDone) {
            setButtonsEnabled(false);
        }
        
        showStatus(response.status.message, response.status.is_done ? (response.status.success ? 'success' : 'error') : 'info');
    } catch (error) {
        showStatus(`Action failed: ${error.message}`, 'error');
    } finally {
        hideLoading();
    }
}

async function handleAutoPlay() {
    if (isEpisodeDone) {
        showStatus('Episode ended. Start a new conversation.', 'warning');
        return;
    }
    
    showLoading('Agent playing episode...');
    setButtonsEnabled(false);
    
    try {
        const response = await api.autoPlay();
        
        renderConversation(response.conversation);
        renderState(response.state);
        renderEpisodeStatus(response.status);
        
        isEpisodeDone = true;
        showStatus(`${response.status.message} | Total Reward: ${response.total_reward.toFixed(2)}`, 
                   response.status.success ? 'success' : 'error');
    } catch (error) {
        showStatus(`Auto-play failed: ${error.message}`, 'error');
        setButtonsEnabled(true);
    } finally {
        hideLoading();
    }
}

// ============== Initialization ==============

async function init() {
    // Initialize chart
    initQValueChart();
    
    // Bind event handlers
    elements.loadBtn.addEventListener('click', handleLoadModel);
    elements.startBtn.addEventListener('click', handleStartConversation);
    elements.actionBtn.addEventListener('click', handleTakeAction);
    elements.autoBtn.addEventListener('click', handleAutoPlay);
    
    // Check API health
    try {
        const health = await api.healthCheck();
        console.log('API Health:', health);
        
        if (health.models_loaded.dqn || health.models_loaded.ddq) {
            currentModel = health.models_loaded.ddq ? 'ddq' : 'dqn';
            showStatus(`✅ Ready - ${currentModel.toUpperCase()} model loaded`, 'success');
        } else {
            showStatus('⚠️ No model loaded. Click "Load Model" to start.', 'warning');
        }
    } catch (error) {
        showStatus('❌ Cannot connect to API server', 'error');
    }
    
    hideLoading();
}

// Start app when DOM is ready
document.addEventListener('DOMContentLoaded', init);
