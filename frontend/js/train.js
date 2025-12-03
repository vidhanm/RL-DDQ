/**
 * Training Page JavaScript
 * WebSocket connection for live training updates
 */

// DOM Elements
const algorithmSelect = document.getElementById('algorithm-select');
const episodesInput = document.getElementById('episodes-input');
const difficultySelect = document.getElementById('difficulty-select');
const useLlmCheckbox = document.getElementById('use-llm');
const startBtn = document.getElementById('start-training-btn');
const stopBtn = document.getElementById('stop-training-btn');
const statusBar = document.getElementById('status-bar');
const episodeCount = document.getElementById('episode-count');
const totalEpisodes = document.getElementById('total-episodes');
const progressBar = document.getElementById('progress-bar');
const successRate = document.getElementById('success-rate');
const lastReward = document.getElementById('last-reward');
const trainingStatus = document.getElementById('training-status');
const dialogueContainer = document.getElementById('dialogue-container');
const loadingOverlay = document.getElementById('loading-overlay');

// State
let ws = null;
let rewardChart = null;
let rewardHistory = [];

// Initialize reward chart
function initChart() {
    const ctx = document.getElementById('reward-chart').getContext('2d');
    rewardChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Episode Reward',
                data: [],
                borderColor: '#3b82f6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Episode' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Reward' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Update chart with new data
function updateChart(episode, reward) {
    rewardHistory.push({ episode, reward });
    
    // Keep last 100 points
    if (rewardHistory.length > 100) {
        rewardHistory.shift();
    }
    
    rewardChart.data.labels = rewardHistory.map(r => r.episode);
    rewardChart.data.datasets[0].data = rewardHistory.map(r => r.reward);
    rewardChart.update('none');
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/training/ws`;
    
    ws = new WebSocket(wsUrl);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        statusBar.textContent = 'Connected to training server';
        statusBar.className = 'status-bar success';
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        statusBar.textContent = 'Disconnected from server';
        statusBar.className = 'status-bar error';
        ws = null;
    };
    
    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        statusBar.textContent = 'Connection error';
        statusBar.className = 'status-bar error';
    };
}

// Handle incoming WebSocket messages
function handleMessage(data) {
    switch (data.type) {
        case 'started':
            trainingStatus.textContent = 'Training...';
            statusBar.textContent = data.message;
            statusBar.className = 'status-bar';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            loadingOverlay.style.display = 'none';
            break;
            
        case 'episode':
            episodeCount.textContent = data.episode;
            totalEpisodes.textContent = data.total_episodes;
            progressBar.style.width = `${(data.episode / data.total_episodes) * 100}%`;
            lastReward.textContent = data.reward.toFixed(2);
            successRate.textContent = `${(data.success_rate * 100).toFixed(1)}%`;
            successRate.style.color = data.success ? '#22c55e' : '#ef4444';
            updateChart(data.episode, data.reward);
            break;
            
        case 'dialogue':
            addDialogue(data.episode, data.agent_utterance, data.debtor_response);
            break;
            
        case 'complete':
            trainingStatus.textContent = 'Complete!';
            statusBar.textContent = data.message;
            statusBar.className = 'status-bar success';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;
            
        case 'stopped':
            trainingStatus.textContent = 'Stopped';
            statusBar.textContent = data.message;
            statusBar.className = 'status-bar warning';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;
            
        case 'stopping':
            trainingStatus.textContent = 'Stopping...';
            statusBar.textContent = data.message;
            break;
            
        case 'error':
            trainingStatus.textContent = 'Error';
            statusBar.textContent = `Error: ${data.message}`;
            statusBar.className = 'status-bar error';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            loadingOverlay.style.display = 'none';
            break;
            
        case 'status':
            if (data.is_training) {
                trainingStatus.textContent = 'Training...';
                episodeCount.textContent = data.current_episode;
                totalEpisodes.textContent = data.total_episodes;
                successRate.textContent = `${(data.success_rate * 100).toFixed(1)}%`;
            }
            break;
    }
}

// Add dialogue to the container
function addDialogue(episode, agentUtterance, debtorResponse) {
    // Clear placeholder if first message
    if (dialogueContainer.querySelector('p[style]')) {
        dialogueContainer.innerHTML = '';
    }
    
    const turnDiv = document.createElement('div');
    turnDiv.className = 'dialogue-turn';
    turnDiv.innerHTML = `
        <div class="episode-marker">Episode ${episode}</div>
        <div class="message agent">
            <strong>🤖 Agent:</strong> ${agentUtterance || '[Action taken]'}
        </div>
        <div class="message debtor">
            <strong>👤 Debtor:</strong> ${debtorResponse || '[Response]'}
        </div>
    `;
    
    dialogueContainer.appendChild(turnDiv);
    dialogueContainer.scrollTop = dialogueContainer.scrollHeight;
    
    // Keep only last 20 dialogue turns
    while (dialogueContainer.children.length > 20) {
        dialogueContainer.removeChild(dialogueContainer.firstChild);
    }
}

// Start training
function startTraining() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
        // Wait for connection then send
        setTimeout(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                sendStartCommand();
            }
        }, 1000);
    } else {
        sendStartCommand();
    }
}

function sendStartCommand() {
    loadingOverlay.style.display = 'flex';
    document.getElementById('loading-text').textContent = 'Starting training...';
    
    // Reset state
    rewardHistory = [];
    if (rewardChart) {
        rewardChart.data.labels = [];
        rewardChart.data.datasets[0].data = [];
        rewardChart.update();
    }
    dialogueContainer.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding-top: 50px;">Waiting for dialogues...</p>';
    
    ws.send(JSON.stringify({
        action: 'start',
        algorithm: algorithmSelect.value,
        episodes: parseInt(episodesInput.value),
        difficulty: difficultySelect.value,
        use_llm: useLlmCheckbox.checked
    }));
}

// Stop training
function stopTraining() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'stop' }));
    }
}

// Event listeners
startBtn.addEventListener('click', startTraining);
stopBtn.addEventListener('click', stopTraining);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initChart();
    connectWebSocket();
});

// Add some styles for dialogue
const style = document.createElement('style');
style.textContent = `
    .dialogue-turn {
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
    .episode-marker {
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
    .status-bar.warning {
        background: rgba(249, 115, 22, 0.2);
        border-color: #f97316;
    }
`;
document.head.appendChild(style);
