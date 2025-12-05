/**
 * Adversarial Arena JavaScript
 * WebSocket connection for self-play training with dual-agent visualization
 */

// DOM Elements
const generationsInput = document.getElementById('generations-input');
const episodesInput = document.getElementById('episodes-input');
const useLlmCheckbox = document.getElementById('use-llm');
const zeroSumCheckbox = document.getElementById('zero-sum');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const statusBar = document.getElementById('status-bar');
const currentGen = document.getElementById('current-gen');
const currentEpisode = document.getElementById('current-episode');
const totalBattles = document.getElementById('total-battles');
const collectorWinrate = document.getElementById('collector-winrate');
const adversaryWinrate = document.getElementById('adversary-winrate');
const battleDialogue = document.getElementById('battle-dialogue');
const loadingOverlay = document.getElementById('loading-overlay');

// State
let ws = null;
let winrateChart = null;
let rewardChart = null;
let generationHistory = [];
let battleCount = 0;

// Collector strategies (from EnvironmentConfig)
const COLLECTOR_STRATEGIES = [
    'empathetic_listening',
    'ask_about_situation',
    'firm_reminder',
    'offer_payment_plan',
    'propose_settlement',
    'hard_close',
    'acknowledge_validate',
    'validate_then_offer',
    'empathy_sandwich'
];

// Adversary strategies (from SelfPlayConfig)
const ADVERSARY_STRATEGIES = [
    'aggressive',
    'evasive',
    'emotional',
    'negotiate_hard',
    'partial_cooperate',
    'stall',
    'dispute'
];

// Initialize charts
function initCharts() {
    // Win Rate Chart
    const winrateCtx = document.getElementById('winrate-chart').getContext('2d');
    winrateChart = new Chart(winrateCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Collector Win Rate',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: true,
                    tension: 0.3
                },
                {
                    label: 'Adversary Win Rate',
                    data: [],
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    fill: true,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Generation' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Win Rate' },
                    min: 0,
                    max: 1,
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            },
            plugins: {
                legend: { position: 'top' }
            }
        }
    });

    // Reward Chart
    const rewardCtx = document.getElementById('reward-chart').getContext('2d');
    rewardChart = new Chart(rewardCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Collector Reward',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Adversary Reward',
                    data: [],
                    borderColor: '#f44336',
                    backgroundColor: 'rgba(244, 67, 54, 0.1)',
                    fill: false,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Generation' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Avg Reward' },
                    grid: { color: 'rgba(255,255,255,0.1)' }
                }
            },
            plugins: {
                legend: { position: 'top' }
            }
        }
    });
}

// Update charts with generation data
function updateCharts(genData) {
    generationHistory.push(genData);

    // Update labels
    const labels = generationHistory.map((_, i) => `Gen ${i + 1}`);

    // Win rate chart
    winrateChart.data.labels = labels;
    winrateChart.data.datasets[0].data = generationHistory.map(g => g.collector_win_rate);
    winrateChart.data.datasets[1].data = generationHistory.map(g => g.adversary_win_rate);
    winrateChart.update('none');

    // Reward chart
    rewardChart.data.labels = labels;
    rewardChart.data.datasets[0].data = generationHistory.map(g => g.avg_collector_reward);
    rewardChart.data.datasets[1].data = generationHistory.map(g => g.avg_adversary_reward);
    rewardChart.update('none');
}

// Update strategy distribution bars
function updateStrategyBars(containerId, strategies, distribution, color) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    strategies.forEach(strategy => {
        const pct = distribution[strategy] || 0;
        const bar = document.createElement('div');
        bar.className = 'strategy-bar';
        bar.innerHTML = `
            <span class="strategy-name">${strategy.substring(0, 12)}</span>
            <div class="strategy-fill" style="width: ${pct * 100}%; background: ${color};"></div>
            <span class="strategy-pct">${(pct * 100).toFixed(0)}%</span>
        `;
        container.appendChild(bar);
    });
}

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/selfplay/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        statusBar.textContent = 'Connected to self-play server';
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
            statusBar.textContent = data.message || 'Self-play training started';
            statusBar.className = 'status-bar success';
            startBtn.disabled = true;
            stopBtn.disabled = false;
            loadingOverlay.style.display = 'none';
            break;

        case 'generation':
            currentGen.textContent = data.generation;

            // Update win rates
            collectorWinrate.textContent = `${(data.collector_win_rate * 100).toFixed(1)}%`;
            adversaryWinrate.textContent = `${(data.adversary_win_rate * 100).toFixed(1)}%`;

            // Update charts
            updateCharts(data);

            // Update strategy distributions if provided
            if (data.collector_strategy_dist) {
                updateStrategyBars('collector-strategies', COLLECTOR_STRATEGIES.slice(0, 4),
                    data.collector_strategy_dist, '#4CAF50');
            }
            if (data.adversary_strategy_dist) {
                updateStrategyBars('adversary-strategies', ADVERSARY_STRATEGIES.slice(0, 4),
                    data.adversary_strategy_dist, '#f44336');
            }
            break;

        case 'episode':
            currentEpisode.textContent = data.episode;
            battleCount++;
            totalBattles.textContent = battleCount;
            break;

        case 'battle':
            addBattleDialogue(data);
            break;

        case 'complete':
            statusBar.textContent = data.message || 'Training complete!';
            statusBar.className = 'status-bar success';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;

        case 'stopped':
            statusBar.textContent = data.message || 'Training stopped';
            statusBar.className = 'status-bar warning';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            break;

        case 'error':
            statusBar.textContent = `Error: ${data.message}`;
            statusBar.className = 'status-bar error';
            startBtn.disabled = false;
            stopBtn.disabled = true;
            loadingOverlay.style.display = 'none';
            break;
    }
}

// Add battle dialogue
function addBattleDialogue(data) {
    // Clear placeholder if first message
    if (battleDialogue.querySelector('p[style]')) {
        battleDialogue.innerHTML = '';
    }

    const battleDiv = document.createElement('div');
    battleDiv.className = 'battle-turn';
    battleDiv.innerHTML = `
        <div class="battle-message collector-msg">
            <div class="speaker">
                🎯 Collector 
                <span class="strategy-tag">${data.collector_strategy || 'unknown'}</span>
            </div>
            <div class="content">${data.collector_utterance || '[action taken]'}</div>
        </div>
        <div class="battle-message adversary-msg">
            <div class="speaker">
                🛡️ Adversary
                <span class="strategy-tag">${data.adversary_strategy || 'unknown'}</span>
            </div>
            <div class="content">${data.adversary_response || '[response]'}</div>
        </div>
    `;

    battleDialogue.appendChild(battleDiv);
    battleDialogue.scrollTop = battleDialogue.scrollHeight;

    // Keep only last 15 battle turns
    while (battleDialogue.children.length > 15) {
        battleDialogue.removeChild(battleDialogue.firstChild);
    }
}

// Start self-play training
function startSelfPlay() {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        connectWebSocket();
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
    document.getElementById('loading-text').textContent = 'Initializing self-play...';

    // Reset state
    generationHistory = [];
    battleCount = 0;
    currentGen.textContent = '0';
    currentEpisode.textContent = '0';
    totalBattles.textContent = '0';
    collectorWinrate.textContent = '0%';
    adversaryWinrate.textContent = '0%';

    if (winrateChart) {
        winrateChart.data.labels = [];
        winrateChart.data.datasets.forEach(ds => ds.data = []);
        winrateChart.update();
    }
    if (rewardChart) {
        rewardChart.data.labels = [];
        rewardChart.data.datasets.forEach(ds => ds.data = []);
        rewardChart.update();
    }

    battleDialogue.innerHTML = '<p style="color: var(--text-muted); text-align: center; padding: 30px;">Starting battles...</p>';

    ws.send(JSON.stringify({
        action: 'start',
        generations: parseInt(generationsInput.value),
        episodes_per_gen: parseInt(episodesInput.value),
        use_llm: useLlmCheckbox.checked,
        zero_sum: zeroSumCheckbox.checked
    }));
}

// Stop training
function stopSelfPlay() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'stop' }));
        statusBar.textContent = 'Stopping...';
    }
}

// Event listeners
startBtn.addEventListener('click', startSelfPlay);
stopBtn.addEventListener('click', stopSelfPlay);

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    connectWebSocket();
});

// Add status bar styles
const style = document.createElement('style');
style.textContent = `
    .status-bar.success {
        background: rgba(76, 175, 80, 0.2);
        border-color: #4CAF50;
    }
    .status-bar.error {
        background: rgba(244, 67, 54, 0.2);
        border-color: #f44336;
    }
    .status-bar.warning {
        background: rgba(255, 152, 0, 0.2);
        border-color: #ff9800;
    }
    .battle-turn {
        margin-bottom: 15px;
        padding-bottom: 15px;
        border-bottom: 1px solid var(--border-color);
    }
`;
document.head.appendChild(style);
