// LOCCA Dashboard - Live monitoring JS (runs page)

let ws = null;
let eventCount = 0;
let lastSecondEvents = 0;
let errorCount = 0;
let lastMinuteErrors = [];
let activeSessions = [];

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/ws/live`;

    ws = new WebSocket(wsUrl);

    ws.onopen = function() {
        updateConnectionStatus('Connected', 'success');
        console.log('WebSocket connected');
    };

    ws.onmessage = function(event) {
        const message = JSON.parse(event.data);
        handleMessage(message);
    };

    ws.onclose = function() {
        updateConnectionStatus('Disconnected', 'danger');
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = function(error) {
        updateConnectionStatus('Error', 'danger');
        console.error('WebSocket error:', error);
    };
}

function updateConnectionStatus(text, statusClass) {
    const el = document.getElementById('connection-status');
    if (el) {
        el.className = `badge bg-${statusClass}`;
        el.innerHTML = `<i class="fas fa-circle me-1"></i> ${text}`;
    }
}

function handleMessage(message) {
    switch (message.type) {
        case 'stats_update':
            updateStats(message.data);
            break;
        case 'event':
            loadLiveRuns();
            loadLiveSessions();
            break;
        default:
            break;
    }
}

function updateStats(data) {
    document.getElementById('active-sessions-count').textContent = data.active_sessions || 0;
    document.getElementById('events-per-second').textContent = (data.events_per_second || 0).toFixed(1);
    document.getElementById('total-events').textContent = data.total_events || 0;
    document.getElementById('error-count').textContent = data.error_count_last_minute || 0;
}

function formatDateTime(dateString) {
    if (!dateString) return '-';
    return new Date(dateString).toLocaleString();
}

function formatDuration(seconds) {
    if (!seconds || seconds === 0) return '-';
    if (seconds < 60) return seconds.toFixed(1) + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ' + Math.floor(seconds % 60) + 's';
    return Math.floor(seconds / 3600) + 'h ' + Math.floor((seconds % 3600) / 60) + 'm';
}

function getStatusColor(status) {
    switch (status) {
        case 'completed': return 'success';
        case 'running': return 'primary';
        case 'error': return 'danger';
        case 'paused': return 'warning';
        default: return 'secondary';
    }
}

// Poll live runs data every 5 seconds
async function loadLiveRuns() {
    try {
        const response = await fetch('/api/live/runs');
        const runs = await response.json();
        displayLiveRuns(runs);
    } catch (e) {
        console.error('Error loading live runs:', e);
        const container = document.getElementById('live-runs-container');
        if (container) {
            container.innerHTML = '<div class="text-center text-danger py-4"><p>Failed to load runs</p></div>';
        }
    }
}

async function loadLiveSessions() {
    try {
        const response = await fetch('/api/live/sessions');
        const sessions = await response.json();
        displayLiveSessions(sessions);
    } catch (e) {
        console.error('Error loading live sessions:', e);
        const container = document.getElementById('active-sessions-container');
        if (container) {
            container.innerHTML = '<div class="text-center text-danger py-4"><p>Failed to load sessions</p></div>';
        }
    }
}

function displayLiveRuns(runs) {
    const container = document.getElementById('live-runs-container');
    if (!container) return;

    if (!runs || runs.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-4"><p>No active runs</p></div>';
        return;
    }

    const html = runs.map(run => `
        <div class="card mb-3 border-start border-${getStatusColor(run.status)} clickable-row" onclick="window.location='/runs/${run.run_id}'">
            <div class="card-body">
                <div class="row align-items-center">
                    <div class="col-md-4">
                        <h6 class="mb-1">Run: <code>${run.run_id.substring(0, 8)}...</code></h6>
                        <small class="text-muted">Session: ${run.session_id.substring(0, 8)}...</small>
                    </div>
                    <div class="col-md-3">
                        <span class="badge bg-${getStatusColor(run.status)} status-badge">${run.status}</span>
                        <br>
                        <small class="text-muted">Turn ${run.turn_index}</small>
                    </div>
                    <div class="col-md-2">
                        <span class="tokens-badge">${run.input_tokens.toLocaleString()} in / ${run.output_tokens.toLocaleString()} out</span>
                    </div>
                    <div class="col-md-3 text-end">
                        <a href="/runs/${run.run_id}" class="btn btn-sm btn-outline-primary">View Details</a>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
    container.innerHTML = html;
}

function displayLiveSessions(sessions) {
    const container = document.getElementById('active-sessions-container');
    if (!container) return;

    if (!sessions || sessions.length === 0) {
        container.innerHTML = '<div class="text-center text-muted py-4"><p>No active sessions</p></div>';
        return;
    }

    const html = sessions.map(s => `
        <div class="card mb-3 border-start border-success clickable-row" onclick="window.location='/sessions/${s.session_id}'">
            <div class="card-body">
                <div class="row align-items-center">
                    <div class="col-md-5">
                        <h6 class="mb-1">Session: <code>${s.session_id.substring(0, 8)}...</code></h6>
                        <small class="text-muted">Started: ${formatDateTime(s.started_at)}</small>
                    </div>
                    <div class="col-md-4">
                        <span class="badge bg-info">${s.events_count || 0} events</span>
                    </div>
                    <div class="col-md-3 text-end">
                        <a href="/sessions/${s.session_id}" class="btn btn-sm btn-outline-primary">View Details</a>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
    container.innerHTML = html;
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    connectWebSocket();
    loadLiveRuns();
    loadLiveSessions();
    setInterval(loadLiveRuns, 5000);
    setInterval(loadLiveSessions, 5000);

    // Manual refresh
    const refreshBtn = document.getElementById('refresh-runs-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', function() {
            loadLiveRuns();
            loadLiveSessions();
        });
    }

    // Periodic ping
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: 'ping'}));
        }
    }, 30000);

    window.addEventListener('beforeunload', function() {
        if (ws) ws.close();
    });
});
