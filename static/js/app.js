// vLLM Manager - 前端邏輯

const API = {
    status: '/api/status',
    start: '/api/start',
    stop: '/api/stop',
    restart: '/api/restart',
    config: '/api/config',
    modelParams: '/api/model-params',
    models: '/api/models',
    modelsScan: '/api/models/scan',
    modelsClone: '/api/models/clone',
    cloneStatus: '/api/clone-status',
    logs: '/api/logs',
    logsStream: '/api/logs/stream',
    gpu: '/api/gpu',
    cpu: '/api/cpu',
};
let refreshTimer = null;
let paramsHasChanges = false;

// ---- 未儲存提示 ----

function markParamsChanged() {
    if (!paramsHasChanges) {
        paramsHasChanges = true;
        const badge = document.getElementById('unsaved-badge');
        if (badge) badge.style.display = 'inline';
    }
}

function clearParamsChanged() {
    paramsHasChanges = false;
    const badge = document.getElementById('unsaved-badge');
    if (badge) badge.style.display = 'none';
}

// 離開頁面時提醒
window.addEventListener('beforeunload', (e) => {
    if (paramsHasChanges) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// ---- 初始化 ----

document.addEventListener('DOMContentLoaded', () => {
    loadModelParams();
    refreshAll();
    // 每 5 秒自動刷新狀態
    refreshTimer = setInterval(refreshAll, 5000);
    
    // 監聽所有參數欄位的變動
    const form = document.getElementById('params-form');
    if (form) {
        form.addEventListener('input', markParamsChanged);
        form.addEventListener('change', markParamsChanged);
    }

    // MTP checkbox toggle
    const specCheckbox = document.getElementById('param-speculative-enabled');
    if (specCheckbox) {
        specCheckbox.addEventListener('change', (e) => {
            toggleSpeculativeOptions(e.target.checked);
        });
    }
});

async function loadModelParams() {
    const resp = await fetchJSON(API.modelParams);
    if (!resp || resp.status !== 'ok') {
        const label = document.getElementById('model-params-label');
        if (label) label.textContent = resp && resp.message ? ` (${resp.message})` : ' (無啟用模型)';
        clearParamsChanged();
        return;
    }
    const p = resp.params;
    const label = document.getElementById('model-params-label');
    if (label) label.textContent = ` — ${resp.name || resp.model}`;
    
    setVal('param-port', p.port);
    setVal('param-host', p.host);
    setVal('param-dtype', p.dtype);
    setVal('param-max-model-len', Math.round((p.max_model_len || 128000) / 1000));
    setVal('param-gpu-memory-utilization', p.gpu_memory_utilization);
    setVal('param-max-num-seqs', p.max_num_seqs);
    setVal('param-max-num-batched-tokens', p.max_num_batched_tokens);
    setVal('param-kv-cache-dtype', p.kv_cache_dtype);
    setVal('param-tool-call-parser', p.tool_call_parser);
    setChecked('param-enable-prefix-caching', p.enable_prefix_caching);
    setChecked('param-enable-auto-tool-choice', p.enable_auto_tool_choice);
    setChecked('param-trust-remote-code', p.trust_remote_code);
    
    // Speculative decoding (MTP)
    const specEnabled = p.speculative_enabled || false;
    setChecked('param-speculative-enabled', specEnabled);
    setVal('param-speculative-method', p.speculative_method || 'qwen3_next_mtp');
    setVal('param-speculative-num-tokens', p.speculative_num_tokens || 1);
    toggleSpeculativeOptions(specEnabled);
    
    clearParamsChanged();
}

function toggleSpeculativeOptions(enabled) {
    const options = document.getElementById('speculative-options');
    if (options) {
        options.style.display = enabled ? 'block' : 'none';
    }
}

function toggleMtpTuning() {
    const content = document.getElementById('mtp-tuning-content');
    const arrow = document.getElementById('mtp-tuning-arrow');
    if (content) {
        if (content.style.display === 'none') {
            content.style.display = 'block';
            arrow.textContent = '▲';
        } else {
            content.style.display = 'none';
            arrow.textContent = '▼';
        }
    }
}

// ---- 刷新 ----

async function refreshAll() {
    const s = await fetchJSON(API.status);
    const running = s?.running || false;
    renderStatus(s);
    refreshModels(running);
    refreshGPU();
    refreshCPU();
    if (!logEventSource) refreshLogs();
}

function renderStatus(s) {
    if (!s) return;

    const badge = document.getElementById('status-badge');
    badge.textContent = s.running ? '運行中' : '停止';
    badge.className = 'badge ' + (s.running ? 'badge-running' : 'badge-stopped');

    document.getElementById('status-pid').textContent = s.pid || '-';

    const uptimeEl = document.getElementById('status-uptime');
    if (s.uptime && s.uptime > 0) {
        const hrs = Math.floor(s.uptime / 3600);
        const mins = Math.floor((s.uptime % 3600) / 60);
        const secs = s.uptime % 60;
        uptimeEl.textContent = `${hrs}h ${mins}m ${secs}s`;
    } else {
        uptimeEl.textContent = '-';
    }

    document.getElementById('last-updated').textContent =
        '最後更新: ' + new Date().toLocaleTimeString('zh-TW');
}

async function refreshModels(running) {
    const models = await fetchJSON(API.models);
    renderModels(models, running);
}

function renderModels(models, running) {
    const container = document.getElementById('model-list');
    if (!models || models.length === 0) {
        container.innerHTML = '<p style="color:#8b949e">尚無模型</p>';
        return;
    }

    container.innerHTML = models.map(m => {
        let btn = '';
        if (m.enabled && running) {
            btn = `<button class="btn btn-sm btn-red" onclick="stopVLLM()">停止</button>`;
        } else if (m.enabled) {
            btn = `<button class="btn btn-sm btn-green" onclick="startModel('${esc(m.path)}')">啟動</button>`;
        } else {
            btn = `<button class="btn btn-sm btn-blue" onclick="enableModel('${esc(m.path)}')">啟用</button>`;
        }
        return `
        <div class="model-item">
            <div class="model-info">
                <div class="model-name">${esc(m.name)} ${m.enabled ? '<span class="badge badge-running" style="font-size:0.7em">啟用</span>' : ''}</div>
                <div class="model-path">${esc(m.path)}</div>
            </div>
            <div class="model-actions">
                ${btn}
                <button class="btn btn-sm btn-red" onclick="removeModel('${esc(m.path)}')">移除</button>
            </div>
        </div>
    `}).join('');
}

async function refreshGPU() {
    const gpus = await fetchJSON(API.gpu);
    const container = document.getElementById('status-gpu');
    const tempEl = document.getElementById('status-temp');
    const gpuInfo = document.getElementById('gpu-info');

    if (!gpus || gpus.length === 0 || gpus[0].error) {
        container.textContent = '-- / -- GB';
        tempEl.textContent = '--°C';
        gpuInfo.innerHTML = `<div class="sys-card"><div class="sys-title">無法讀取 GPU 資訊</div></div>`;
        return;
    }

    let html = '';
    for (const g of gpus) {
        container.textContent = `${(g.memory_used / 1024).toFixed(1)} / ${(g.memory_total / 1024).toFixed(1)} GB`;
        tempEl.textContent = `${g.temperature}°C`;

        const utilColor = g.utilization > 90 ? '#f85149' : g.utilization > 70 ? '#d29922' : '#3fb950';
        const tempColor = g.temperature > 85 ? '#f85149' : g.temperature > 75 ? '#d29922' : '#3fb950';
        const memPct = g.memory_total > 0 ? (g.memory_used / g.memory_total * 100).toFixed(0) : 0;
        const memColor = memPct > 90 ? '#f85149' : memPct > 70 ? '#d29922' : '#3fb950';

        html += `
        <div class="sys-card">
            <div class="sys-title">GPU 0: ${esc(g.name)}</div>
            <div class="sys-row">
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${memColor}">${(g.memory_used / 1024).toFixed(1)}<span class="sys-unit">GB</span></div>
                    <div class="sys-metric-label">顯存 ${memPct}%</div>
                    <div class="sys-bar"><div class="sys-bar-fill" style="width:${memPct}%;background:${memColor}"></div></div>
                </div>
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${utilColor}">${g.utilization}<span class="sys-unit">%</span></div>
                    <div class="sys-metric-label">GPU 使用率</div>
                    <div class="sys-bar"><div class="sys-bar-fill" style="width:${g.utilization}%;background:${utilColor}"></div></div>
                </div>
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${tempColor}">${g.temperature}<span class="sys-unit">°C</span></div>
                    <div class="sys-metric-label">溫度</div>
                    <div class="sys-bar"><div class="sys-bar-fill" style="width:${g.temperature};background:${tempColor}"></div></div>
                </div>
                <div class="sys-metric">
                    <div class="sys-metric-value">${g.power_draw.toFixed(0)}<span class="sys-unit">W</span></div>
                    <div class="sys-metric-label">功耗</div>
                    <div class="sys-bar" style="visibility:hidden"><div class="sys-bar-fill" style="width:${(g.power_draw / g.power_limit * 100).toFixed(0)}%"></div></div>
                </div>
            </div>
        </div>`;
    }
    gpuInfo.innerHTML = html;
}

async function refreshCPU() {
    const info = await fetchJSON(API.cpu);
    const container = document.getElementById('cpu-info');

    if (!info || info.error) {
        container.innerHTML = `<div class="sys-card"><div class="sys-title">${info?.error ? '錯誤: ' + info.error : '無法讀取系統資訊'}</div></div>`;
        return;
    }

    const memPct = info.mem_total_mb > 0 ? ((info.mem_used_mb / info.mem_total_mb) * 100).toFixed(0) : 0;
    const loadColor = info.load_avg_1m > info.cpu_count ? '#f85149' : info.load_avg_1m > info.cpu_count * 0.7 ? '#d29922' : '#3fb950';
    const memColor = memPct > 90 ? '#f85149' : memPct > 70 ? '#d29922' : '#3fb950';
    const cpuColor = info.cpu_percent > 90 ? '#f85149' : info.cpu_percent > 70 ? '#d29922' : '#3fb950';

    const tempRows = Object.entries(info.temperatures || {}).map(([k, v]) => {
        const c = v > 85 ? '#f85149' : v > 75 ? '#d29922' : '#3fb950';
        return `<div class="sys-metric">
            <div class="sys-metric-value" style="color:${c}">${v}<span class="sys-unit">°C</span></div>
            <div class="sys-metric-label">${esc(k)}</div>
        </div>`;
    }).join('');

    container.innerHTML = `
        <div class="sys-card">
            <div class="sys-title">系統資源</div>
            <div class="sys-row">
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${cpuColor}">${info.cpu_percent}<span class="sys-unit">%</span></div>
                    <div class="sys-metric-label">CPU</div>
                </div>
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${loadColor}">${info.load_avg_1m}</div>
                    <div class="sys-metric-label">Load 1m</div>
                </div>
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${loadColor}">${info.load_avg_5m}</div>
                    <div class="sys-metric-label">Load 5m</div>
                </div>
                <div class="sys-metric">
                    <div class="sys-metric-value" style="color:${memColor}">${(info.mem_used_mb / 1024).toFixed(1)}<span class="sys-unit">GB</span></div>
                    <div class="sys-metric-label">RAM ${memPct}%</div>
                    <div class="sys-bar"><div class="sys-bar-fill" style="width:${memPct}%;background:${memColor}"></div></div>
                </div>
                ${tempRows}
            </div>
        </div>`;
}

let logEventSource = null;
let logLines = [];
let logReconnectAttempts = 0;
const MAX_LOG_RECONNECT = 5;

function toggleLogStream() {
    const btn = document.querySelector('button[onclick="toggleLogStream()"]');
    const badge = document.getElementById('log-status');
    
    if (logEventSource) {
        logEventSource.close();
        logEventSource = null;
        logReconnectAttempts = MAX_LOG_RECONNECT;
        badge.textContent = '串流關閉';
        badge.className = 'badge badge-stopped';
        btn.textContent = '開始串流';
    } else {
        logLines = [];
        logReconnectAttempts = 0;
        document.getElementById('log-output').innerHTML = '<div class="log-line">正在連接串流...</div>';
        
        connectLogStream();
        
        badge.textContent = '串流中';
        badge.className = 'badge badge-running';
        btn.textContent = '停止串流';
    }
}

function connectLogStream() {
    logEventSource = new EventSource('/api/logs/stream');
    logEventSource.onmessage = function(e) {
        logReconnectAttempts = 0;
        logLines.push(e.data);
        if (logLines.length > 500) logLines = logLines.slice(-500);
        
        const container = document.getElementById('log-output');
        container.innerHTML = logLines.map(l => `<div class="log-line">${esc(l)}</div>`).join('');
        container.scrollTop = container.scrollHeight;
    };
    logEventSource.onerror = function() {
        logEventSource.close();
        logEventSource = null;
        logReconnectAttempts++;
        const badge = document.getElementById('log-status');
        const btn = document.querySelector('button[onclick="toggleLogStream()"]');
        if (logReconnectAttempts < MAX_LOG_RECONNECT) {
            badge.textContent = `串流斷線 (重連 ${logReconnectAttempts}/${MAX_LOG_RECONNECT})`;
            setTimeout(connectLogStream, 2000);
        } else {
            badge.textContent = '串流斷線';
            badge.className = 'badge badge-stopped';
            btn.textContent = '開始串流';
        }
    };
}

async function refreshLogs() {
    const n = document.getElementById('log-lines').value;
    const logs = await fetchJSON(`${API.logs}?n=${n}`);
    const container = document.getElementById('log-output');
    if (!logs || logs.length === 0) {
        container.innerHTML = '<div class="log-line">尚無日誌</div>';
        return;
    }
    container.innerHTML = logs.map(l => `<div class="log-line">${esc(l)}</div>`).join('');
    container.scrollTop = container.scrollHeight;
}

// ---- 控制 ----

async function startVLLM() {
    const btn = document.querySelector('button[onclick="startVLLM()"]');
    if (btn) btn.disabled = true;
    showStartingState();
    const resp = await fetchJSON(API.start, { method: 'POST', body: JSON.stringify({}) });
    if (btn) btn.disabled = false;
    if (resp) {
        if (resp.status === 'starting') {
            await waitForRunning(15000);
        } else if (resp.status !== 'already_running') {
            alert(`vLLM ${resp.status}: ${resp.message || '啟動成功'}`);
        }
        refreshAll();
    }
}

async function startModel(path) {
    showStartingState();
    const resp = await fetchJSON(API.start, { method: 'POST', body: JSON.stringify({ model_path: path }) });
    if (resp) {
        if (resp.status === 'starting') {
            await waitForRunning(15000);
        } else if (resp.status !== 'already_running') {
            alert(`vLLM ${resp.status}: ${resp.message || '啟動成功'}`);
        }
        refreshAll();
    }
}

async function stopVLLM() {
    const btn = document.querySelector('button[onclick="stopVLLM()"]');
    if (btn) btn.disabled = true;
    showStoppingState();
    const resp = await fetchJSON(API.stop, { method: 'POST', body: JSON.stringify({}) });
    if (btn) btn.disabled = false;
    if (resp) {
        alert(`vLLM ${resp.status}`);
        if (resp.status === 'stopped' || resp.status === 'not_running') {
            await waitForStopped(10000);
        }
        refreshAll();
    }
}

async function waitForRunning(timeout = 15000) {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
        await new Promise(r => setTimeout(r, 1000));
        const s = await fetchJSON(API.status);
        if (s?.running) return true;
    }
    return false;
}

async function waitForStopped(timeout = 10000) {
    const startTime = Date.now();
    while (Date.now() - startTime < timeout) {
        await new Promise(r => setTimeout(r, 500));
        const s = await fetchJSON(API.status);
        if (s && !s.running) return true;
    }
    return false;
}

function showStartingState() {
    const badge = document.getElementById('status-badge');
    badge.textContent = '啟動中';
    badge.className = 'badge badge-starting';
    document.getElementById('status-pid').textContent = '-';
    refreshModels(true);
}

function showStoppingState() {
    const badge = document.getElementById('status-badge');
    badge.textContent = '停止中';
    badge.className = 'badge badge-stopped';
    document.getElementById('status-pid').textContent = '-';
    refreshModels(false);
}

// ---- 模型管理 ----

async function addModel(e) {
    e.preventDefault();
    const name = document.getElementById('new-model-name').value;
    const path = document.getElementById('new-model-path').value;
    const resp = await fetchJSON(API.models, {
        method: 'POST',
        body: JSON.stringify({ name, path })
    });
    if (resp && resp.status === 'added') {
        document.getElementById('new-model-name').value = '';
        document.getElementById('new-model-path').value = '';
        refreshModels();
    } else if (resp && resp.status === 'exists') {
        alert('模型已存在');
    }
}

async function enableModel(path) {
    const resp = await fetchJSON(`${API.models}/enable`, {
        method: 'POST',
        body: JSON.stringify({ path })
    });
    if (resp) {
        refreshAll();
        loadModelParams();
    }
}

async function removeModel(path) {
    if (!confirm(`確定要移除模型: ${path} ?`)) return;
    const resp = await fetchJSON(`${API.models}/remove`, {
        method: 'POST',
        body: JSON.stringify({ path })
    });
    if (resp) {
        alert(`移除結果: ${resp.status}`);
    }
    refreshModels();
}

async function scanModels() {
    const btn = event.target;
    btn.textContent = '掃描中...';
    btn.disabled = true;
    try {
        const result = await fetchJSON(API.modelsScan, { method: 'POST' });
        if (result.status === 'scanned') {
            alert(`掃描完成！共 ${result.total} 個模型，新增 ${result.new} 個`);
        } else {
            alert(`掃描失敗: ${result.message}`);
        }
        refreshModels();
    } catch (e) {
        alert(`掃描失敗: ${e.message}`);
    } finally {
        btn.textContent = '掃描目錄';
        btn.disabled = false;
    }
}

let clonePollInterval = null;

async function cloneModel() {
    const repo = document.getElementById('hf-repo').value;
    if (!repo) {
        alert('請輸入 Hugging Face repo 名稱');
        return;
    }

    const container = document.getElementById('clone-progress-container');
    const statusText = document.getElementById('clone-status-text');
    const percentage = document.getElementById('clone-percentage');
    const bar = document.getElementById('clone-progress-bar');
    const btn = event.target;
    
    btn.disabled = true;
    btn.textContent = 'Clone 中...';
    container.style.display = 'block';
    bar.style.width = '0%';
    percentage.textContent = '0%';
    statusText.textContent = `開始 clone: ${repo}...`;

    // Start polling
    if (clonePollInterval) clearInterval(clonePollInterval);
    clonePollInterval = setInterval(async () => {
        const status = await fetchJSON(API.cloneStatus);
        
        // Update progress bar
        bar.style.width = `${status.percentage || 0}%`;
        percentage.textContent = `${status.percentage || 0}%`;
        if (status.progress) {
            statusText.textContent = status.progress;
        }
        
        if (!status.running) {
            clearInterval(clonePollInterval);
            btn.disabled = false;
            btn.textContent = 'Clone';
            
            if (status.result && status.result.status === 'cloned') {
                statusText.textContent = `✅ Clone 完成: ${status.result.name}`;
                percentage.textContent = '100%';
                bar.style.width = '100%';
                document.getElementById('hf-repo').value = '';
                refreshModels();
            } else if (status.result && status.result.status === 'error') {
                statusText.textContent = `❌ Clone 失敗: ${status.result.message}`;
                bar.style.background = 'var(--red)';
            } else {
                statusText.textContent = status.progress || 'Clone 完成';
            }
        }
    }, 1000);

    try {
        await fetchJSON(API.modelsClone, {
            method: 'POST',
            body: JSON.stringify({ repo_url: repo })
        });
    } catch (e) {
        alert(`Clone 啟動失敗: ${e.message}`);
        clearInterval(clonePollInterval);
        btn.disabled = false;
        btn.textContent = 'Clone';
        container.style.display = 'none';
    }
}

// ---- 參數 ----

async function saveParams(e) {
    e.preventDefault();
    const params = {
        port: parseInt(document.getElementById('param-port').value) || 8001,
        host: document.getElementById('param-host').value || '0.0.0.0',
        dtype: document.getElementById('param-dtype').value || 'float16',
        max_model_len: (parseInt(document.getElementById('param-max-model-len').value) || 128) * 1000,
        gpu_memory_utilization: parseFloat(document.getElementById('param-gpu-memory-utilization').value) || 0.9,
        max_num_seqs: parseInt(document.getElementById('param-max-num-seqs').value) || 4,
        max_num_batched_tokens: parseInt(document.getElementById('param-max-num-batched-tokens').value) || 0,
        kv_cache_dtype: document.getElementById('param-kv-cache-dtype').value || 'fp8',
        tool_call_parser: document.getElementById('param-tool-call-parser').value || 'qwen3_xml',
        enable_prefix_caching: document.getElementById('param-enable-prefix-caching').checked,
        enable_auto_tool_choice: document.getElementById('param-enable-auto-tool-choice').checked,
        trust_remote_code: document.getElementById('param-trust-remote-code').checked,
        speculative_enabled: document.getElementById('param-speculative-enabled').checked,
        speculative_method: document.getElementById('param-speculative-method').value || 'qwen3_next_mtp',
        speculative_num_tokens: parseInt(document.getElementById('param-speculative-num-tokens').value) || 1,
    };
    const resp = await fetchJSON(API.modelParams, {
        method: 'POST',
        body: JSON.stringify(params)
    });
    if (resp && resp.status === 'saved') {
        alert(`參數已儲存 (${resp.model})`);
        clearParamsChanged();
        loadModelParams();
    }
}

// ---- 工具函式 ----

async function fetchJSON(url, options = {}) {
    try {
        options.headers = { 'Content-Type': 'application/json' };
        const resp = await fetch(url, options);
        return await resp.json();
    } catch (e) {
        console.error('fetchJSON error:', e);
        return null;
    }
}

function setVal(id, val) {
    const el = document.getElementById(id);
    if (el) el.value = val;
}

function setChecked(id, val) {
    const el = document.getElementById(id);
    if (el) el.checked = val;
}

function esc(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
