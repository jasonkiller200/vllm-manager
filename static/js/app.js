// vLLM Manager - 前端邏輯

const API = {
    status: '/api/status',
    start: '/api/start',
    stop: '/api/stop',
    restart: '/api/restart',
    config: '/api/config',
    models: '/api/models',
    modelsScan: '/api/models/scan',
    modelsClone: '/api/models/clone',
    cloneStatus: '/api/clone-status',
    logs: '/api/logs',
    logsStream: '/api/logs/stream',
    gpu: '/api/gpu',
};
let refreshTimer = null;

// ---- 初始化 ----

document.addEventListener('DOMContentLoaded', () => {
    loadConfig();
    refreshAll();
    // 每 5 秒自動刷新狀態
    refreshTimer = setInterval(refreshStatus, 5000);
});

async function loadConfig() {
    const resp = await fetchJSON(API.config);
    if (resp && resp.defaults) {
        const d = resp.defaults;
        setVal('param-port', d.port);
        setVal('param-host', d.host);
        setVal('param-dtype', d.dtype);
        setVal('param-max-model-len', d.max_model_len);
        setVal('param-gpu-memory-utilization', d.gpu_memory_utilization);
        setVal('param-max-num-seqs', d.max_num_seqs);
        setVal('param-kv-cache-dtype', d.kv_cache_dtype);
        setVal('param-tool-call-parser', d.tool_call_parser);
        setChecked('param-enable-prefix-caching', d.enable_prefix_caching);
        setChecked('param-enable-auto-tool-choice', d.enable_auto_tool_choice);
        setChecked('param-trust-remote-code', d.trust_remote_code);
    }
}

// ---- 刷新 ----

async function refreshAll() {
    refreshStatus();
    refreshModels();
    refreshGPU();
    refreshLogs();
}

async function refreshStatus() {
    const s = await fetchJSON(API.status);
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

    // 按鈕狀態
    document.getElementById('btn-start').disabled = s.running;
    document.getElementById('btn-stop').disabled = !s.running;
    document.getElementById('btn-restart').disabled = !s.running;

    document.getElementById('last-updated').textContent =
        '最後更新: ' + new Date().toLocaleTimeString('zh-TW');
}

async function refreshModels() {
    const models = await fetchJSON(API.models);
    const container = document.getElementById('model-list');
    if (!models || models.length === 0) {
        container.innerHTML = '<p style="color:#8b949e">尚無模型</p>';
        return;
    }

    container.innerHTML = models.map(m => `
        <div class="model-item">
            <div class="model-info">
                <div class="model-name">${esc(m.name)} ${m.enabled ? '<span class="badge badge-running" style="font-size:0.7em">啟用</span>' : ''}</div>
                <div class="model-path">${esc(m.path)}</div>
            </div>
            <div class="model-actions">
                ${!m.enabled ? `<button class="btn btn-sm btn-blue" onclick="enableModel('${esc(m.path)}')">啟用</button>` : ''}
                <button class="btn btn-sm btn-red" onclick="removeModel('${esc(m.path)}')">移除</button>
            </div>
        </div>
    `).join('');
}

async function refreshGPU() {
    const gpus = await fetchJSON(API.gpu);
    const container = document.getElementById('status-gpu');
    const tempEl = document.getElementById('status-temp');
    const gpuInfo = document.getElementById('gpu-info');

    if (!gpus || gpus.length === 0 || gpus[0].error) {
        container.textContent = '-- / -- GB';
        tempEl.textContent = '--°C';
        gpuInfo.textContent = gpus && gpus[0].error ? `錯誤: ${gpus[0].error}` : '無法讀取 GPU 資訊';
        return;
    }

    let html = '';
    for (const g of gpus) {
        container.textContent = `${(g.memory_used / 1024).toFixed(1)} / ${(g.memory_total / 1024).toFixed(1)} GB`;
        tempEl.textContent = `${g.temperature}°C`;

        html += `
        <div class="gpu-card">
            <div class="gpu-name">${esc(g.name)}</div>
            <div class="gpu-stats">
                <div class="gpu-stat"><span class="stat-label">顯存使用</span><span class="stat-value">${(g.memory_used / 1024).toFixed(1)} / ${(g.memory_total / 1024).toFixed(1)} GB</span></div>
                <div class="gpu-stat"><span class="stat-label">GPU 使用率</span><span class="stat-value">${g.utilization}%</span></div>
                <div class="gpu-stat"><span class="stat-label">溫度</span><span class="stat-value">${g.temperature}°C</span></div>
                <div class="gpu-stat"><span class="stat-label">功耗</span><span class="stat-value">${g.power_draw.toFixed(1)} / ${g.power_limit.toFixed(0)} W</span></div>
                <div class="gpu-stat"><span class="stat-label">風扇</span><span class="stat-value">${g.fan_speed}%</span></div>
            </div>
        </div>`;
    }
    gpuInfo.innerHTML = html;
}

let logEventSource = null;
let logLines = [];

function toggleLogStream() {
    const btn = document.querySelector('button[onclick="toggleLogStream()"]');
    const badge = document.getElementById('log-status');
    
    if (logEventSource) {
        // 停止串流
        logEventSource.close();
        logEventSource = null;
        badge.textContent = '串流關閉';
        badge.className = 'badge badge-stopped';
        btn.textContent = '開始串流';
    } else {
        // 開始串流
        logLines = [];
        document.getElementById('log-output').innerHTML = '<div class="log-line">正在連接串流...</div>';
        
        logEventSource = new EventSource('/api/logs/stream');
        logEventSource.onmessage = function(e) {
            logLines.push(e.data);
            // 只保留最近 500 行
            if (logLines.length > 500) logLines = logLines.slice(-500);
            
            const container = document.getElementById('log-output');
            container.innerHTML = logLines.map(l => `<div class="log-line">${esc(l)}</div>`).join('');
            container.scrollTop = container.scrollHeight;
        };
        logEventSource.onerror = function() {
            badge.textContent = '串流斷線';
            badge.className = 'badge badge-stopped';
        };
        
        badge.textContent = '串流中';
        badge.className = 'badge badge-running';
        btn.textContent = '停止串流';
    }
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
    const resp = await fetchJSON(API.start, { method: 'POST' });
    if (resp) {
        alert(`vLLM ${resp.status}: ${resp.message || '啟動成功'}`);
        refreshAll();
    }
}

async function stopVLLM() {
    const resp = await fetchJSON(API.stop, { method: 'POST' });
    if (resp) {
        alert(`vLLM ${resp.status}`);
        refreshAll();
    }
}

async function restartVLLM() {
    const resp = await fetchJSON(API.restart, { method: 'POST' });
    if (resp) {
        alert(`vLLM ${resp.status}: ${resp.message || '重啟成功'}`);
        refreshAll();
    }
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
    await fetchJSON(`${API.models}/enable/${encodeURIComponent(path)}`, { method: 'POST' });
    refreshModels();
}

async function removeModel(path) {
    if (!confirm(`確定要移除模型: ${path} ?`)) return;
    await fetchJSON(`${API.models}?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
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
    const defaults = {
        port: parseInt(document.getElementById('param-port').value) || 8001,
        host: document.getElementById('param-host').value || '0.0.0.0',
        dtype: document.getElementById('param-dtype').value || 'float16',
        max_model_len: parseInt(document.getElementById('param-max-model-len').value) || 128000,
        gpu_memory_utilization: parseFloat(document.getElementById('param-gpu-memory-utilization').value) || 0.9,
        max_num_seqs: parseInt(document.getElementById('param-max-num-seqs').value) || 4,
        kv_cache_dtype: document.getElementById('param-kv-cache-dtype').value || 'fp8',
        tool_call_parser: document.getElementById('param-tool-call-parser').value || 'qwen3_xml',
        enable_prefix_caching: document.getElementById('param-enable-prefix-caching').checked,
        enable_auto_tool_choice: document.getElementById('param-enable-auto-tool-choice').checked,
        trust_remote_code: document.getElementById('param-trust-remote-code').checked,
    };
    const resp = await fetchJSON(API.config, {
        method: 'POST',
        body: JSON.stringify({ defaults })
    });
    if (resp && resp.status === 'saved') {
        alert('參數已儲存');
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
