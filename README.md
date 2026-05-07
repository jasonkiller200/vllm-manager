# vLLM Manager

輕量級 vLLM 服務管理面板，透過 Web 介面管理 vLLM 進程、模型、參數與 GPU 狀態。

## 功能

- **vLLM 進程管理**: 啟動 / 停止 / 重啟 vLLM 服務
- **模型管理**: 自動掃描模型目錄、從 Hugging Face clone 模型、手動新增模型
- **即時日誌**: SSE 即時串流 vLLM 日誌
- **GPU 監控**: 即時顯示 GPU 顯存使用與溫度
- **參數調整**: 即時修改 vLLM 啟動參數 (tensor parallel, max models, quantization 等)
- **開機自啟**: systemd 系統服務，開機自動啟動

## 系統需求

- Ubuntu 22.04/24.04 (Linux)
- NVIDIA GPU (建議 RTX 4090/5090 或更高)
- Python 3.12+
- CUDA 12.x + cuDNN
- Git + Git LFS
- NVIDIA Container Toolkit (如需 Docker)

---

## 新主機完整部署指南

以下是在全新 AI 主機上部署 vLLM + 管理面板的完整流程。

### 前置作業

```bash
# 更新系統
sudo apt update && sudo apt upgrade -y

# 安裝基礎工具
sudo apt install -y git curl wget software-properties-common python3 python3-venv python3-pip

# 安裝 Git LFS (Hugging Face 模型下載必備)
sudo apt install -y git-lfs
git lfs install
```

### 1. 安裝 NVIDIA 驅動與 CUDA

```bash
# 檢查 NVIDIA 驅動是否已安裝
nvidia-smi

# 如果沒有，安裝驱动 (根據你的 GPU 型號)
sudo apt install -y nvidia-driver-550  # 或最新版本

# 重啟系統
sudo reboot

# 驗證驅動安裝成功
nvidia-smi
```

### 2. 安裝 vLLM

```bash
# 建立 vLLM 專用虛擬環境
cd ~
python3 -m venv vllm
source vllm/bin/activate

# 升級 pip
pip install --upgrade pip

# 安裝 vLLM (含 CUDA 支援)
pip install vllm

# 驗證安裝
python -c "import vllm; print(vllm.__version__)"

# 測試 vLLM 是否正常運行
python -m vllm.entrypoints.api_server --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --max-model-len 2048 &
# 測試 API
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
# 停止測試伺服器
kill %1

# 退出 venv
deactivate
```

### 3. 設定模型目錄

```bash
# 建立模型存放目錄
mkdir -p ~/models

# 下載測試模型 (範例：小模型測試)
cd ~/models
git clone https://huggingface.co/Qwen/Qwen2.5-7B-Instruct --depth 1
cd Qwen2.5-7B-Instruct && git lfs pull

# 驗證模型檔案
du -sh ~/models/Qwen2.5-7B-Instruct
```

### 4. 部署管理面板

```bash
# 複製專案到新主機 (或直接 clone)
cd ~/app
git clone <your-repo-url> vllm   # 或將專案資料拷貝過來

# 建立虛擬環境
cd ~/app/vllm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
```

### 5. 設定 systemd 服務

```bash
# 複製服務檔
sudo cp vllm-manager.service /etc/systemd/system/

# 重新載入 systemd
sudo systemctl daemon-reload

# 啟用開機自動啟動
sudo systemctl enable vllm-manager

# 啟動服務
sudo systemctl start vllm-manager

# 檢查狀態
sudo systemctl status vllm-manager --no-pager
```

### 6. 驗證部署

```bash
# 檢查管理面板是否正常
curl http://localhost:5555/api/status

# 檢查 vLLM 狀態
curl http://localhost:5555/api/models

# 在瀏覽器開啟管理面板
# http://<你的主機IP>:5555
```

### 7. 防火牆設定 (如需從外部存取)

```bash
# 開放 5555 端口
sudo ufw allow 5555/tcp

# 重新整理防火牆
sudo ufw reload
```

---

### 2. 設定 systemd 服務

```bash
sudo cp vllm-manager.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vllm-manager
sudo systemctl start vllm-manager
```

### 3. 手動啟動 (開發模式)

```bash
cd ~/app/vllm
source venv/bin/activate
python -m flask --app app.py run --host=0.0.0.0 --port=5555 --reload
```

## 專案結構

```
~/app/vllm/
├── app.py                  # Flask 主程式 (API 端點)
├── config.yaml             # vLLM 參數與模型設定
├── requirements.txt        # Python 依賴
├── venv/                   # 虛擬環境
├── models/
│   └── vllm_manager.py     # vLLM 進程管理核心
├── templates/
│   └── index.html          # 前端頁面
├── static/
│   ├── css/
│   │   └── style.css       # 樣式
│   └── js/
│       └── app.js          # 前端邏輯
├── logs/                   # vLLM 日誌
└── vllm-manager.service    # systemd 服務檔
```

## 設定檔 (config.yaml)

```yaml
vllm:
  python: /home/jason/vllm/venv/bin/python
  base_dir: /home/jason/models
  log_dir: /home/jason/app/vllm/logs
  tensor_parallel_size: 1
  max_model_len: 4096
  max_running_models: 1
  quantization: awq
  gpu_memory_utilization: 0.95
  enable_prefix_caching: true
  disable_log_requests: true

models:
  - name: Qwen3.6-27B-AWQ-INT4
    path: /home/jason/models/Qwen3.6-27B-AWQ-INT4
    enabled: true
    # vLLM 啟動參數 (覆蓋預設)
    params:
      tensor_parallel_size: 1
      max_model_len: 4096

models_dir: /home/jason/models
log_max_lines: 2000
```

## API 端點

| 方法 | 端點 | 說明 |
|------|------|------|
| GET | `/api/status` | vLLM 狀態 (running, pid, uptime, GPU) |
| POST | `/api/start` | 啟動 vLLM |
| POST | `/api/stop` | 停止 vLLM |
| POST | `/api/restart` | 重啟 vLLM |
| GET | `/api/logs` | 取得日誌 (支援 line_offset/limit) |
| GET | `/api/logs/stream` | SSE 即時日誌串流 |
| GET | `/api/models` | 取得模型清單 |
| POST | `/api/models/scan` | 掃描模型目錄 |
| POST | `/api/models/clone` | 從 Hugging Face clone 模型 |
| GET | `/api/clone-status` | clone 進度與狀態 |
| POST | `/api/config` | 更新 vLLM 參數 |

### 範例

```bash
# 檢查狀態
curl http://localhost:5555/api/status

# 啟動 vLLM
curl -X POST http://localhost:5555/api/start

# 從 Hugging Face clone 模型
curl -X POST http://localhost:5555/api/models/clone \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "Qwen/Qwen3-Coder-30B-A3B-Instruct-AWQ-4bit"}'

# 檢查 clone 進度
curl http://localhost:5555/api/clone-status
```

## 常用管理指令

```bash
# 查看服務狀態
sudo systemctl status vllm-manager

# 重新啟動服務
sudo systemctl restart vllm-manager

# 查看服務日誌
sudo journalctl -u vllm-manager -f

# 停止服務
sudo systemctl stop vllm-manager

# 開機不自動啟動
sudo systemctl disable vllm-manager
```

## 技術架構

- **後端**: Flask (Python)
- **前端**: 原生 HTML + CSS + Vanilla JS (無框架)
- **進程管理**: Python `subprocess.Popen`
- **即時日誌**: Server-Sent Events (SSE)
- **GPU 監控**: `nvidia-smi` CLI
- **部署**: systemd 系統服務

## 注意事項

1. 管理層純 CPU 運行，vLLM 由獨立進程管理
2. clone 模型時會自動執行 `git lfs pull` 下載實際權重
3. vLLM 日誌輸出到 `/dev/pts/0` 而非檔案，日誌功能從 `/proc/[pid]/fd/1` 讀取
4. 修改 `config.yaml` 後需重啟服務才會生效

## 作者

vLLM Manager - 輕量級 vLLM 管理面板
