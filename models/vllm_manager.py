"""
Model 層 - vLLM 進程與設定管理

負責:
- vLLM 子進程的生命週期 (start/stop/restart)
- 讀取/寫入 config.yaml
- 捕捉 vLLM stdout/stderr 日誌
- GPU 狀態查詢 (nvidia-smi)
"""

import os
import subprocess
import time
import threading
import re
import yaml
from pathlib import Path

# 專案根目錄
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
LOG_DIR = BASE_DIR / "logs"


class VLLMManager:
    """管理 vLLM 子進程與設定"""

    _instance = None

    def __init__(self):
        self.config = self._load_config()
        self.log_dir = Path(self.config.get("log_dir", str(LOG_DIR)))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_max_lines = self.config.get("log_max_lines", 2000)
        self._process = None  # subprocess.Popen instance
        self._detect_running()
        
        # Auto-start last running model if enabled
        self._auto_start_model()
        
        # Clone state
        self._clone_proc = None
        self._clone_result = None
        self._clone_progress = ""
        self._clone_percentage = 0
        self._clone_model_name = ""
        self._clone_model_dir = None
        self._clone_thread = None

    def _auto_start_model(self):
        """If auto_start is enabled and vLLM is not running, start the last enabled model."""
        if not self.config.get("auto_start", True):
            return
        if self.is_running:
            return  # Already running, skip
        # Try the last explicitly marked running model, or fall back to enabled model
        last_model = self.config.get("last_running_model")
        model_path = last_model or self._get_enabled_model()
        if not model_path:
            return
        # Start in background thread to not block web UI
        def _do_auto_start():
            import time
            time.sleep(3)  # Wait for web UI to be ready
            try:
                result = self.start(model_path=model_path)
                if result.get("status") == "started":
                    print(f"[AUTO-START] vLLM started: {model_path} (PID {result.get('pid')})")
                else:
                    print(f"[AUTO-START] Failed to start vLLM: {result.get('message', 'unknown')}")
            except Exception as e:
                print(f"[AUTO-START] Error: {e}")
        t = threading.Thread(target=_do_auto_start, daemon=True)
        t.start()

    def _load_config(self):
        """載入 YAML 設定"""
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _save_config(self):
        """儲存設定到 YAML"""
        with open(CONFIG_PATH, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def _detect_running(self):
        """啟動時檢測是否已有 vLLM 在跑"""
        try:
            pids = self._get_vllm_pids()
            if pids:
                # 找 PID 最小的（最早啟動的，排除 manager 自身的 pgrep）
                self._detected_pid = min(pids)
        except Exception:
            pass

    def _get_vllm_pids(self):
        """取得所有 vLLM API server 的 PID 清單"""
        result = subprocess.run(
            ["pgrep", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            pids = [int(p) for p in result.stdout.strip().split("\n") if p.isdigit()]
            # 排除 Flask manager 自身的 pgrep 進程
            current_pid = os.getpid()
            pids = [p for p in pids if p != current_pid]
            return pids
        return []

    @property
    def is_running(self):
        """vLLM 是否正在運行"""
        return len(self._get_vllm_pids()) > 0

    @property
    def pid(self):
        """取得 vLLM 主 PID"""
        pids = self._get_vllm_pids()
        if pids:
            return min(pids)  # 最早的進程是主進程
        return None

    def _get_model_params(self, model_path):
        """取得模型的獨立參數 (從 model.entry.params)"""
        for entry in self.config.get("models", []):
            if entry.get("path") == model_path:
                return entry.get("params", {})
        return {}

    def _get_model_name(self, model_path):
        """從 model entry 取得模型名稱"""
        for entry in self.config.get("models", []):
            if entry.get("path") == model_path:
                return entry.get("name", model_path)
        return model_path

    def get_model_params(self):
        """取得目前啟用模型的合併參數 (defaults + model.params)"""
        model = self._get_enabled_model()
        if not model:
            return {"status": "error", "message": "沒有啟用的模型", "params": {}}
        defaults = self.config.get("defaults", {})
        model_params = self._get_model_params(model)
        merged = {**defaults, **model_params}
        return {"status": "ok", "model": model, "name": self._get_model_name(model), "params": merged}

    def update_model_params(self, params):
        """儲存參數到目前啟用模型的 params"""
        model = self._get_enabled_model()
        if not model:
            return {"status": "error", "message": "沒有啟用的模型"}
        for entry in self.config.get("models", []):
            if entry.get("path") == model:
                if "params" not in entry:
                    entry["params"] = {}
                entry["params"].update(params)
                self._save_config()
                return {"status": "saved", "model": model}
        return {"status": "error", "message": "找不到模型"}

    def start(self, model_path=None, params=None):
        """啟動 vLLM 服務

        Args:
            model_path: 模型路徑，None 則用 config 中 enabled 的模型
            params: 參數覆寫 dict (runtime override)
        """
        if self.is_running:
            if model_path:
                self.stop()
                time.sleep(3)
            else:
                return {"status": "already_running", "pid": self.pid}

        if model_path:
            model = model_path
        else:
            model = self._get_enabled_model()
            if not model:
                return {"status": "error", "message": "沒有啟用的模型"}

        defaults = self.config.get("defaults", {})
        model_params = self._get_model_params(model)
        settings = {**defaults, **model_params, **(params or {})}

        venv_python = Path(self.config["venv_path"]) / "bin" / "python3"

        cmd = [
            str(venv_python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", model,
            "--host", str(settings.get("host", "0.0.0.0")),
            "--port", str(settings.get("port", 8001)),
            "--dtype", str(settings.get("dtype", "float16")),
            "--max-model-len", str(settings.get("max_model_len", 128000)),
            "--kv-cache-dtype", str(settings.get("kv_cache_dtype", "fp8")),
            "--gpu-memory-utilization", str(settings.get("gpu_memory_utilization", 0.9)),
            "--max-num-seqs", str(settings.get("max_num_seqs", 4)),
            "--max-num-batched-tokens", str(settings.get("max_num_batched_tokens", 4096)),
            "--enable-prefix-caching",
            "--enable-auto-tool-choice",
            "--tool-call-parser", str(settings.get("tool_call_parser", "qwen3_xml")),
            "--trust-remote-code",
            "--no-enable-log-requests",
            "--uvicorn-log-level", str(settings.get("uvicorn_log_level", "info")),
        ]

        spec_enabled = settings.get("speculative_enabled", False)
        spec_method = settings.get("speculative_method", "mtp")
        spec_num_tokens = settings.get("speculative_num_tokens", 1)

        if spec_enabled:
            import json
            spec_config = json.dumps({
                "method": spec_method,
                "num_speculative_tokens": spec_num_tokens
            })
            cmd.extend(["--speculative-config", spec_config])

        log_file = self.log_dir / f"vllm_{int(time.time())}.log"

        try:
            with open(log_file, "w") as f:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
            self.config["last_running_model"] = model
            self._save_config()
            
            return {
                "status": "starting",
                "pid": self._process.pid,
                "port": settings.get("port", 8001),
                "model": model,
                "log_file": str(log_file),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def stop(self):
        """停止 vLLM 服務"""
        if not self.is_running:
            return {"status": "not_running"}

        pids = []
        try:
            result = subprocess.run(
                ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
                capture_output=True, text=True
            )
            pids = [self.pid] if self.pid else []
        except Exception as e:
            return {"status": "error", "message": str(e)}

        self._process = None
        # Clear last running model on stop
        if "last_running_model" in self.config:
            del self.config["last_running_model"]
            self._save_config()
        return {"status": "stopped", "pids": pids}

    def restart(self, model_path=None, params=None):
        """重啟 vLLM 服務"""
        self.stop()
        time.sleep(3)  # 等 GPU 釋放
        return self.start(model_path, params)

    def get_status(self):
        """取得完整狀態"""
        running = self.is_running
        result = {
            "running": running,
            "pid": self.pid if running else None,
            "uptime": None,
        }
        if running and self.pid:
            try:
                with open(f"/proc/{self.pid}/stat", "r") as f:
                    stat = f.read().split()
                    # starttime is field 22 (0-indexed 21)
                    starttime = int(stat[21])
                    with open("/proc/uptime", "r") as f:
                        uptime = float(f.read().split()[0])
                    clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                    elapsed = uptime - (starttime / clk_tck)
                    result["uptime"] = int(elapsed)
            except Exception:
                result["uptime"] = -1

        return result

    def get_config(self):
        """取得目前設定 (不暴露 venv_path)"""
        safe = {k: v for k, v in self.config.items()
                if k not in ("venv_path",)}
        return safe

    def update_config(self, updates):
        """更新設定"""
        if "defaults" in updates:
            # 支援個別欄位或整個 defaults 覆寫
            if isinstance(updates["defaults"], dict):
                self.config["defaults"].update(updates["defaults"])
            else:
                self.config["defaults"] = updates["defaults"]
        if "models" in updates:
            self.config["models"] = updates["models"]
        if "log_max_lines" in updates:
            self.config["log_max_lines"] = updates["log_max_lines"]
        self._save_config()
        return {"status": "saved"}

    def get_log_file(self):
        """取得目前最新的日誌檔案路徑"""
        log_files = sorted(self.log_dir.glob("vllm_*.log"),
                           key=os.path.getmtime, reverse=True)
        return log_files[0] if log_files else None

    def stream_logs(self):
        """SSE 串流日誌 - 持續 yield 新增的日誌行"""
        log_file = None
        pos = 0
        newline = "\n"
        data_prefix = "data: "
        idle_count = 0
        while True:
            current_log = self.get_log_file()
            if current_log and current_log != log_file:
                log_file = current_log
                pos = 0
            
            if log_file:
                try:
                    with open(log_file, "r") as f:
                        f.seek(pos)
                        new_lines = f.readlines()
                        if new_lines:
                            pos = f.tell()
                            idle_count = 0
                            for line in new_lines:
                                yield f"{data_prefix}{line.rstrip(newline)}{newline}{newline}"
                        else:
                            idle_count += 1
                except FileNotFoundError:
                    log_file = None
                    idle_count = 0
            else:
                idle_count += 1
                yield f"{data_prefix}[等待 vLLM 啟動...]@{newline}{newline}"
            
            if idle_count >= 20:
                idle_count = 0
                yield f"{data_prefix}[心跳]@{newline}{newline}"
            
            time.sleep(0.5)

    def get_logs(self, lines=100):
        """取得最新 vLLM 日誌
    
        優先從 manager 管理的日誌檔讀取，如果沒有則嘗試從
        /proc/{pid}/fd/1 讀取 vLLM 目前的輸出。
        """
        # 先檢查是否有 manager 管理的日誌檔
        log_files = sorted(self.log_dir.glob("vllm_*.log"),
                           key=os.path.getmtime, reverse=True)
        if log_files:
            with open(log_files[0], "r") as f:
                all_lines = f.readlines()
            return [l.rstrip("\n") for l in all_lines[-lines:]]

        # 沒有日誌檔，嘗試從 vLLM 進程的 stdout 讀取
        pid = self.pid
        if pid:
            proc_stdout = f"/proc/{pid}/fd/1"
            if os.path.exists(proc_stdout):
                try:
                    # 使用 tail 取得最後 n 行
                    result = subprocess.run(
                        ["tail", "-n", str(lines), proc_stdout],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.stdout:
                        return result.stdout.rstrip("\n").split("\n")
                except Exception:
                    pass

        return ["[vLLM 日誌空白 - 可能 vLLM 非由 manager 啟動，stdout 未導向檔案]"]

    def get_gpu_info(self):
        """取得 GPU 狀態"""
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,memory.used,memory.free,"
                 "utilization.gpu,temperature.gpu,power.draw,power.limit,"
                 "fan.speed",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )
            gpus = []
            for line in result.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 8:
                    gpus.append({
                        "name": parts[0],
                        "memory_total": int(parts[1]),
                        "memory_used": int(parts[2]),
                        "memory_free": int(parts[3]),
                        "utilization": int(parts[4]),
                        "temperature": int(parts[5]),
                        "power_draw": float(parts[6]),
                        "power_limit": float(parts[7]),
                        "fan_speed": int(parts[8]) if len(parts) > 8 else 0,
                    })
            return gpus
        except Exception as e:
            return [{"error": str(e)}]

    def get_cpu_info(self):
        """取得 CPU/RAM 狀態"""
        try:
            result = subprocess.run(
                ["ps", "-C", "python3", "-o", "%cpu,%mem,cmd", "--no-headers"],
                capture_output=True, text=True, timeout=5
            )
            total_cpu = 0.0
            total_mem = 0.0
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            try:
                                total_cpu += float(parts[0])
                                total_mem += float(parts[1])
                            except ValueError:
                                pass

            result = subprocess.run(
                ["cat", "/proc/meminfo"],
                capture_output=True, text=True, timeout=5
            )
            mem_total = mem_free = mem_available = swap_total = swap_used = 0
            for line in result.stdout.strip().split("\n"):
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) / 1024
                elif line.startswith("MemFree:"):
                    mem_free = int(line.split()[1]) / 1024
                elif line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) / 1024
                elif line.startswith("SwapTotal:"):
                    swap_total = int(line.split()[1]) / 1024
                elif line.startswith("SwapFree:"):
                    swap_free = int(line.split()[1]) / 1024
                    swap_used = swap_total - swap_free

            cpu_count = os.cpu_count() or 1
            load_avg = []
            try:
                with open("/proc/loadavg", "r") as f:
                    la = f.read().split()
                    load_avg = [float(la[i]) for i in range(3)]
            except Exception:
                pass

            temps = {}
            for hwmon in Path("/sys/class/hwmon").iterdir() if Path("/sys/class/hwmon").exists() else []:
                try:
                    name_file = hwmon / "name"
                    if name_file.exists():
                        name = name_file.read_text().strip()
                        if name in ("coretemp", "cpu_thermal", "k10temp", "nvme"):
                            for temp_file in sorted((hwmon / "device" / "temp*_input" if (hwmon / "device").exists() else hwmon.glob("temp*_input"))):
                                label_file = temp_file.parent / (temp_file.name.replace("_input", "_label"))
                                label = label_file.read_text().strip() if label_file.exists() else f"temp{len(temps)}"
                                temp = int(temp_file.read_text().strip()) / 1000
                                temps[label] = round(temp, 1)
                            if temps:
                                break
                except Exception:
                    pass

            return {
                "cpu_count": cpu_count,
                "cpu_percent": round(total_cpu, 1),
                "mem_percent": round(total_mem, 1),
                "mem_total_mb": round(mem_total),
                "mem_used_mb": round(mem_total - mem_free),
                "mem_free_mb": round(mem_free),
                "mem_available_mb": round(mem_available),
                "swap_total_mb": round(swap_total),
                "swap_used_mb": round(swap_used),
                "load_avg_1m": round(load_avg[0], 2) if len(load_avg) > 0 else 0,
                "load_avg_5m": round(load_avg[1], 2) if len(load_avg) > 1 else 0,
                "load_avg_15m": round(load_avg[2], 2) if len(load_avg) > 2 else 0,
                "temperatures": temps,
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_enabled_model(self):
        """取得 config 中 enabled=true 的模型路徑"""
        for m in self.config.get("models", []):
            if m.get("enabled"):
                return m["path"]
        return None

    def scan_models(self):
        """掃描 models_dir，自動加入新發現的模型"""
        models_dir = Path(self.config.get("models_dir", "/home/jason/models"))
        if not models_dir.exists():
            return {"status": "error", "message": f"目錄不存在: {models_dir}"}

        existing_paths = {m["path"] for m in self.config.get("models", [])}
        scanned = []
        for d in sorted(models_dir.iterdir()):
            if d.is_dir():
                path = str(d)
                if path not in existing_paths:
                    self.config["models"].append({
                        "name": d.name,
                        "path": path,
                        "enabled": False
                    })
                    scanned.append(d.name)
                else:
                    scanned.append(d.name)

        self._save_config()
        return {"status": "scanned", "new": len(scanned) - len(existing_paths), "total": len(scanned)}

    def clone_model(self, repo_url):
        """從 Hugging Face clone 模型到 models_dir (背景執行)
        
        Args:
            repo_url: Hugging Face 模型路徑 (如 Qwen/Qwen3.6-27B-AWQ-INT4) 或 git clone URL
        """
        if self._clone_thread and self._clone_thread.is_alive():
            return {"status": "error", "message": "已有 clone 任務正在進行"}
        
        models_dir = Path(self.config.get("models_dir", "/home/jason/models"))
        models_dir.mkdir(parents=True, exist_ok=True)

        # 解析 repo ID 和 repo 名稱
        if "https://" in repo_url or "git@" in repo_url:
            repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
            # 從 URL 提取 repo_id (如 cyankiwi/Qwen3.5-9B-AWQ-4bit)
            repo_id = "/".join(repo_url.rstrip("/").split("/")[-2:]) if len(repo_url.rstrip("/").split("/")) >= 3 else repo_name
        else:
            parts = repo_url.strip("/").split("/")
            repo_name = parts[-1]
            repo_id = repo_url.strip("/")

        model_dir = models_dir / repo_name
        if model_dir.exists():
            return {"status": "error", "message": f"Model directory already exists: {model_dir}"}
        
        # Reset state
        self._clone_result = None
        self._clone_progress = f"開始 clone: {repo_id}"
        self._clone_percentage = 0
        self._clone_model_name = repo_name
        self._clone_model_dir = model_dir
        
        # Run in background thread
        self._clone_thread = threading.Thread(target=self._clone_worker, args=(repo_id, model_dir, repo_name), daemon=True)
        self._clone_thread.start()
        
        return {"status": "started", "name": repo_name}

    def _clone_worker(self, repo_id, model_dir, repo_name):
        """Background worker: use huggingface_hub.snapshot_download (handles LFS automatically)"""
        env = dict(os.environ)
        try:
            import huggingface_hub
            
            self._clone_progress = f"⏳ 下載中: {repo_id}"
            self._clone_percentage = 5
            
            # snapshot_download handles LFS automatically - downloads real files, not pointers
            # No progress callback API in this version, so track by monitoring directory size
            start_time = time.time()
            last_size = 0
            
            # Start download in a thread so we can monitor progress concurrently
            download_done = threading.Event()
            download_error = [None]
            
            def do_download():
                try:
                    huggingface_hub.snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(model_dir),
                        local_dir_use_symlinks=False,
                        token=None,
                    )
                except Exception as e:
                    download_error[0] = e
                finally:
                    download_done.set()
            
            download_thread = threading.Thread(target=do_download, daemon=True)
            download_thread.start()
            
            # Monitor progress by directory size
            while not download_done.is_set():
                current_size = self._dir_size(model_dir)
                if current_size > last_size and last_size > 0:
                    speed = current_size / max(time.time() - start_time, 1)
                    mb = current_size / (1024 * 1024)
                    speed_mbps = speed / (1024 * 1024)
                    self._clone_progress = f"⏳ 已下載 {mb:.1f} MB ({speed_mbps:.1f} MB/s)"
                    # Incremental progress (cap at 95% until done)
                    self._clone_percentage = min(95, self._clone_percentage + 1)
                last_size = current_size
                time.sleep(1)
            
            if download_error[0]:
                raise download_error[0]
            
            self._clone_percentage = 100
            
            # Add model to config
            self.config["models"].append({
                "name": repo_name,
                "path": str(model_dir),
                "enabled": False
            })
            self._save_config()
            
            final_size = self._dir_size(model_dir)
            size_gb = final_size / (1024 * 1024 * 1024)
            self._clone_progress = f"✅ Clone 完成: {repo_name} ({size_gb:.2f} GB)"
            self._clone_result = {
                "status": "cloned",
                "name": repo_name,
                "path": str(model_dir)
            }
            
        except Exception as e:
            self._clone_progress = f"❌ Clone error: {str(e)}"
            self._clone_result = {"status": "error", "message": str(e)}

    def _dir_size(self, path):
        """Get total size of directory in bytes"""
        total = 0
        try:
            for p in path.rglob("*"):
                if p.is_file():
                    total += p.stat().st_size
        except (OSError, PermissionError):
            pass
        return total

    def get_models(self):
        """取得模型清單"""
        return self.config.get("models", [])

    def add_model(self, name, path):
        """新增模型"""
        models = self.config.get("models", [])
        # 檢查是否已存在
        for m in models:
            if m["path"] == path:
                return {"status": "exists", "model": m}
        models.append({"name": name, "path": path, "enabled": False})
        self.config["models"] = models
        self._save_config()
        return {"status": "added"}

    def remove_model(self, path):
        """移除模型"""
        models = self.config["models"] = [
            m for m in self.config.get("models", []) if m["path"] != path
        ]
        self._save_config()
        return {"status": "removed"}

    def set_enabled_model(self, path):
        """設定預設啟用的模型"""
        for m in self.config.get("models", []):
            m["enabled"] = (m["path"] == path)
        self._save_config()
        return {"status": "enabled", "path": path}
