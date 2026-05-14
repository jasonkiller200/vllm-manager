#!/usr/bin/env python3
"""
Qwen3.5 MTP 性能測試腳本
比較開啟/關閉 Speculative Decoding (MTP) 的性能差異

測試指標:
1. Generation Throughput (tokens/s) - 每秒產生的 token 數量
2. Acceptance Rate (%) - MTP 預猜 token 被接受的比率 (理想 80%~95%)
3. Mean Acceptance Length - 每次 MTP 步驟平均接受的 token 數量

使用方式:
    python3 benchmark_mtp.py
"""

import requests
import json
import time
import statistics
import subprocess
import copy
import yaml
from datetime import datetime
from pathlib import Path

# API 配置
MANAGER_URL = "http://localhost:5555"  # vllm-manager Web UI
VLLM_URL = "http://localhost:8001"    # vLLM API
CONFIG_PATH = Path("config.yaml")
BENCHMARK_DIR = Path("logs") / "benchmark"
VLLM_BENCHMARK_LOG = BENCHMARK_DIR / "vllm_benchmark.log"

# 測試提示詞 (不同長度以測試不同場景)
PROMPTS = [
    # 短提示詞
    "請用簡體中文解釋什麼是量子計算機，並說明其潛在應用。",
    
    # 中等長度提示詞
    """請寫一個 Python 函數，實現快速排序算法，並包含以下功能：
    1. 支持自定義比較函數
    2. 支持原地排序和返回新列表兩種模式
    3. 包含詳細的註釋說明
    4. 提供使用示例""",
    
    # 長提示詞
    """請詳細解釋 Transformer 模型的架構原理，包括：
    1. Self-Attention 機制的工作原理和數學公式
    2. Position Encoding 的作用和實現方式
    3. Multi-Head Attention 的好處
    4. Feed-Forward Network 的設計
    5. Layer Normalization 的位置和作用
    6. Residual Connection 的重要性
    
    請用技術性的語言，並包含必要的公式說明。"""
]

# 生成參數
GENERATE_PARAMS = {
    "max_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1,
}

RUNS_PER_PROMPT = 3
MTP_METHOD_ALIASES = {"qwen3_next_mtp", "qwen3_5_mtp"}


def load_config():
    """載入配置文件"""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def ensure_benchmark_dir():
    """確保 benchmark 輸出目錄存在。"""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)


def normalize_speculative_method(method):
    """Map deprecated model-specific MTP aliases to the generic vLLM method."""
    if not method:
        return "mtp"
    return "mtp" if method in MTP_METHOD_ALIASES else method


def save_config(config):
    """保存配置文件"""
    with open(CONFIG_PATH, 'w') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def get_enabled_model():
    """獲取當前啟用的模型"""
    config = load_config()
    for model in config.get("models", []):
        if model.get("enabled", False):
            return model
    return None


def snapshot_enabled_model_params():
    """保存目前啟用模型的 params，方便 benchmark 後還原。"""
    config = load_config()
    for model in config.get("models", []):
        if model.get("enabled", False):
            return {
                "had_params": "params" in model,
                "params": copy.deepcopy(model.get("params", {})),
            }
    return {"had_params": False, "params": {}}


def restore_enabled_model_params(snapshot):
    """還原 benchmark 前的模型參數。"""
    config = load_config()
    for model in config.get("models", []):
        if model.get("enabled", False):
            if snapshot.get("had_params", False):
                model["params"] = copy.deepcopy(snapshot.get("params", {}))
            else:
                model.pop("params", None)
            save_config(config)
            print("已還原 benchmark 前的模型參數")
            return True
    return False


def update_benchmark_config(enable_mtp, num_tokens, spec_method, prefix_caching):
    """更新 benchmark 所需的 MTP / prefix caching 配置並重啟 vLLM。"""
    config = load_config()

    for model in config.get("models", []):
        if model.get("enabled", False):
            if "params" not in model:
                model["params"] = {}
            model["params"]["enable_prefix_caching"] = prefix_caching
            model["params"]["speculative_enabled"] = enable_mtp
            model["params"]["speculative_method"] = spec_method
            model["params"]["speculative_num_tokens"] = num_tokens

    save_config(config)
    print(
        "配置已更新："
        f"speculative_enabled={enable_mtp}, "
        f"speculative_method={spec_method}, "
        f"speculative_num_tokens={num_tokens}, "
        f"enable_prefix_caching={prefix_caching}"
    )

    print("正在重啟 vLLM 服務...")
    return restart_vllm()


def restart_vllm():
    """通過 pkill 重啟 vLLM"""
    try:
        # 停止 vLLM
        print("正在停止 vLLM...")
        result = subprocess.run(
            ["pkill", "-f", "vllm.entrypoints.openai.api_server"],
            capture_output=True, text=True
        )
        time.sleep(5)  # 等待 GPU 釋放
        
        # 從 config 獲取模型路徑並啟動
        config = load_config()
        model_path = None
        for model in config.get("models", []):
            if model.get("enabled", False):
                model_path = model.get("path")
                break
        
        if not model_path:
            print("✗ 找不到啟用的模型")
            return False
        
        print(f"正在啟動模型：{model_path}")
        
        # 構建啟動命令
        model = next((m for m in config.get("models", []) if m.get("path") == model_path), None)
        defaults = config.get("defaults", {})
        model_params = model.get("params", {}) if model else {}
        settings = {**defaults, **model_params}
        
        venv_python = Path(config.get("venv_path", "/home/jason/vllm/venv")) / "bin" / "python3"
        
        cmd = [
            str(venv_python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--host", str(settings.get("host", "0.0.0.0")),
            "--port", str(settings.get("port", 8001)),
            "--dtype", str(settings.get("dtype", "float16")),
            "--max-model-len", str(settings.get("max_model_len", 128000)),
            "--kv-cache-dtype", str(settings.get("kv_cache_dtype", "fp8")),
            "--gpu-memory-utilization", str(settings.get("gpu_memory_utilization", 0.9)),
            "--max-num-seqs", str(settings.get("max_num_seqs", 4)),
            "--max-num-batched-tokens", str(settings.get("max_num_batched_tokens", 4096)),
            "--enable-auto-tool-choice",
            "--tool-call-parser", str(settings.get("tool_call_parser", "qwen3_xml")),
            "--trust-remote-code",
            "--no-enable-log-requests",
            "--uvicorn-log-level", str(settings.get("uvicorn_log_level", "info")),
        ]

        if settings.get("enable_prefix_caching", False):
            cmd.append("--enable-prefix-caching")
        
        spec_enabled = settings.get("speculative_enabled", False)
        spec_method = normalize_speculative_method(settings.get("speculative_method", "mtp"))
        spec_num_tokens = settings.get("speculative_num_tokens", 1)
        
        if spec_enabled:
            import json
            spec_config = json.dumps({
                "method": spec_method,
                "num_speculative_tokens": spec_num_tokens
            })
            cmd.extend(["--speculative-config", spec_config])
        
        # 後台啟動
        ensure_benchmark_dir()
        with open(VLLM_BENCHMARK_LOG, "w") as log_file:
            subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        
        print("等待 vLLM 啟動...")
        time.sleep(15)  # 等待 vLLM 啟動
        
        # 等待 vLLM 完全就緒
        for i in range(30):
            if check_vllm_ready():
                print("✓ vLLM 重啟完成")
                return True
            print(f"  等待中... ({i+1}/30)")
            time.sleep(2)
        
        print("✗ vLLM 重啟超時")
        return False
    except Exception as e:
        print(f"重啟失敗：{e}")
        import traceback
        traceback.print_exc()
        return False


def check_vllm_ready():
    """檢查 vLLM 是否就緒"""
    try:
        response = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_manager_status():
    """檢查 manager API 是否可用"""
    try:
        response = requests.get(f"{MANAGER_URL}/api/status", timeout=5)
        # 即使需要登入也算可用
        if response.status_code in [200, 401]:
            print(f"✓ Manager API 可用")
            return True
    except Exception as e:
        print(f"✗ Manager API 連接失敗：{e}")
    return False


def check_vllm_status():
    """檢查 vLLM API 是否可用"""
    try:
        response = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✓ vLLM API 可用，模型：{[m['id'] for m in models.get('data', [])]}")
            return True
        else:
            print(f"✗ vLLM 返回錯誤狀態碼：{response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ vLLM 連接失敗：{e}")
        return False


def get_metrics():
    """從 vLLM 獲取 metrics"""
    try:
        response = requests.get(f"{VLLM_URL}/metrics", timeout=10)
        if response.status_code == 200:
            return response.text
    except:
        pass
    return None


def parse_mtp_metrics(metrics_text, baseline_accepted=None, baseline_rejected=None):
    """解析 MTP 相關指標
    
    返回:
        acceptance_rate: Acceptance Rate (%)
        mean_acceptance_length: Mean Acceptance Length
        accepted_tokens: 被接受的 token 數
        rejected_tokens: 被拒絕的 token 數
    """
    import re
    
    if not metrics_text:
        return 0.0, 0.0, 0, 0
    
    accepted_tokens = 0
    rejected_tokens = 0
    
    # 解析 vLLM MTP metrics
    # vLLM 輸出的 metrics 格式類似:
    # vllm:num_accepted_tokens{...} 1234
    # vllm:num_rejected_tokens{...} 567
    
    accepted_match = re.search(r'vllm:num_accepted_tokens(?:\{[^}]*\})?\s+([\d.]+)', metrics_text)
    rejected_match = re.search(r'vllm:num_rejected_tokens(?:\{[^}]*\})?\s+([\d.]+)', metrics_text)
    
    if accepted_match:
        accepted_tokens = int(float(accepted_match.group(1)))
    if rejected_match:
        rejected_tokens = int(float(rejected_match.group(1)))
    
    # 計算 acceptance rate
    total = accepted_tokens + rejected_tokens
    acceptance_rate = 0.0
    mean_acceptance_length = 0.0
    
    if total > 0:
        acceptance_rate = (accepted_tokens / total) * 100
    
    # Mean Acceptance Length 計算
    # 理論：Mean Acceptance Length = 1 + num_speculative_tokens * acceptance_rate
    # 實際：從 metrics 中獲取或估算
    if accepted_tokens > 0 and rejected_tokens >= 0:
        # 簡單的估算：假設每次 speculative 嘗試最多 num_speculative_tokens + 1 個 token 被接受
        # Mean Acceptance Length ≈ accepted / (accepted + rejected) * (num_speculative + 1) + rejected / total * 1
        if total > 0:
            # 更精確的估算：平均每次 speculative 步驟接受的 token 數
            mean_acceptance_length = 1 + (accepted_tokens / total) * 2  # 假設 num_speculative_tokens=2
    
    return acceptance_rate, mean_acceptance_length, accepted_tokens, rejected_tokens


def run_generation(prompt, model_name, num_speculative_tokens=0, baseline_metrics=None):
    """執行一次生成請求並返回性能數據"""
    # vLLM 使用 chat/completions API
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        **GENERATE_PARAMS
    }
    
    # 獲取測試前的 metrics baseline
    pre_metrics = get_metrics()
    pre_ar, pre_mal, pre_accepted, pre_rejected = parse_mtp_metrics(pre_metrics)
    
    start_time = time.perf_counter()
    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=payload,
            timeout=300
        )
        end_time = time.perf_counter()
        
        if response.status_code == 200:
            result = response.json()
            output_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            output_tokens = result.get("usage", {}).get("completion_tokens", 0)
            
            elapsed = end_time - start_time
            tps = output_tokens / elapsed if elapsed > 0 else 0
            
            # 獲取測試後的 metrics
            post_metrics = get_metrics()
            post_ar, post_mal, post_accepted, post_rejected = parse_mtp_metrics(post_metrics)
            
            # 計算本次請求的 MTP 指標
            if num_speculative_tokens > 0:
                accepted = post_accepted - pre_accepted
                rejected = post_rejected - pre_rejected
                total = accepted + rejected
                
                acceptance_rate = (accepted / total * 100) if total > 0 else 0
                # Mean Acceptance Length = 1 + num_speculative_tokens * acceptance_rate (理論上限)
                mean_acceptance_length = 1 + num_speculative_tokens * (acceptance_rate / 100) if acceptance_rate > 0 else 1
            else:
                acceptance_rate = 0
                mean_acceptance_length = 0
            
            return {
                "success": True,
                "elapsed_time": elapsed,
                "output_tokens": output_tokens,
                "tokens_per_second": tps,
                "acceptance_rate": acceptance_rate,
                "mean_acceptance_length": mean_acceptance_length,
                "accepted_tokens": post_accepted - pre_accepted if num_speculative_tokens > 0 else 0,
                "rejected_tokens": post_rejected - pre_rejected if num_speculative_tokens > 0 else 0
            }
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "tokens_per_second": 0,
                "acceptance_rate": 0,
                "mean_acceptance_length": 0
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timeout",
            "tokens_per_second": 0,
            "acceptance_rate": 0,
            "mean_acceptance_length": 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tokens_per_second": 0,
            "acceptance_rate": 0,
            "mean_acceptance_length": 0
        }


def build_matrix_variants():
    """建立要測試的 MTP matrix。"""
    variants = []
    for spec_method in ["qwen3_next_mtp", "mtp"]:
        for num_tokens in [1, 2]:
            for prefix_caching in [False, True]:
                variant_id = f"{spec_method}_n{num_tokens}_pc_{'on' if prefix_caching else 'off'}"
                variants.append({
                    "id": variant_id,
                    "label": (
                        f"method={spec_method}, num_tokens={num_tokens}, "
                        f"prefix_cache={'on' if prefix_caching else 'off'}"
                    ),
                    "speculative_enabled": True,
                    "speculative_method": spec_method,
                    "speculative_num_tokens": num_tokens,
                    "enable_prefix_caching": prefix_caching,
                })
    return variants


def run_benchmark_variant(variant, model_name):
    """運行單一 matrix 組合的 benchmark。"""
    print(f"\n{'='*70}")
    print(f"測試模式：{variant['label']}")
    print(f"{'='*70}")

    if not update_benchmark_config(
        variant["speculative_enabled"],
        variant["speculative_num_tokens"],
        variant["speculative_method"],
        variant["enable_prefix_caching"],
    ):
        print("✗ 無法套用此組 benchmark 配置")
        return [], [], [], []
    
    results = []
    all_tps = []
    all_acceptance_rates = []
    all_mean_acceptance_lengths = []

    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n提示詞 {i}/{len(PROMPTS)} (長度：{len(prompt)} 字符)...")

        prompt_results = []
        for run in range(RUNS_PER_PROMPT):
            print(f"  運行 {run+1}/{RUNS_PER_PROMPT}...", end=" ", flush=True)
            result = run_generation(prompt, model_name, variant["speculative_num_tokens"])

            if result["success"]:
                tps = result['tokens_per_second']
                ar = result['acceptance_rate']
                mal = result['mean_acceptance_length']

                print(f"✓ {result['output_tokens']} tokens, {tps:.2f} tok/s, Acceptance: {ar:.1f}%")

                prompt_results.append(result)
                all_tps.append(tps)
                all_acceptance_rates.append(ar)
                all_mean_acceptance_lengths.append(mal)
            else:
                print(f"✗ {result.get('error', 'Unknown error')}")

        if prompt_results:
            avg_result = {
                "prompt_index": i,
                "prompt_length": len(prompt),
                "avg_tokens": statistics.mean([r["output_tokens"] for r in prompt_results]),
                "avg_tps": statistics.mean([r["tokens_per_second"] for r in prompt_results]),
                "avg_time": statistics.mean([r["elapsed_time"] for r in prompt_results]),
                "avg_acceptance_rate": statistics.mean([r["acceptance_rate"] for r in prompt_results]),
                "avg_mean_acceptance_length": statistics.mean([r["mean_acceptance_length"] for r in prompt_results]),
            }
            results.append(avg_result)

    return results, all_tps, all_acceptance_rates, all_mean_acceptance_lengths


def summarize_variant(variant, detailed_results, all_tps, all_acceptance_rates, all_mean_acceptance_lengths):
    """彙整單一組合的 benchmark 結果。"""
    return {
        "id": variant["id"],
        "label": variant["label"],
        "config": {
            "speculative_method": variant["speculative_method"],
            "speculative_num_tokens": variant["speculative_num_tokens"],
            "enable_prefix_caching": variant["enable_prefix_caching"],
        },
        "avg_tps": statistics.mean(all_tps) if all_tps else 0,
        "avg_acceptance_rate": statistics.mean(all_acceptance_rates) if all_acceptance_rates else 0,
        "avg_mean_acceptance_length": statistics.mean(all_mean_acceptance_lengths) if all_mean_acceptance_lengths else 0,
        "total_tokens": sum(r["avg_tokens"] for r in detailed_results),
        "detailed": detailed_results,
    }


def print_matrix_summary(matrix_results):
    """打印 matrix benchmark 摘要與排序。"""
    print(f"\n{'='*70}")
    print("MTP Benchmark Matrix 總結")
    print(f"{'='*70}")

    ranked = sorted(matrix_results, key=lambda item: item["avg_tps"], reverse=True)
    best_tps = ranked[0]["avg_tps"] if ranked else 0

    print("\n【吞吐排序】")
    print(f"{'Rank':<6} {'Method':<16} {'NumTok':<8} {'Prefix':<8} {'TPS':<10} {'AccRate':<10} {'MAL':<8} {'vs Best':<10}")
    print("-" * 86)
    for idx, item in enumerate(ranked, 1):
        best_delta = ((item["avg_tps"] / best_tps) - 1) * 100 if best_tps > 0 else 0
        print(
            f"{idx:<6} "
            f"{item['config']['speculative_method']:<16} "
            f"{item['config']['speculative_num_tokens']:<8} "
            f"{('on' if item['config']['enable_prefix_caching'] else 'off'):<8} "
            f"{item['avg_tps']:<10.2f} "
            f"{item['avg_acceptance_rate']:<10.1f} "
            f"{item['avg_mean_acceptance_length']:<8.2f} "
            f"{best_delta:+.1f}%"
        )

    print("\n【各維度最佳組合】")
    best_acceptance = max(matrix_results, key=lambda item: item["avg_acceptance_rate"], default=None)
    best_mal = max(matrix_results, key=lambda item: item["avg_mean_acceptance_length"], default=None)
    if ranked:
        print(f"  吞吐最佳：{ranked[0]['label']} -> {ranked[0]['avg_tps']:.2f} tok/s")
    if best_acceptance:
        print(f"  Acceptance 最佳：{best_acceptance['label']} -> {best_acceptance['avg_acceptance_rate']:.1f}%")
    if best_mal:
        print(f"  Mean Acceptance Length 最佳：{best_mal['label']} -> {best_mal['avg_mean_acceptance_length']:.2f}")

    print("\n【各提示詞最佳吞吐】")
    for prompt_index in range(1, len(PROMPTS) + 1):
        prompt_rows = []
        for item in matrix_results:
            detail = next((row for row in item["detailed"] if row["prompt_index"] == prompt_index), None)
            if detail:
                prompt_rows.append((item["label"], detail["avg_tps"], detail["avg_acceptance_rate"]))
        if prompt_rows:
            best_label, best_tps, best_ar = max(prompt_rows, key=lambda row: row[1])
            print(f"  Prompt {prompt_index}: {best_label} -> {best_tps:.2f} tok/s, Acceptance {best_ar:.1f}%")


def save_results(results, filename):
    """保存測試結果到文件"""
    ensure_benchmark_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = BENCHMARK_DIR / f"{filename}_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n測試結果已保存到：{output_file}")


def main():
    print("="*70)
    print("Qwen3.5 MTP Benchmark Matrix")
    print("測試指標：Throughput, Acceptance Rate, Mean Acceptance Length")
    print("="*70)
    
    # 檢查 Manager API 狀態
    print("\n檢查服務狀態...")
    if not check_manager_status():
        print("\n請先啟動 vllm-manager 服務")
        print("啟動命令：systemctl start vllm-manager 或 ./auto-start.sh")
        return
    
    # 檢查 vLLM 狀態
    if not check_vllm_status():
        print("\n請先啟動 vLLM 服務")
        return
    
    # 獲取當前啟用的模型
    model = get_enabled_model()
    if not model:
        print("\n錯誤：沒有啟用的模型")
        return
    
    model_name = model.get("name", "unknown")
    model_path = model.get("path", "")
    # vLLM API 需要完整的模型路徑作為模型 ID
    model_id = model_path  # 使用完整路徑作為 API 中的模型標識符
    print(f"\n當前模型：{model_name} ({model_path})")

    variants = build_matrix_variants()

    print("\n測試將比較以下 8 組 MTP matrix:")
    for idx, variant in enumerate(variants, 1):
        print(f"{idx}. {variant['label']}")
    print("\n每個測試會自動重啟 vLLM 服務以應用新配置")
    print("\n測試即將開始...")
    time.sleep(2)  # 給用戶 2 秒時間準備

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "model_path": model_path,
        "test_prompts": len(PROMPTS),
        "runs_per_prompt": RUNS_PER_PROMPT,
        "matrix": []
    }

    original_snapshot = snapshot_enabled_model_params()

    try:
        for variant in variants:
            detailed, all_tps, all_acceptance_rates, all_mean_acceptance_lengths = run_benchmark_variant(
                variant,
                model_id,
            )
            all_results["matrix"].append(
                summarize_variant(
                    variant,
                    detailed,
                    all_tps,
                    all_acceptance_rates,
                    all_mean_acceptance_lengths,
                )
            )
    finally:
        if restore_enabled_model_params(original_snapshot):
            print("正在還原原始 vLLM 服務配置...")
            restart_vllm()

    print_matrix_summary(all_results["matrix"])
    save_results(all_results, "benchmark_matrix")
    
    print(f"\n{'='*70}")
    print("測試完成!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
