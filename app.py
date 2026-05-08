"""
Controller 層 - Flask 路由

API 端點:
  GET  /                    - 主頁面
  GET  /api/status          - vLLM 狀態
  POST /api/start           - 啟動 vLLM
  POST /api/stop            - 停止 vLLM
  POST /api/restart         - 重啟 vLLM
  GET  /api/config          - 取得設定
  POST /api/config          - 更新設定
  GET  /api/models          - 模型清單
  POST /api/models          - 新增模型
  POST /api/models/scan     - 掃描 models_dir
  POST /api/models/clone    - 從 Hugging Face clone 模型
  POST /api/models/<path>   - 啟用模型
  DELETE /api/models/<path> - 移除模型
  GET  /api/clone-status    - clone 進度
  GET  /api/logs            - 最新日誌
  GET  /api/logs/stream     - SSE 即時日誌串流
  GET  /api/logs/<n>        - 取 n 行日誌
  GET  /api/gpu             - GPU 狀態
"""

from flask import Flask, jsonify, request, render_template, Response
import threading
import time
from models.vllm_manager import VLLMManager

app = Flask(__name__)
manager = VLLMManager()

# Clone 操作狀態
_clone_status = {"running": False, "progress": "", "result": None}


# ---- 頁面 ----

@app.route("/")
def index():
    return render_template("index.html")


# ---- vLLM 狀態與控制 ----

@app.route("/api/status")
def api_status():
    return jsonify(manager.get_status())


@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.get_json(silent=True) or {}
    result = manager.start(
        model_path=data.get("model_path"),
        params=data.get("params")
    )
    return jsonify(result)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    return jsonify(manager.stop())


@app.route("/api/restart", methods=["POST"])
def api_restart():
    data = request.get_json(silent=True) or {}
    result = manager.restart(
        model_path=data.get("model_path"),
        params=data.get("params")
    )
    return jsonify(result)


# ---- 設定 ----

@app.route("/api/config")
def api_get_config():
    return jsonify(manager.get_config())


@app.route("/api/config", methods=["POST"])
def api_update_config():
    result = manager.update_config(request.get_json(silent=True) or {})
    return jsonify(result)


# ---- 模型參數 (per-model params) ----

@app.route("/api/model-params", methods=["GET"])
def api_get_model_params():
    """取得目前啟用模型的合併參數 (defaults + model.params)"""
    result = manager.get_model_params()
    return jsonify(result)


@app.route("/api/model-params", methods=["POST"])
def api_update_model_params():
    """儲存參數到目前啟用模型的 params"""
    data = request.get_json(silent=True) or {}
    result = manager.update_model_params(data)
    return jsonify(result)


# ---- 模型管理 ----

@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify(manager.get_models())


@app.route("/api/models/scan", methods=["POST"])
def api_models_scan():
    return jsonify(manager.scan_models())


@app.route("/api/models/clone", methods=["POST"])
def api_models_clone():
    global _clone_status
    data = request.json or {}
    repo_url = data.get("repo_url", "").strip()
    if not repo_url:
        return jsonify({"status": "error", "message": "請提供 repo_url"}), 400
    if _clone_status["running"]:
        return jsonify({"status": "error", "message": "正在 clone 中，請等待"}), 400
    
    _clone_status = {"running": True, "progress": f"開始 clone: {repo_url}", "result": None}
    
    def do_clone():
        try:
            result = manager.clone_model(repo_url)
            _clone_status["result"] = result
            _clone_status["progress"] = "完成" if result.get("status") == "cloned" else f"失敗: {result.get('message', '')}"
        except Exception as e:
            _clone_status["result"] = {"status": "error", "message": str(e)}
            _clone_status["progress"] = f"錯誤: {e}"
        finally:
            _clone_status["running"] = False
    
    threading.Thread(target=do_clone, daemon=True).start()
    return jsonify({"status": "started", "progress": _clone_status["progress"]})


@app.route("/api/clone-status", methods=["GET"])
def api_clone_status():
    return jsonify({
        "running": manager._clone_thread and manager._clone_thread.is_alive(),
        "progress": manager._clone_progress,
        "percentage": manager._clone_percentage,
        "model_name": manager._clone_model_name,
        "result": manager._clone_result
    })


@app.route("/api/models", methods=["POST"])
def api_add_model():
    data = request.get_json(silent=True) or {}
    result = manager.add_model(
        name=data.get("name", ""),
        path=data.get("path", "")
    )
    return jsonify(result)


@app.route("/api/models/enable", methods=["POST"])
def api_enable_model():
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"status": "error", "message": "缺少 model path"}), 400
    return jsonify(manager.set_enabled_model(path))


@app.route("/api/models/remove", methods=["POST"])
def api_remove_model():
    from flask import request
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"status": "error", "message": "缺少 model path"}), 400
    return jsonify(manager.remove_model(path))


# ---- 日誌 ----

@app.route("/api/logs")
def api_logs():
    n = request.args.get("n", 100, type=int)
    return jsonify(manager.get_logs(lines=n))


@app.route("/api/logs/<int:n>")
def api_logs_n(n):
    return jsonify(manager.get_logs(lines=n))


@app.route("/api/logs/stream")
def api_logs_stream():
    """SSE 即時日誌串流"""
    def generate():
        for chunk in manager.stream_logs():
            yield chunk
    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


# ---- GPU ----

@app.route("/api/gpu")
def api_gpu():
    return jsonify(manager.get_gpu_info())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
