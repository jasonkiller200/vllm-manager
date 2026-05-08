"""
Controller 層 - Flask 路由

API 端點:
  GET  /                    - 主頁面 (需登入)
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
  POST /api/models/enable   - 啟用模型
  POST /api/models/remove   - 移除模型
  GET  /api/clone-status    - clone 進度
  GET  /api/logs            - 最新日誌
  GET  /api/logs/stream     - SSE 即時日誌串流
  GET  /api/logs/<n>        - 取 n 行日誌
  GET  /api/gpu             - GPU 狀態

  GET  /login               - 登入頁面
  POST /api/auth/login      - 登入
  GET  /logout              - 登出
  GET  /init                - 初始化管理者
  POST /api/auth/init       - 建立首位管理者
  GET  /admin               - 系統管理頁面 (Admin only)
  GET  /api/auth/users      - 使用者清單 (Admin only)
  POST /api/auth/users      - 新增使用者 (Admin only)
  GET  /api/auth/login-logs - 登入日誌 (Admin only)
"""

from flask import Flask, jsonify, request, render_template, Response, session, redirect, url_for
from datetime import datetime
import threading
import time
from models.vllm_manager import VLLMManager
from models.auth import (
    has_any_users, has_admin, authenticate_user, init_first_admin,
    create_user, get_all_users, update_user_role, toggle_user_active,
    unlock_user, reset_failed_attempts, delete_user,
    get_login_logs, clear_login_logs
)

app = Flask(__name__)
app.secret_key = 'vllm-manager-secret-key-' + str(time.time()).replace('.', '')
manager = VLLMManager()

# Clone 操作狀態
_clone_status = {"running": False, "progress": "", "result": None}


# ---- 權限檢查工具 ----

def require_login():
    """Check if user is logged in. Returns user dict or None."""
    return session.get('user')


def require_admin():
    """Check if current user is admin. Returns user dict or None."""
    user = session.get('user')
    if user and user.get('role') == 'admin':
        return user
    return None


def get_client_ip():
    """Get client IP address."""
    return request.headers.get('X-Forwarded-For', request.remote_addr or 'unknown')


# ---- 頁面 ----

@app.route("/")
def index():
    user = require_login()
    if not user:
        # Check if system is initialized
        if not has_any_users():
            return redirect(url_for('init_page'))
        return redirect(url_for('login_page'))
    return render_template("index.html")


@app.route("/login")
def login_page():
    user = require_login()
    if user:
        return redirect(url_for('index'))
    return render_template("login.html", now=datetime.now())


@app.route("/logout")
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))


@app.route("/init")
def init_page():
    # Only accessible when no users exist
    if has_any_users():
        return redirect(url_for('login_page'))
    return render_template("init.html")


@app.route("/admin")
def admin_page():
    user = require_admin()
    if not user:
        return redirect(url_for('login_page'))
    return render_template("admin.html", current_user=user)


# ---- 認證 API ----

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    ip_address = get_client_ip()

    if not username or not password:
        return jsonify({"status": "error", "message": "請輸入帳號和密碼"}), 400

    result = authenticate_user(username, password, ip_address)

    if result["status"] == "ok":
        session['user'] = result['user']
        return jsonify(result)
    else:
        return jsonify(result), 401


@app.route("/api/auth/init", methods=["POST"])
def api_init():
    if has_any_users():
        return jsonify({"status": "error", "message": "系統已有使用者"}), 400

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"status": "error", "message": "請輸入帳號和密碼"}), 400

    if len(password) < 6:
        return jsonify({"status": "error", "message": "密碼至少需要 6 個字元"}), 400

    result = init_first_admin(username, password)
    if result["status"] == "ok":
        return jsonify(result)
    else:
        return jsonify(result), 400


# ---- 使用者管理 API (Admin only) ----

@app.route("/api/auth/users", methods=["GET"])
def api_get_users():
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403
    return jsonify(get_all_users())


@app.route("/api/auth/users", methods=["POST"])
def api_create_user():
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403

    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    role = data.get("role", "operator")

    if not username or not password:
        return jsonify({"status": "error", "message": "請輸入帳號和密碼"}), 400

    if len(password) < 6:
        return jsonify({"status": "error", "message": "密碼至少需要 6 個字元"}), 400

    result = create_user(username, password, role)
    if result["status"] == "ok":
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route("/api/auth/users/<int:user_id>/role", methods=["POST"])
def api_update_user_role(user_id):
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403

    data = request.get_json(silent=True) or {}
    role = data.get("role", "")
    if role not in ("admin", "operator"):
        return jsonify({"status": "error", "message": "無效的角色"}), 400

    result = update_user_role(user_id, role)
    if result["status"] == "ok":
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route("/api/auth/users/<int:user_id>/toggle", methods=["POST"])
def api_toggle_user(user_id):
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403

    data = request.get_json(silent=True) or {}
    is_active = data.get("is_active", 0)

    result = toggle_user_active(user_id, is_active)
    if result["status"] == "ok":
        return jsonify(result)
    else:
        return jsonify(result), 400


@app.route("/api/auth/users/<int:user_id>/unlock", methods=["POST"])
def api_unlock_user(user_id):
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403

    result = unlock_user(user_id)
    return jsonify(result)


@app.route("/api/auth/users/<int:user_id>/reset-attempts", methods=["POST"])
def api_reset_attempts(user_id):
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403

    result = reset_failed_attempts(user_id)
    return jsonify(result)


@app.route("/api/auth/users/<int:user_id>", methods=["DELETE"])
def api_delete_user(user_id):
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403

    result = delete_user(user_id)
    if result["status"] == "ok":
        return jsonify(result)
    else:
        return jsonify(result), 400


# ---- 登入日誌 API (Admin only) ----

@app.route("/api/auth/login-logs", methods=["GET"])
def api_get_login_logs():
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403
    return jsonify(get_login_logs(limit=200))


@app.route("/api/auth/login-logs", methods=["DELETE"])
def api_clear_login_logs():
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403
    result = clear_login_logs()
    return jsonify(result)


# ---- vLLM 狀態與控制 ----

@app.route("/api/status")
def api_status():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.get_status())


@app.route("/api/start", methods=["POST"])
def api_start():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    data = request.get_json(silent=True) or {}
    result = manager.start(
        model_path=data.get("model_path"),
        params=data.get("params")
    )
    return jsonify(result)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.stop())


@app.route("/api/restart", methods=["POST"])
def api_restart():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    data = request.get_json(silent=True) or {}
    result = manager.restart(
        model_path=data.get("model_path"),
        params=data.get("params")
    )
    return jsonify(result)


# ---- 設定 ----

@app.route("/api/config")
def api_get_config():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.get_config())


@app.route("/api/config", methods=["POST"])
def api_update_config():
    user = require_admin()
    if not user:
        return jsonify({"status": "error", "message": "需要管理者權限"}), 403
    result = manager.update_config(request.get_json(silent=True) or {})
    return jsonify(result)


# ---- 模型參數 (per-model params) ----

@app.route("/api/model-params", methods=["GET"])
def api_get_model_params():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    result = manager.get_model_params()
    return jsonify(result)


@app.route("/api/model-params", methods=["POST"])
def api_update_model_params():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    data = request.get_json(silent=True) or {}
    result = manager.update_model_params(data)
    return jsonify(result)


# ---- 模型管理 ----

@app.route("/api/models", methods=["GET"])
def api_models():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.get_models())


@app.route("/api/models/scan", methods=["POST"])
def api_models_scan():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.scan_models())


@app.route("/api/models/clone", methods=["POST"])
def api_models_clone():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
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
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify({
        "running": manager._clone_thread and manager._clone_thread.is_alive(),
        "progress": manager._clone_progress,
        "percentage": manager._clone_percentage,
        "model_name": manager._clone_model_name,
        "result": manager._clone_result
    })


@app.route("/api/models", methods=["POST"])
def api_add_model():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    data = request.get_json(silent=True) or {}
    result = manager.add_model(
        name=data.get("name", ""),
        path=data.get("path", "")
    )
    return jsonify(result)


@app.route("/api/models/enable", methods=["POST"])
def api_enable_model():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"status": "error", "message": "缺少 model path"}), 400
    return jsonify(manager.set_enabled_model(path))


@app.route("/api/models/remove", methods=["POST"])
def api_remove_model():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    from flask import request
    data = request.get_json(silent=True) or {}
    path = data.get("path")
    if not path:
        return jsonify({"status": "error", "message": "缺少 model path"}), 400
    return jsonify(manager.remove_model(path))


# ---- 日誌 ----

@app.route("/api/logs")
def api_logs():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    n = request.args.get("n", 100, type=int)
    return jsonify(manager.get_logs(lines=n))


@app.route("/api/logs/<int:n>")
def api_logs_n(n):
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.get_logs(lines=n))


@app.route("/api/logs/stream")
def api_logs_stream():
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
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
    user = require_login()
    if not user:
        return jsonify({"status": "error", "message": "需要登入"}), 401
    return jsonify(manager.get_gpu_info())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
