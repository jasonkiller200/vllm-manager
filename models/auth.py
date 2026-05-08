"""
Authentication module - SQLite-based user management & login tracking
"""

import sqlite3
import time
import hashlib
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "auth.db")
LOCKOUT_DURATION = 1800  # 30 minutes in seconds
WARN_ATTEMPTS = 3
LOCK_ATTEMPTS = 6


def _get_db():
    """Get database connection, creating tables if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'operator',
            is_active INTEGER NOT NULL DEFAULT 1,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            success INTEGER NOT NULL,
            reason TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _hash_password(password):
    """Hash password using SHA-256 with salt."""
    import secrets
    salt = secrets.token_hex(16)
    hash_val = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hash_val}"


def _verify_password(password, stored_hash):
    """Verify password against stored hash."""
    try:
        salt, hash_val = stored_hash.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hash_val
    except Exception:
        return False


# ---- User queries ----

def has_any_users():
    """Check if any users exist in the database."""
    conn = _get_db()
    row = conn.execute("SELECT COUNT(*) as cnt FROM users").fetchone()
    conn.close()
    return row["cnt"] > 0


def has_admin():
    """Check if any admin users exist."""
    conn = _get_db()
    row = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE role='admin'").fetchone()
    conn.close()
    return row["cnt"] > 0


def create_user(username, password, role="operator"):
    """Create a new user. Returns dict with status."""
    conn = _get_db()
    try:
        existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            conn.close()
            return {"status": "error", "message": "帳號已存在"}

        password_hash = _hash_password(password)
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role)
        )
        conn.commit()

        conn.execute(
            "INSERT INTO login_logs (username, ip_address, success, reason) VALUES (?, ?, ?, ?)",
            (username, "system", 1, "Account created")
        )
        conn.commit()
        conn.close()
        return {"status": "ok", "message": f"已建立使用者: {username} (角色: {role})"}
    except Exception as e:
        conn.close()
        return {"status": "error", "message": str(e)}


def init_first_admin(username, password):
    """Initialize the first admin user. Only allowed when no users exist."""
    if has_any_users():
        return {"status": "error", "message": "系統已有使用者，無法初始化"}
    return create_user(username, password, role="admin")


def authenticate_user(username, password, ip_address):
    """Authenticate a user. Returns dict with login result and user info."""
    conn = _get_db()

    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

    if not user:
        # Log failed attempt for non-existent user
        conn.execute(
            "INSERT INTO login_logs (username, ip_address, success, reason) VALUES (?, ?, ?, ?)",
            (username, ip_address, 0, "User not found")
        )
        conn.commit()
        conn.close()
        return {"status": "error", "message": "帳號或密碼錯誤", "locked": False}

    if not user["is_active"]:
        conn.execute(
            "INSERT INTO login_logs (username, ip_address, success, reason) VALUES (?, ?, ?, ?)",
            (username, ip_address, 0, "Account disabled")
        )
        conn.commit()
        conn.close()
        return {"status": "error", "message": "帳號已被停用", "locked": False}

    # Check lockout
    if user["locked_until"] and user["locked_until"] > time.time():
        remaining = int((user["locked_until"] - time.time()) / 60)
        conn.execute(
            "INSERT INTO login_logs (username, ip_address, success, reason) VALUES (?, ?, ?, ?)",
            (username, ip_address, 0, f"Account locked ({remaining} min remaining)")
        )
        conn.commit()
        conn.close()
        return {
            "status": "error",
            "message": f"帳號已鎖定，請等待 {remaining} 分鐘後再試",
            "locked": True
        }

    # Verify password
    if not _verify_password(password, user["password_hash"]):
        # Failed attempt
        new_attempts = user["failed_attempts"] + 1
        if new_attempts >= LOCK_ATTEMPTS:
            locked_until = int(time.time()) + LOCKOUT_DURATION
            conn.execute(
                "UPDATE users SET failed_attempts=?, locked_until=? WHERE id=?",
                (new_attempts, locked_until, user["id"])
            )
            reason = f"Wrong password ({new_attempts} attempts, locked)"
        else:
            conn.execute(
                "UPDATE users SET failed_attempts=? WHERE id=?",
                (new_attempts, user["id"])
            )
            if new_attempts >= WARN_ATTEMPTS:
                reason = f"Wrong password ({new_attempts}/{LOCK_ATTEMPTS} - WARNING)"
            else:
                reason = f"Wrong password ({new_attempts}/{LOCK_ATTEMPTS} attempts)"

        conn.execute(
            "INSERT INTO login_logs (username, ip_address, success, reason) VALUES (?, ?, ?, ?)",
            (username, ip_address, 0, reason)
        )
        conn.commit()
        conn.close()

        if new_attempts >= LOCK_ATTEMPTS:
            return {"status": "error", "message": f"密碼錯誤，帳號已鎖定 30 分鐘", "locked": True}
        else:
            remaining_warn = LOCK_ATTEMPTS - new_attempts
            if new_attempts >= WARN_ATTEMPTS:
                msg = f"密碼錯誤！已累積 {new_attempts} 次失敗，再失敗 {remaining_warn} 次將鎖定 30 分鐘"
            else:
                msg = f"帳號或密碼錯誤（剩餘 {remaining_warn + 1} 次嘗試）"
            return {"status": "error", "message": msg, "locked": False}

    # Successful login
    conn.execute(
        "UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?",
        (user["id"],)
    )
    conn.execute(
        "INSERT INTO login_logs (username, ip_address, success, reason) VALUES (?, ?, ?, ?)",
        (username, ip_address, 1, "Login successful")
    )
    conn.commit()
    conn.close()

    return {
        "status": "ok",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "role": user["role"]
        }
    }


def get_all_users():
    """Get all users for admin panel."""
    conn = _get_db()
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(u) for u in users]


def update_user_role(user_id, role):
    """Update user role. Admin cannot be downgraded."""
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return {"status": "error", "message": "使用者不存在"}
    if user["role"] == "admin" and role != "admin":
        conn.close()
        return {"status": "error", "message": "管理者不可降權"}
    conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": f"已更新角色為 {role}"}


def toggle_user_active(user_id, is_active):
    """Enable/disable a user. Admin cannot be disabled."""
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return {"status": "error", "message": "使用者不存在"}
    if user["role"] == "admin":
        conn.close()
        return {"status": "error", "message": "管理者不可被停用"}
    conn.execute("UPDATE users SET is_active=? WHERE id=?", (1 if is_active else 0, user_id))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": f"已{'啟用' if is_active else '停用'}使用者"}


def unlock_user(user_id):
    """Unlock a locked user."""
    conn = _get_db()
    conn.execute("UPDATE users SET locked_until=NULL, failed_attempts=0 WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "已解鎖使用者"}


def delete_user(user_id):
    """Delete a user. Cannot delete admin."""
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return {"status": "error", "message": "使用者不存在"}
    if user["role"] == "admin":
        # Check if there are other admins
        count = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE role='admin'").fetchone()["cnt"]
        if count <= 1:
            conn.close()
            return {"status": "error", "message": "至少需要一位管理者"}
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "已刪除使用者"}


def reset_failed_attempts(user_id):
    """Reset failed login attempts for a user."""
    conn = _get_db()
    conn.execute("UPDATE users SET failed_attempts=0, locked_until=NULL WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "已重置失敗次數"}


def get_login_logs(limit=100):
    """Get login logs for admin panel."""
    conn = _get_db()
    logs = conn.execute(
        "SELECT * FROM login_logs ORDER BY timestamp DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(l) for l in logs]


def clear_login_logs():
    """Clear all login logs."""
    conn = _get_db()
    conn.execute("DELETE FROM login_logs")
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "已清除登入日誌"}
