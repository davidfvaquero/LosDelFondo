"""
aws/rds_logger.py
==================
Módulo de logging a RDS PostgreSQL para telemetría del chatbot.
Reemplaza el log_chat() de main.py con persistencia en base de datos.

Tabla: chat_logs
  - id (SERIAL)
  - timestamp (TIMESTAMPTZ)
  - user_ip (VARCHAR)
  - prompt (TEXT)
  - is_toxic (BOOLEAN)
  - toxic_score (FLOAT)
  - response_length (INT)
  - lang (VARCHAR)

Variables de entorno:
  DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger("deportedata.rds")

# ── Conexión lazy a PostgreSQL ───────────────────────────────────────────────
_conn = None

def _get_connection():
    """Retorna conexión singleton a RDS. Si falla, devuelve None (degraded mode)."""
    global _conn
    if _conn is not None:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    db_host = os.environ.get("DB_HOST", "")
    db_name = os.environ.get("DB_NAME", "deportedata")
    db_user = os.environ.get("DB_USER", "deporteadmin")
    db_pass = os.environ.get("DB_PASSWORD", "")

    if not db_host:
        return None  # Sin RDS configurado (entorno local)

    try:
        import psycopg2
        _conn = psycopg2.connect(
            host=db_host,
            dbname=db_name,
            user=db_user,
            password=db_pass,
            connect_timeout=5,
        )
        _conn.autocommit = True
        _ensure_table(_conn)
        log.info(f"✅ Conectado a RDS PostgreSQL: {db_host}/{db_name}")
        return _conn
    except Exception as e:
        log.warning(f"⚠️ No se pudo conectar a RDS: {e}. Usando log local como fallback.")
        return None


def _ensure_table(conn):
    """Crea la tabla chat_logs si no existe."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id               SERIAL PRIMARY KEY,
                timestamp        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                user_ip          VARCHAR(45),
                prompt           TEXT,
                is_toxic         BOOLEAN,
                toxic_score      FLOAT,
                response_length  INTEGER,
                lang             VARCHAR(5)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_logs_timestamp ON chat_logs(timestamp DESC);
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_logs_ip ON chat_logs(user_ip);
        """)


# ── Función principal de log ─────────────────────────────────────────────────
CHAT_LOGS_FILE = os.environ.get("CHAT_LOGS_FILE", "chat_logs.jsonl")

def log_chat(
    ip: str,
    prompt: str,
    is_toxic: bool,
    toxic_score: float = 0.0,
    response_length: int = 0,
    lang: str = "ES",
):
    """
    Guarda un registro de chat en RDS PostgreSQL.
    Fallback automático a archivo JSONL local si RDS no está disponible.
    """
    conn = _get_connection()

    if conn is not None:
        # ── Guardar en RDS ────────────────────────────────────────────────
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_logs (user_ip, prompt, is_toxic, toxic_score, response_length, lang)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (ip, prompt, is_toxic, toxic_score, response_length, lang)
                )
            return
        except Exception as e:
            log.error(f"Error guardando en RDS: {e}")

    # ── Fallback: archivo JSONL local ─────────────────────────────────────
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ip": ip,
            "prompt": prompt,
            "is_toxic": is_toxic,
            "toxic_score": round(toxic_score, 4),
            "response_length": response_length,
            "lang": lang,
        }
        with open(CHAT_LOGS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Error guardando log local: {e}")


# ── Función para el endpoint admin/chat_logs ──────────────────────────────────
def get_recent_logs(limit: int = 100) -> list[dict]:
    """
    Recupera los últimos N logs desde RDS o desde el archivo local.
    """
    conn = _get_connection()

    if conn is not None:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT timestamp, user_ip, prompt, is_toxic, toxic_score, response_length, lang
                    FROM chat_logs
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
                rows = cur.fetchall()
                return [
                    {
                        "timestamp": str(r[0]),
                        "ip": r[1],
                        "prompt": r[2],
                        "is_toxic": r[3],
                        "toxic_score": r[4],
                        "response_length": r[5],
                        "lang": r[6],
                    }
                    for r in rows
                ]
        except Exception as e:
            log.error(f"Error leyendo RDS: {e}")

    # Fallback a archivo local
    logs = []
    if os.path.exists(CHAT_LOGS_FILE):
        with open(CHAT_LOGS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(logs))[-limit:]


# ── Estadísticas para el panel admin ─────────────────────────────────────────
def get_admin_stats() -> dict:
    """Devuelve estadísticas reales de uso desde RDS."""
    conn = _get_connection()

    if conn is None:
        return None

    try:
        with conn.cursor() as cur:
            # Total consultas
            cur.execute("SELECT COUNT(*) FROM chat_logs")
            total = cur.fetchone()[0]

            # Consultas últimas 24h
            cur.execute("""
                SELECT COUNT(*) FROM chat_logs
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            last24h = cur.fetchone()[0]

            # Intentos tóxicos
            cur.execute("SELECT COUNT(*) FROM chat_logs WHERE is_toxic = TRUE")
            toxic_count = cur.fetchone()[0]

            # IPs únicas últimas 24h (usuarios activos)
            cur.execute("""
                SELECT COUNT(DISTINCT user_ip) FROM chat_logs
                WHERE timestamp > NOW() - INTERVAL '24 hours'
            """)
            active_ips = cur.fetchone()[0]

            # Consultas por día (últimos 7 días)
            cur.execute("""
                SELECT DATE(timestamp), COUNT(*)
                FROM chat_logs
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY DATE(timestamp)
                ORDER BY DATE(timestamp)
            """)
            daily = cur.fetchall()

            return {
                "total_queries": total,
                "queries_24h": last24h,
                "toxic_attempts": toxic_count,
                "active_users": active_ips,
                "daily_counts": [{"date": str(r[0]), "count": r[1]} for r in daily],
            }
    except Exception as e:
        log.error(f"Error obteniendo stats de RDS: {e}")
        return None
