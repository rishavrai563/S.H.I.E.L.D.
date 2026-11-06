"""
SHIELD Alert Manager
--------------------
Single place where security events become alerts:

  * a uniform alert schema (id, type, risk_level, message, camera, metadata),
  * per-key cooldowns so one incident doesn't flood operators,
  * a persistent event log (logs/events.jsonl, size-rotated) for audit/replay,
  * optional forwarding to an existing Command & Control system via HTTP
    webhook (JSON POST, background thread, never blocks the video pipeline).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import urllib.request
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shield.alerts")

SEVERITY_ORDER = {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}


class AlertManager:
    def __init__(
        self,
        events_log_path: Optional[str] = "logs/events.jsonl",
        max_log_bytes: int = 50 * 1024 * 1024,
        webhook_url: str = "",
        webhook_min_severity: str = "WARNING",
        webhook_timeout_sec: float = 3.0,
        webhook_headers: Optional[Dict[str, str]] = None,
        site_id: str = "",
    ):
        self.events_log_path = events_log_path
        self.max_log_bytes = int(max_log_bytes)
        self.site_id = site_id
        self._cooldowns: Dict[str, float] = {}
        self._log_lock = threading.Lock()
        self.counts: Dict[str, int] = {}

        if events_log_path:
            os.makedirs(os.path.dirname(os.path.abspath(events_log_path)), exist_ok=True)

        self.webhook_url = webhook_url or ""
        self.webhook_min = SEVERITY_ORDER.get(str(webhook_min_severity).upper(), 1)
        self.webhook_timeout = float(webhook_timeout_sec)
        self.webhook_headers = {"Content-Type": "application/json", **(webhook_headers or {})}
        self._webhook_q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)
        self.webhook_stats = {"sent": 0, "failed": 0, "dropped": 0}
        if self.webhook_url:
            threading.Thread(target=self._webhook_worker, daemon=True, name="AlertWebhook").start()

    # ── creation / throttling ──────────────────────────────

    def allow(self, key: str, cooldown_sec: float, now: Optional[float] = None) -> bool:
        """True if an alert with this key may fire now (and starts its cooldown)."""
        now = time.time() if now is None else now
        last = self._cooldowns.get(key)
        if last is not None and now - last < cooldown_sec:
            return False
        self._cooldowns[key] = now
        if len(self._cooldowns) > 5000:  # bound memory
            cutoff = now - 3600
            self._cooldowns = {k: v for k, v in self._cooldowns.items() if v >= cutoff}
        return True

    def make_alert(self, alert_type: str, camera_id: str, risk_level: str, message: str,
                   **metadata: Any) -> Dict[str, Any]:
        ts = time.time()
        alert = {
            "id": f"{alert_type}-{uuid.uuid4().hex[:10]}",
            "type": alert_type,
            "alert_type": alert_type,
            "risk_level": risk_level,
            "message": message,
            "camera": camera_id,
            "camera_id": camera_id,
            "timestamp": ts,
        }
        if self.site_id:
            alert["site_id"] = self.site_id
        alert.update({k: v for k, v in metadata.items() if v is not None})
        self.counts[alert_type] = self.counts.get(alert_type, 0) + 1
        return alert

    # ── output ─────────────────────────────────────────────

    def publish(self, alerts: List[Dict[str, Any]]) -> None:
        """Persist and forward a batch of alerts produced in one frame."""
        if not alerts:
            return
        self._write_log(alerts)
        if self.webhook_url:
            for a in alerts:
                if SEVERITY_ORDER.get(a.get("risk_level", "NORMAL"), 0) >= self.webhook_min:
                    try:
                        self._webhook_q.put_nowait(a)
                    except queue.Full:
                        self.webhook_stats["dropped"] += 1

    def _write_log(self, alerts: List[Dict[str, Any]]) -> None:
        if not self.events_log_path:
            return
        with self._log_lock:
            try:
                if os.path.exists(self.events_log_path) and os.path.getsize(self.events_log_path) > self.max_log_bytes:
                    os.replace(self.events_log_path, self.events_log_path + ".1")
                with open(self.events_log_path, "a") as f:
                    for a in alerts:
                        f.write(json.dumps(a, default=str) + "\n")
            except Exception as e:
                logger.warning("event log write failed: %s", e)

    def read_events(self, limit: int = 100, alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Most recent events from the persistent log (newest first)."""
        if not self.events_log_path or not os.path.exists(self.events_log_path):
            return []
        with self._log_lock:
            with open(self.events_log_path) as f:
                lines = f.readlines()
        out = []
        for line in reversed(lines):
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if alert_type and ev.get("type") != alert_type:
                continue
            out.append(ev)
            if len(out) >= limit:
                break
        return out

    def _webhook_worker(self) -> None:
        while True:
            alert = self._webhook_q.get()
            body = json.dumps({"source": "SHIELD-IBVAP", "event": alert}, default=str).encode()
            for attempt in range(2):
                try:
                    req = urllib.request.Request(self.webhook_url, data=body, headers=self.webhook_headers,
                                                 method="POST")
                    with urllib.request.urlopen(req, timeout=self.webhook_timeout):
                        pass
                    self.webhook_stats["sent"] += 1
                    break
                except Exception as e:
                    if attempt == 1:
                        self.webhook_stats["failed"] += 1
                        logger.debug("webhook delivery failed: %s", e)
                    else:
                        time.sleep(0.5)
