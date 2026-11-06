"""Tests for the AlertManager (cooldowns, persistent event log, webhook forwarding)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from core.alerts import AlertManager


def test_make_alert_schema(tmp_path):
    am = AlertManager(events_log_path=str(tmp_path / "e.jsonl"), site_id="BOP-7")
    a = am.make_alert("INTRUSION", "cam1", "CRITICAL", "breach", track_id=4, skip=None)
    for key in ("id", "type", "alert_type", "risk_level", "message", "camera", "camera_id", "timestamp"):
        assert key in a
    assert a["site_id"] == "BOP-7" and a["track_id"] == 4 and "skip" not in a
    assert am.counts["INTRUSION"] == 1


def test_cooldown(tmp_path):
    am = AlertManager(events_log_path=None)
    assert am.allow("k", 10, now=0)
    assert not am.allow("k", 10, now=5)
    assert am.allow("k", 10, now=11)
    assert am.allow("other", 10, now=5)


def test_event_log_persist_and_read(tmp_path):
    path = tmp_path / "logs" / "e.jsonl"
    am = AlertManager(events_log_path=str(path))
    am.publish([am.make_alert("A", "cam1", "NORMAL", "x"), am.make_alert("B", "cam2", "WARNING", "y")])
    am.publish([am.make_alert("A", "cam1", "NORMAL", "z")])
    lines = path.read_text().strip().splitlines()
    assert len(lines) == 3
    events = am.read_events(limit=10)
    assert events[0]["message"] == "z"
    assert [e["type"] for e in am.read_events(alert_type="B")] == ["B"]


def test_log_rotation(tmp_path):
    path = tmp_path / "e.jsonl"
    am = AlertManager(events_log_path=str(path), max_log_bytes=200)
    for _ in range(10):
        am.publish([am.make_alert("A", "cam1", "NORMAL", "x" * 50)])
    assert (tmp_path / "e.jsonl.1").exists()


def test_webhook_delivery_respects_severity(tmp_path):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        am = AlertManager(events_log_path=None, webhook_url=f"http://127.0.0.1:{server.server_port}/hook",
                          webhook_min_severity="WARNING")
        am.publish([am.make_alert("INFO", "cam1", "NORMAL", "ignored"),
                    am.make_alert("INTRUSION", "cam1", "CRITICAL", "breach")])
        deadline = time.time() + 5
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert len(received) == 1
        assert received[0]["source"] == "SHIELD-IBVAP"
        assert received[0]["event"]["type"] == "INTRUSION"
    finally:
        server.shutdown()
