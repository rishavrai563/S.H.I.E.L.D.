"""
Live Webcam Demo — Phase 1 Milestone Gate Test.

Wires up: Camera → Detector → ReID Encoder → Tracker → Display

This tests the full single-camera pipeline end-to-end.
Goal: 3 people tracked with stable IDs, no swaps.

Controls:
    Q / ESC  → Quit
    S        → Screenshot
"""
import sys
import os
import time
from datetime import datetime, timedelta
import csv
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import yaml
try:
    import mediapipe as mp
except Exception:
    mp = None
try:
    import pywhatkit
except Exception:
    pywhatkit = None

from core.detector import PersonDetector
from core.reid_encoder import ReIDEncoder
from core.tracker import SingleCameraTracker


def _normalize_embedding(emb):
    if emb is None:
        return None
    vec = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-8:
        return None
    return vec / norm


def _match_or_create_identity(
    embedding,
    global_db,
    next_global_id,
    threshold=0.62,
    margin_gate=0.05,
    keep_last=12,
    now_ts=0.0,
    reentry_window_sec=120.0,
    recent_boost_window_sec=20.0,
    recent_threshold=0.58,
    recent_margin_gate=0.03,
    blocked_gids=None,
):
    blocked = set(blocked_gids or [])
    scored_candidates = []
    for gid, data in global_db.items():
        if gid in blocked:
            continue
        embs = data.get("embeddings", [])
        if not embs:
            continue
        last_seen = float(data.get("last_seen", 0.0))
        if now_ts > 0 and last_seen > 0 and (now_ts - last_seen) > reentry_window_sec:
            # Ignore very old identities to reduce accidental cross-person matches.
            continue
        sims = [float(np.dot(embedding, e)) for e in embs]
        max_sim = max(sims)
        mean_sim = float(np.mean(sims))
        # Blend max + mean for robustness against noisy single embeddings.
        score = 0.7 * max_sim + 0.3 * mean_sim
        scored_candidates.append((gid, score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    best_gid = scored_candidates[0][0] if scored_candidates else None
    best_score = scored_candidates[0][1] if scored_candidates else -1.0
    second_score = scored_candidates[1][1] if len(scored_candidates) > 1 else -1.0
    margin = best_score - second_score if second_score >= 0 else 1.0

    # Adaptive acceptance:
    # - Recently seen identities get slightly relaxed gate (stability on motion/pose change)
    # - Older identities use stricter gate (avoid wrong merges)
    accept = False
    if best_gid is not None:
        best_last_seen = float(global_db.get(best_gid, {}).get("last_seen", 0.0))
        is_recent = now_ts > 0 and best_last_seen > 0 and (now_ts - best_last_seen) <= recent_boost_window_sec
        thr = recent_threshold if is_recent else threshold
        mg = recent_margin_gate if is_recent else margin_gate
        accept = best_score >= thr and margin >= mg

    if accept:
        embs = global_db[best_gid].setdefault("embeddings", [])
        embs.append(embedding.copy())
        if len(embs) > keep_last:
            del embs[:-keep_last]
        global_db[best_gid]["last_seen"] = now_ts
        return best_gid, best_score, next_global_id, False

    gid = next_global_id
    next_global_id += 1
    global_db[gid] = {"embeddings": [embedding.copy()], "last_seen": now_ts}
    return gid, 0.0, next_global_id, True


def send_whatsapp_message(global_id, camera_name, event_time, phone_number, message=None):
    if message is None:
        message = (
            f"[ALERT] New person detected\n"
            f"Global ID: {global_id}\n"
            f"Camera: {camera_name}\n"
            f"Time: {event_time}"
        )
    msg = message
    if pywhatkit is None:
        print("  ⚠️ pywhatkit not installed; cannot send WhatsApp message.")
        return False
    print(
        f"  📤 Attempting WhatsApp send for G#{global_id} "
        f"to {phone_number} from {camera_name} at {event_time}"
    )
    try:
        pywhatkit.sendwhatmsg_instantly(
            phone_no=phone_number,
            message=msg,
            wait_time=15,
            tab_close=True,
            close_time=5,
        )
        print(f"  ✅ WhatsApp message sent successfully for G#{global_id} (instant)")
        return True
    except Exception as exc:
        print(f"  ⚠️ Instant send failed for G#{global_id}: {exc}")

    # Fallback: schedule for next minute.
    try:
        target = datetime.now() + timedelta(minutes=1)
        hh, mm = target.hour, target.minute
        print(
            f"  📤 Fallback scheduling WhatsApp for G#{global_id} at "
            f"{hh:02d}:{mm:02d}"
        )
        pywhatkit.sendwhatmsg(
            phone_no=phone_number,
            message=msg,
            time_hour=hh,
            time_min=mm,
            wait_time=20,
            tab_close=True,
            close_time=5,
        )
        print(
            f"  ✅ WhatsApp message scheduled/sent successfully for G#{global_id} "
            f"(fallback)"
        )
        return True
    except Exception as exc:
        print(f"  ❌ WhatsApp fallback send failed for G#{global_id}: {exc}")
        return False


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(hi, v))


def _point_in_polygon(point, polygon):
    if not polygon:
        return False
    pts = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(pts, point, False) >= 0


def _risk_color(score):
    if score >= 70:
        return (0, 0, 255)      # Red
    if score >= 40:
        return (0, 255, 255)    # Yellow
    return (0, 200, 0)          # Green


def _draw_pose_skeleton_on_crop(frame, bbox, pose_result, connections, color=(255, 255, 0)):
    if pose_result is None or pose_result.pose_landmarks is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    landmarks = pose_result.pose_landmarks.landmark
    for a, b in connections:
        if a >= len(landmarks) or b >= len(landmarks):
            continue
        la, lb = landmarks[a], landmarks[b]
        ax, ay = int(x1 + la.x * w), int(y1 + la.y * h)
        bx, by = int(x1 + lb.x * w), int(y1 + lb.y * h)
        cv2.line(frame, (ax, ay), (bx, by), color, 2)
    for lm in landmarks:
        px, py = int(x1 + lm.x * w), int(y1 + lm.y * h)
        cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)


def load_system_config(config_yaml: str = "config/config.yaml") -> dict:
    """Load detector/tracker/reid/system settings from YAML if available."""
    if not os.path.exists(config_yaml):
        return {}
    with open(config_yaml, "r") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def main():
    # ── Configuration ──
    camera_source = 0                         # 0 = default webcam
    camera_id = "cam1"
    
    # Allow command line override: python run_demo.py <source>
    # source can be: 0 (webcam), "video.mp4" (file), "rtsp://..." (stream)
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        try:
            camera_source = int(arg)
        except ValueError:
            camera_source = arg

    cfg = load_system_config()
    det_cfg = cfg.get("detection", {})
    reid_cfg = cfg.get("reid", {})
    trk_cfg = cfg.get("tracker", {})
    sys_cfg = cfg.get("system", {})
    runtime_cfg = cfg.get("runtime", {})
    risk_cfg = cfg.get("risk", {})
    privacy_cfg = cfg.get("privacy", {})
    zones_cfg = cfg.get("zones", {})
    visual_cfg = cfg.get("visual", {})
    device = sys_cfg.get("device", "cuda")

    print("=" * 60)
    print("  MULTI-CAMERA TRACKING — Phase 1 Live Demo")
    print("=" * 60)
    print(f"  Source: {camera_source}")
    print(f"  Controls: Q/ESC=Quit, S=Screenshot")
    print("=" * 60)

    # ── Initialize Models ──
    print("\n[1/3] Loading YOLOv11m...")
    detector = PersonDetector(
        model_path=det_cfg.get("model_path", "models/yolo11m.pt"),
        device=device,
        high_conf=det_cfg.get("high_conf", 0.40),
        low_conf=det_cfg.get("low_conf", 0.10),
        min_area=det_cfg.get("min_area", 400),
    )
    print("       ✓ Detector ready")

    print("[2/3] Loading OSNet_x1_0...")
    encoder = ReIDEncoder(
        model_path=reid_cfg.get("model_path", "models/osnet_x1_0_msmt17.pth"),
        device=device,
    )
    print("       ✓ ReID Encoder ready")

    # ── Open Camera ──
    print(f"[3/3] Opening camera: {camera_source}")
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera source: {camera_source}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    print(f"       ✓ Camera opened: {frame_w}×{frame_h} @ {fps:.0f}fps")

    # ── Initialize Tracker ──
    tracker = SingleCameraTracker(
        camera_id=camera_id,
        frame_w=frame_w,
        frame_h=frame_h,
        n_confirm=trk_cfg.get("n_confirm", 3),
        tentative_max_miss=trk_cfg.get("tentative_max_miss", 3),
        max_miss=trk_cfg.get("max_miss", 20),
        lost_buffer_sec=trk_cfg.get("lost_buffer_sec", 15.0),
        gate_dist_px=trk_cfg.get("gate_dist_px", 200),
        mahal_gate=trk_cfg.get("mahal_gate", 9.48),
        cost_thr_high=trk_cfg.get("cost_thr_high", trk_cfg.get("cost_threshold_high", 0.65)),
        cost_thr_low=trk_cfg.get("cost_thr_low", trk_cfg.get("cost_threshold_low", 0.70)),
        weights_normal=tuple(trk_cfg.get("weights_normal", (0.50, 0.40, 0.10))),
        weights_crowd=tuple(trk_cfg.get("weights_crowd", (0.30, 0.45, 0.25))),
        crowd_overlap_min=trk_cfg.get("crowd_overlap_min", 0.10),
        crowd_score_thr=trk_cfg.get("crowd_score_thr", 0.08),
        sim_thr_live=trk_cfg.get("sim_thr_live", trk_cfg.get("sim_threshold_live", 0.45)),
        sim_thr_lost=trk_cfg.get("sim_thr_lost", trk_cfg.get("sim_threshold_lost", 0.35)),
        embedding_pool_k=trk_cfg.get("embedding_pool_k", 5),
        dup_iou_thr=trk_cfg.get("dup_iou_thr", 0.50),
        merge_sim_thr=trk_cfg.get("merge_sim_thr", 0.60),
        min_export_frames=trk_cfg.get("min_export_frames", 5),
        quality_threshold=trk_cfg.get("quality_threshold", 0.60),
    )
    print("       ✓ Tracker initialized")
    print("       ✓ Output recording: disabled")

    print("\n🎬 Starting tracking... Press Q to quit.\n")

    # ── Metrics ──
    frame_count = 0
    start_time = time.time()
    fps_display = 0.0
    fps_update_interval = 10  # Update FPS display every N frames
    global_db = {}
    next_global_id = 1
    local_to_global = {}
    alerted_global_ids = set()          # for new-person alerts
    risk_alerted_global_ids = set()     # for risk-score alerts (separate)
    whatsapp_number = "+919638793135"
    alert_send_delay_sec = 1.0
    global_match_threshold = 0.62
    global_margin_gate = 0.05
    reentry_window_sec = 120.0
    reid_refresh_every_n_frames = 4
    alerts_enabled = bool(runtime_cfg.get("alerts_enabled", True))

    # Risk engine config
    risk_alert_threshold = 90.0  # Only send suspicious WhatsApp when risk >= 90
    risk_decay_per_sec = float(risk_cfg.get("decay_per_sec", 1.5))
    loiter_seconds = float(risk_cfg.get("loiter_seconds", 10.0))
    loiter_displacement_px = float(risk_cfg.get("loiter_displacement_px", 40.0))
    entry_zone_dwell_sec = float(risk_cfg.get("entry_zone_dwell_sec", 8.0))
    odd_start_hour = int(risk_cfg.get("odd_hours_start", 0))
    odd_end_hour = int(risk_cfg.get("odd_hours_end", 5))
    high_velocity_px_s = float(risk_cfg.get("high_velocity_px_s", 220.0))
    crowd_threshold = int(risk_cfg.get("crowd_threshold", 4))
    frequent_window_sec = float(risk_cfg.get("entry_exit_window_sec", 180.0))
    frequent_count_thr = int(risk_cfg.get("frequent_entry_exit_count", 4))
    retention_sec = float(privacy_cfg.get("retention_hours", 1.0)) * 3600.0
    enable_pose_skeleton = bool(visual_cfg.get("enable_pose_skeleton", True))
    pose_min_conf = float(visual_cfg.get("pose_min_detection_confidence", 0.5))

    zone_polygons = zones_cfg.get("polygons", {
        "entry": [(0, 0), (220, 0), (220, 220), (0, 220)],
        "restricted": [(420, 80), (frame_w - 20, 80), (frame_w - 20, frame_h - 60), (420, frame_h - 60)],
    })

    risk_state = {}          # local_track_id -> state dict
    global_activity = {}     # global_id -> behavior history

    pose_estimator = None
    pose_connections = []
    if enable_pose_skeleton:
        if mp is None or not hasattr(mp, "solutions"):
            print(
                "  ⚠️ MediaPipe Pose API unavailable (missing `mediapipe.solutions`). "
                "Skeleton overlay disabled."
            )
            enable_pose_skeleton = False
        else:
            mp_pose = mp.solutions.pose
            pose_connections = list(mp_pose.POSE_CONNECTIONS)
            pose_estimator = mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=pose_min_conf,
                min_tracking_confidence=0.5,
            )
            print("       ✓ Pose skeleton overlay enabled")

    log_path = os.path.join("logs", "events.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "camera_id", "global_id", "local_track_id", "event", "risk", "details"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(camera_source, str) and not camera_source.startswith("rtsp"):
                    print("\n📹 Video ended.")
                    break
                continue

            frame_count += 1
            timestamp = time.time() - start_time

            # ── Step 1: Detect ──
            high_dets, low_dets = detector.detect(frame)

            # ── Step 2: Extract embeddings for HIGH-conf only ──
            embeddings = {}
            if high_dets:
                crops = []
                valid_indices = []
                for i, det in enumerate(high_dets):
                    x1, y1, x2, y2 = [int(v) for v in det.bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    if x2 > x1 + 2 and y2 > y1 + 2:
                        crop = frame[y1:y2, x1:x2]

                        # ── Crop clipping: reduce ReID contamination ──
                        # Clip overlapping regions to avoid other people's pixels
                        # contaminating this person's embedding.
                        # SAFE: only clip 15% (not 30%), ensure min width, handle ALL occluders.
                        for j, other_det in enumerate(high_dets):
                            if j == i:
                                continue
                            ox1, oy1, ox2, oy2 = other_det.bbox
                            inter_x1 = max(x1, ox1)
                            inter_y1 = max(y1, oy1)
                            inter_x2 = min(x2, ox2)
                            inter_y2 = min(y2, oy2)
                            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                                other_cx = (ox1 + ox2) / 2
                                my_cx = (x1 + x2) / 2
                                cw = crop.shape[1]
                                # Only clip if crop will remain wide enough (≥64px)
                                if cw > 75:
                                    if other_cx > my_cx:
                                        crop = crop[:, :int(cw * 0.85)]
                                    else:
                                        crop = crop[:, int(cw * 0.15):]
                                # NO break — handle all occluders, not just first

                        # Guard: ensure crop is large enough for valid ReID
                        if crop.shape[1] < 32 or crop.shape[0] < 64:
                            # Crop too small after clipping — use original uncropped
                            crop = frame[y1:y2, x1:x2]

                        crops.append(crop)
                        valid_indices.append(i)

                if crops:
                    # Strong-feature mode: extract on original + horizontal-flipped crop,
                    # then fuse embeddings for better robustness to pose/lighting changes.
                    tta_crops = []
                    for crop in crops:
                        tta_crops.append(crop)
                        tta_crops.append(cv2.flip(crop, 1))
                    batch_embs = encoder.batch_extract(tta_crops)
                    if batch_embs is not None:
                        for j, idx in enumerate(valid_indices):
                            a_i = 2 * j
                            b_i = a_i + 1
                            if b_i < len(batch_embs):
                                fused = batch_embs[a_i] + batch_embs[b_i]
                                norm = float(np.linalg.norm(fused))
                                if norm > 1e-8:
                                    embeddings[idx] = (fused / norm).astype(np.float32)
                            elif a_i < len(batch_embs):
                                embeddings[idx] = batch_embs[a_i]

            # ── Step 3: Track ──
            events = tracker.update(high_dets, low_dets, embeddings, timestamp)

            # Log events
            for ev in events:
                if ev.type == "TRACK_ACTIVATED":
                    print(f"  👀 [{ev.camera_id}] Track #{ev.track_id} ACTIVATED")
                    normalized = _normalize_embedding(ev.embedding)
                    if normalized is None:
                        print(f"  ⚠️ [{ev.camera_id}] Track #{ev.track_id}: embedding unavailable")
                        continue
                    # Hard guard: do not assign a GlobalID already used by another
                    # currently active local track.
                    active_track_ids = {t.track_id for t in tracker.get_active_tracks()}
                    blocked_gids = {
                        g for tid, g in local_to_global.items()
                        if tid != ev.track_id and tid in active_track_ids
                    }
                    gid, score, next_global_id, is_new = _match_or_create_identity(
                        normalized,
                        global_db,
                        next_global_id,
                        threshold=global_match_threshold,
                        margin_gate=global_margin_gate,
                        keep_last=12,
                        now_ts=timestamp,
                        reentry_window_sec=reentry_window_sec,
                        recent_boost_window_sec=20.0,
                        recent_threshold=0.58,
                        recent_margin_gate=0.03,
                        blocked_gids=blocked_gids,
                    )
                    local_to_global[ev.track_id] = gid
                    # Push global identity into tracker so drawn bbox labels show G# IDs.
                    tracker.set_global_id(ev.track_id, gid, f"Person #{gid}")
                    action = "NEW PERSON DETECTED" if is_new else "EXISTING PERSON MATCHED"
                    print(
                        f"  🌍 [{ev.camera_id}] Track #{ev.track_id} → Global #{gid} "
                        f"[{action}, sim={score:.2f}]"
                    )
                    # Initialize risk state for this local track.
                    now_wall = time.time()
                    risk_state[ev.track_id] = {
                        "risk": 0.0,
                        "last_update": timestamp,
                        "pos_hist": deque(maxlen=240),
                        "loitering": False,
                        "zone_enter": {},
                        "last_center": None,
                        "vel_hist": deque(maxlen=10),
                        "last_flags": "",
                    }

                    if gid not in global_activity:
                        global_activity[gid] = {
                            "entries": deque(maxlen=50),
                            "first_seen": now_wall,
                            "last_seen": now_wall,
                        }
                    global_activity[gid]["entries"].append(now_wall)
                    global_activity[gid]["last_seen"] = now_wall

                    # New person baseline risk.
                    if is_new and ev.track_id in risk_state:
                        risk_state[ev.track_id]["risk"] = _clamp(risk_state[ev.track_id]["risk"] + 10.0)

                    with open(log_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([
                            datetime.now().isoformat(timespec="seconds"),
                            ev.camera_id,
                            gid,
                            ev.track_id,
                            "TRACK_ACTIVATED",
                            f"{risk_state.get(ev.track_id, {}).get('risk', 0.0):.1f}",
                            action,
                        ])
                elif ev.type == "TRACK_ENDED":
                    duration = ev.exit_time - ev.entry_time
                    gid = local_to_global.get(ev.track_id)
                    if gid is not None and ev.pooled_embedding is not None:
                        norm_pool = _normalize_embedding(ev.pooled_embedding)
                        if norm_pool is not None and gid in global_db:
                            embs = global_db[gid].setdefault("embeddings", [])
                            embs.append(norm_pool.copy())
                            if len(embs) > 12:
                                del embs[:-12]
                            global_db[gid]["last_seen"] = timestamp
                    print(f"  ⏹  Track #{ev.track_id} ENDED on {ev.camera_id} "
                          f"(G#{gid if gid is not None else 'N/A'}, duration: {duration:.1f}s, quality: {ev.quality:.2f})")
                    local_to_global.pop(ev.track_id, None)
                    risk_state.pop(ev.track_id, None)

            # Periodically refresh global identity prototypes from active tracks.
            if frame_count % reid_refresh_every_n_frames == 0:
                for t in tracker.get_active_tracks():
                    gid = local_to_global.get(t.track_id)
                    if gid is None or t.pooled_embedding is None or gid not in global_db:
                        continue
                    norm_pool = _normalize_embedding(t.pooled_embedding)
                    if norm_pool is None:
                        continue
                    embs = global_db[gid].setdefault("embeddings", [])
                    embs.append(norm_pool.copy())
                    if len(embs) > 12:
                        del embs[:-12]
                    global_db[gid]["last_seen"] = timestamp

            # Risk engine update for each active track.
            active_tracks = tracker.get_active_tracks()
            people_count = len(active_tracks)
            for t in active_tracks:
                tid = t.track_id
                gid = local_to_global.get(tid)
                if gid is None:
                    continue
                st = risk_state.setdefault(tid, {
                    "risk": 0.0,
                    "last_update": timestamp,
                    "pos_hist": deque(maxlen=240),
                    "loitering": False,
                    "zone_enter": {},
                    "last_center": None,
                    "vel_hist": deque(maxlen=10),
                    "last_flags": "",
                })

                dt = max(1e-3, float(timestamp - st["last_update"]))
                st["last_update"] = timestamp
                st["risk"] = _clamp(st["risk"] - risk_decay_per_sec * dt)

                x1, y1, x2, y2 = t.bbox
                cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)
                st["pos_hist"].append((timestamp, cx, cy))
                flags = []

                # Loitering
                if len(st["pos_hist"]) >= 2:
                    t0, x0, y0 = st["pos_hist"][0]
                    disp = ((cx - x0) ** 2 + (cy - y0) ** 2) ** 0.5
                    if (timestamp - t0) >= loiter_seconds and disp < loiter_displacement_px:
                        if not st["loitering"]:
                            st["risk"] = _clamp(st["risk"] + 15.0)
                            st["loitering"] = True
                        flags.append("LOITERING")
                    else:
                        st["loitering"] = False

                # Zones
                in_restricted = False
                in_entry = False
                for zname, poly in zone_polygons.items():
                    inside = _point_in_polygon((cx, cy), poly)
                    enter_t = st["zone_enter"].get(zname)
                    if inside and enter_t is None:
                        st["zone_enter"][zname] = timestamp
                        enter_t = timestamp
                    if not inside and enter_t is not None:
                        st["zone_enter"].pop(zname, None)
                    if inside and zname.lower() == "restricted":
                        in_restricted = True
                        # Restricted area risk only once each 5 seconds.
                        if int(timestamp - enter_t) in (0, 5):
                            st["risk"] = _clamp(st["risk"] + 25.0)
                        flags.append("RESTRICTED")
                    if inside and zname.lower() == "entry":
                        in_entry = True
                        if (timestamp - enter_t) > entry_zone_dwell_sec:
                            st["risk"] = _clamp(st["risk"] + 8.0)
                            flags.append("ENTRY-DWELL")

                # Odd hours
                hr = datetime.now().hour
                if odd_start_hour <= hr < odd_end_hour:
                    st["risk"] = _clamp(st["risk"] + 20.0 * dt / 5.0)
                    flags.append("ODD-HOUR")
                    if in_restricted:
                        st["risk"] = _clamp(st["risk"] + 5.0)

                # Motion anomaly
                if st["last_center"] is not None:
                    px, py = st["last_center"]
                    vel = (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5) / dt
                    st["vel_hist"].append(vel)
                    sm_vel = float(np.mean(st["vel_hist"])) if st["vel_hist"] else vel
                    if sm_vel > high_velocity_px_s:
                        st["risk"] = _clamp(st["risk"] + 10.0)
                        flags.append("FAST")
                st["last_center"] = (cx, cy)

                # Frequent entry/exit pattern
                acts = global_activity.get(gid, {})
                entries = acts.get("entries", deque())
                recent = [t0 for t0 in entries if (time.time() - t0) <= frequent_window_sec]
                if len(recent) >= frequent_count_thr:
                    st["risk"] = _clamp(st["risk"] + 10.0)
                    flags.append("FREQ-ENTRY")

                # Crowd density
                if people_count > crowd_threshold:
                    st["risk"] = _clamp(st["risk"] + 5.0)
                    flags.append("CROWD")

                # Familiar person baseline decrease
                if len(recent) > 6:
                    st["risk"] = _clamp(st["risk"] - 4.0)

                st["last_flags"] = "|".join(sorted(set(flags))) if flags else "NORMAL"

                # Alert once per global_id when risk crosses 90 (suspicious activity).
                if alerts_enabled and st["risk"] >= risk_alert_threshold and gid not in risk_alerted_global_ids:
                    time.sleep(alert_send_delay_sec)
                    ts_text = time.strftime("%Y-%m-%d %H:%M:%S")
                    susp_msg = (
                        f"🚨 [SUSPICIOUS ACTIVITY ALERT]\n"
                        f"Suspicious activity from Person #{gid}\n"
                        f"Risk Score: {st['risk']:.1f}/100\n"
                        f"Flags: {st['last_flags']}\n"
                        f"Camera: {camera_id}\n"
                        f"Time: {ts_text}"
                    )
                    sent = send_whatsapp_message(
                        global_id=gid,
                        camera_name=camera_id,
                        event_time=ts_text,
                        phone_number=whatsapp_number,
                        message=susp_msg,
                    )
                    if sent:
                        risk_alerted_global_ids.add(gid)
                        with open(log_path, "a", newline="", encoding="utf-8") as f:
                            csv.writer(f).writerow([
                                datetime.now().isoformat(timespec="seconds"),
                                camera_id,
                                gid,
                                tid,
                                "RISK_ALERT",
                                f"{st['risk']:.1f}",
                                st["last_flags"],
                            ])

            # Privacy retention: drop very old global identities.
            if retention_sec > 0:
                now_wall = time.time()
                stale = [gid for gid, g in global_db.items() if (now_wall - float(g.get("last_seen", now_wall))) > retention_sec]
                for gid in stale:
                    global_db.pop(gid, None)

            # ── Step 4: Draw ──
            annotated = tracker.draw_on_frame(frame)
            active_tracks = tracker.get_active_tracks()

            # Optional: per-person body skeleton overlay (MediaPipe Pose).
            if enable_pose_skeleton and pose_estimator is not None:
                for t in active_tracks:
                    x1, y1, x2, y2 = [int(v) for v in t.bbox]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    if x2 <= x1 + 20 or y2 <= y1 + 40:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_res = pose_estimator.process(rgb_crop)
                    _draw_pose_skeleton_on_crop(
                        annotated, (x1, y1, x2, y2), pose_res, pose_connections
                    )

            # Draw configured zones.
            for zname, poly in zone_polygons.items():
                if not poly:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                zone_color = (255, 0, 255) if zname.lower() == "restricted" else (255, 255, 0)
                cv2.polylines(annotated, [pts], True, zone_color, 2)
                zx, zy = poly[0]
                cv2.putText(annotated, zname.upper(), (int(zx), int(zy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 2)

            # Risk overlay per track: color-coded bbox + GID|Risk|Status.
            for t in active_tracks:
                tid = t.track_id
                gid = local_to_global.get(tid)
                if gid is None:
                    continue
                st = risk_state.get(tid, {})
                risk_val = float(st.get("risk", 0.0))
                status = st.get("last_flags", "NORMAL")
                color = _risk_color(risk_val)
                x1, y1, x2, y2 = [int(v) for v in t.bbox]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"G{gid} | R:{risk_val:.0f} | {status}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Draw HUD
            if frame_count % fps_update_interval == 0:
                elapsed = time.time() - start_time
                fps_display = frame_count / elapsed if elapsed > 0 else 0

            hud_lines = [
                f"FPS: {fps_display:.1f}",
                f"Frame: {frame_count}",
                f"Persons: {len(active_tracks)}",
                f"Detections: H={len(high_dets)} L={len(low_dets)}",
                f"Lost: {len(tracker.lost_buffer)}",
                f"CrowdAlert: {'ON' if len(active_tracks) > crowd_threshold else 'OFF'}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(annotated, line, (10, 25 + i * 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            # Show
            cv2.imshow("Multi-Camera Tracking - Phase 1 Demo", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # Q or ESC
                break
            elif key in (ord('s'), ord('S')):
                fname = f"screenshot_{frame_count}.jpg"
                cv2.imwrite(fname, annotated)
                print(f"  📸 Screenshot saved: {fname}")

    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")

    # ── Cleanup ──
    final_events = tracker.finalize_all(time.time() - start_time)
    for ev in final_events:
        if ev.type == "TRACK_ENDED":
            duration = ev.exit_time - ev.entry_time
            print(f"  ⏹  Track #{ev.track_id} finalized "
                  f"(duration: {duration:.1f}s, quality: {ev.quality:.2f})")

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Session Summary")
    print(f"{'=' * 60}")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Frames:       {frame_count}")
    print(f"  Avg FPS:      {frame_count / elapsed:.1f}" if elapsed > 0 else "  Avg FPS: N/A")
    print(f"  Total tracks: {tracker._next_id}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
