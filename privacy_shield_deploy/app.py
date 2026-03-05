"""
Privacy Shield — Flask Web App  (multi-object fix)
====================================================
Multi-object bugs fixed:
  1. classes= filter  → pass only our 9 target COCO IDs to YOLO.
                        Without this, NMS across all 80 classes lets
                        irrelevant high-confidence objects (chairs, tables)
                        suppress nearby target detections via IoU overlap.
  2. iou=0.45         → default 0.7 is too aggressive; nearby objects
                        (phone next to keyboard, two people) got merged
                        into one box and the weaker one was dropped.
  3. max_det=100      → explicit cap; nano model can handle 100 objects.
  4. conf=0.25        → 0.35 missed secondary / partially occluded objects.
  5. Separate render  → stream now reads the latest RAW frame and reapplies
                        the last-known boxes every frame at 30fps.
                        Previously the stream was gated by YOLO speed (~7fps)
                        so fast movements made it look like detection dropped.
  6. main_idx set     → all secondary persons (not just index==main_idx)
                        are blurred correctly.

Run:  python app.py
Open: http://localhost:5000
"""

import cv2
import numpy as np
import threading
import queue
import time
import sys
from flask import Flask, render_template, Response, jsonify, request

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics flask opencv-python")
    sys.exit(1)

app = Flask(__name__)

# ── COCO categories we care about ─────────────────────────────────────────────
CATEGORY_COCO_IDS = {
    "Background People":  [0],
    "Phones & Screens":   [63, 67],
    "Documents & Books":  [73],
    "Keyboards & Mice":   [64, 66],
    "Background Objects": [77, 39, 74],
}
CATEGORY_COLORS_BGR = {
    "Background People":  (60,  60, 255),
    "Phones & Screens":   (20, 150, 255),
    "Documents & Books":  (50, 200,  50),
    "Keyboards & Mice":   (210, 140,  40),
    "Background Objects": (170,  70, 150),
}

# FIX 1: flat list of every class ID we want YOLO to detect.
# Passing this to model() restricts NMS to ONLY these classes so they
# never compete with/suppress each other via unrelated class IoU.
ALL_TARGET_IDS = sorted({cid for ids in CATEGORY_COCO_IDS.values() for cid in ids})
# → [0, 39, 63, 64, 66, 67, 73, 74, 77]

# Reverse lookup: cls_id → category name  (built once at startup)
CLS_TO_CAT = {}
for cat, ids in CATEGORY_COCO_IDS.items():
    for cid in ids:
        CLS_TO_CAT[cid] = cat


# ── Shared config ─────────────────────────────────────────────────────────────
class Config:
    lock             = threading.Lock()
    shield_active    = True
    blur_strength    = 51
    confidence       = 0.25          # FIX 4: lowered from 0.35 → catch more objects
    show_boxes       = True
    protect_main     = True
    category_enabled = {k: True for k in CATEGORY_COCO_IDS}
    fps              = 0.0
    detections       = {k: 0 for k in CATEGORY_COCO_IDS}

cfg = Config()

# ── Shared latest frames & boxes ──────────────────────────────────────────────
_state_lock    = threading.Lock()
latest_raw     = None          # latest BGR frame from camera
latest_boxes   = []            # list of dicts written by inference thread
latest_counts  = {k: 0 for k in CATEGORY_COCO_IDS}

# raw_q feeds the inference thread (maxsize=1 → always latest, no backlog)
raw_q = queue.Queue(maxsize=1)

def q_put_latest(q, item):
    try:   q.get_nowait()
    except queue.Empty: pass
    try:   q.put_nowait(item)
    except queue.Full:  pass


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 1 — Capture
# ══════════════════════════════════════════════════════════════════════════════
def capture_thread():
    global latest_raw
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[Capture] ERROR: cannot open webcam")
        return

    print("[Capture] Camera open")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1)
        with _state_lock:
            latest_raw = frame
        q_put_latest(raw_q, frame)   # non-blocking; old frame is dropped

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 2 — Inference
# ══════════════════════════════════════════════════════════════════════════════
def inference_thread():
    global latest_boxes, latest_counts

    print("[Inference] Loading YOLOv8n ...")
    model = YOLO("yolov8n.pt")
    print("[Inference] Ready")

    t_prev, frame_n = time.time(), 0

    while True:
        try:
            frame = raw_q.get(timeout=2.0)
        except queue.Empty:
            continue

        h, w = frame.shape[:2]

        with cfg.lock:
            shield  = cfg.shield_active
            conf_th = cfg.confidence
            cat_en  = dict(cfg.category_enabled)
            prot_mp = cfg.protect_main

        if not shield:
            with _state_lock:
                latest_boxes  = []
                latest_counts = {k: 0 for k in CATEGORY_COCO_IDS}
            continue

        # ── YOLO inference ────────────────────────────────────────────────────
        # FIX 1: classes=ALL_TARGET_IDS  — only detect our 9 target classes
        # FIX 2: iou=0.45                — less aggressive NMS, keep nearby objects
        # FIX 3: max_det=100             — explicit; default 300 is fine but explicit
        results = model(
            frame,
            classes   = ALL_TARGET_IDS,   # ← KEY FIX: restricts NMS to target classes
            conf      = conf_th,
            iou       = 0.45,              # ← KEY FIX: was 0.7, now keeps nearby objects
            max_det   = 100,
            verbose   = False,
        )[0]

        parsed     = []
        det_counts = {k: 0 for k in CATEGORY_COCO_IDS}

        if results.boxes is not None and len(results.boxes) > 0:
            boxes_np  = results.boxes.xyxy.cpu().numpy()
            clsids_np = results.boxes.cls.cpu().numpy().astype(int)
            confs_np  = results.boxes.conf.cpu().numpy()

            # ── Find main person: the LARGEST person box = you ───────────────
            # FIX 6: collect ALL person indices; only skip the single biggest one
            main_person_idx = None
            if prot_mp:
                p_indices = [i for i, c in enumerate(clsids_np) if c == 0]
                if p_indices:
                    areas = [
                        (boxes_np[i][2] - boxes_np[i][0]) *
                        (boxes_np[i][3] - boxes_np[i][1])
                        for i in p_indices
                    ]
                    main_person_idx = p_indices[int(np.argmax(areas))]
                    # All other persons WILL be blurred (they're background people)

            # ── Process every detection independently ─────────────────────────
            for i in range(len(boxes_np)):
                cls_id   = clsids_np[i]
                conf_val = float(confs_np[i])

                # Skip YOU (the main/largest person)
                if cls_id == 0 and i == main_person_idx:
                    continue

                cat = CLS_TO_CAT.get(cls_id)          # O(1) lookup
                if cat is None:
                    continue
                if not cat_en.get(cat, True):
                    continue

                x1 = max(0,     int(boxes_np[i][0]))
                y1 = max(0,     int(boxes_np[i][1]))
                x2 = min(w - 1, int(boxes_np[i][2]))
                y2 = min(h - 1, int(boxes_np[i][3]))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue

                det_counts[cat] += 1
                parsed.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                                   cat=cat, conf=conf_val))

        # Atomically publish boxes for the render thread
        with _state_lock:
            latest_boxes  = parsed
            latest_counts = det_counts

        # FPS counter (inference speed)
        frame_n += 1
        now = time.time()
        if now - t_prev >= 1.0:
            with cfg.lock:
                cfg.fps        = round(frame_n / (now - t_prev), 1)
                cfg.detections = det_counts
            frame_n, t_prev = 0, now


# ══════════════════════════════════════════════════════════════════════════════
#  Blur helper
# ══════════════════════════════════════════════════════════════════════════════
def apply_blur(frame: np.ndarray, x1, y1, x2, y2, k: int):
    """
    Pixelated-Gaussian blur in-place.
    .copy() is mandatory — without it, GaussianBlur on a numpy view
    corrupts the source frame via aliasing.
    """
    roi = frame[y1:y2, x1:x2].copy()   # ← explicit copy, not a view
    if roi.size == 0:
        return
    rh, rw = roi.shape[:2]

    blurred = cv2.GaussianBlur(roi, (k, k), 0)

    # Pixelate: shrink then expand with NEAREST → hard block edges
    scale = max(2, min(rw, rh) // 12)
    sw, sh = max(1, rw // scale), max(1, rh // scale)
    small  = cv2.resize(blurred, (sw, sh), interpolation=cv2.INTER_AREA)
    pixel  = cv2.resize(small,   (rw, rh), interpolation=cv2.INTER_NEAREST)

    frame[y1:y2, x1:x2] = pixel


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 5 — MJPEG generator reads latest RAW frame + reapplies last known boxes
#  Runs at 30 fps regardless of YOLO speed.
#  Result: smooth motion + blur tracks objects even between YOLO runs.
# ══════════════════════════════════════════════════════════════════════════════
def generate_mjpeg():
    TARGET_FPS  = 30
    FRAME_DELAY = 1.0 / TARGET_FPS

    while True:
        t0 = time.time()

        with _state_lock:
            raw    = latest_raw
            boxes  = list(latest_boxes)   # snapshot

        with cfg.lock:
            shield = cfg.shield_active
            blur_k = max(5, cfg.blur_strength) | 1
            show_b = cfg.show_boxes

        if raw is None:
            time.sleep(0.02)
            continue

        out = raw.copy()   # never mutate the shared raw frame

        if shield and boxes:
            for box in boxes:
                apply_blur(out, box["x1"], box["y1"], box["x2"], box["y2"], blur_k)

                if show_b:
                    color = CATEGORY_COLORS_BGR.get(box["cat"], (200, 200, 200))
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

                    # Bounding box
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

                    # Label background + text
                    label = f"{box['cat']}  {box['conf']:.0%}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    ly1 = max(0, y1 - th - 10)
                    lx2 = min(out.shape[1], x1 + tw + 8)
                    cv2.rectangle(out, (x1, ly1), (lx2, y1), color, cv2.FILLED)
                    cv2.putText(out, label, (x1 + 3, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 0), 1, cv2.LINE_AA)

        # HUD badge
        txt = "SHIELD ON" if shield else "SHIELD OFF"
        col = (30, 160, 30) if shield else (30, 30, 200)
        cv2.rectangle(out, (10, 10), (165, 40), col, cv2.FILLED)
        cv2.putText(out, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", out,
                               [cv2.IMWRITE_JPEG_QUALITY, 82,
                                cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")

        # Pace to 30 fps
        elapsed = time.time() - t0
        rest    = FRAME_DELAY - elapsed
        if rest > 0:
            time.sleep(rest)


# ══════════════════════════════════════════════════════════════════════════════
#  Flask routes
# ══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                 "Pragma": "no-cache", "Expires": "0"},
    )

@app.route("/stats")
def stats():
    with cfg.lock:
        return jsonify(
            fps        = cfg.fps,
            shield     = cfg.shield_active,
            detections = cfg.detections,
            total      = sum(cfg.detections.values()),
        )

@app.route("/config", methods=["POST"])
def update_config():
    data = request.json or {}
    with cfg.lock:
        if "shield"       in data: cfg.shield_active = bool(data["shield"])
        if "blur"         in data: cfg.blur_strength  = max(5, int(data["blur"]))
        if "confidence"   in data: cfg.confidence     = float(data["confidence"])
        if "show_boxes"   in data: cfg.show_boxes     = bool(data["show_boxes"])
        if "protect_main" in data: cfg.protect_main   = bool(data["protect_main"])
        if "category"     in data:
            cat, val = data["category"]["name"], data["category"]["enabled"]
            if cat in cfg.category_enabled:
                cfg.category_enabled[cat] = bool(val)
    return jsonify(ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import webbrowser

    print("""
╔══════════════════════════════════════════════════════════╗
║  🛡️  Privacy Shield  — Multi-Object Fix                 ║
╠══════════════════════════════════════════════════════════╣
║  Fix 1: classes=ALL_TARGET_IDS  (no cross-class NMS)    ║
║  Fix 2: iou=0.45                (nearby objects kept)   ║
║  Fix 3: conf=0.25               (catch secondary objs)  ║
║  Fix 4: separate render thread  (30fps regardless YOLO) ║
║  Fix 5: main_idx only skips YOU (all others blurred)    ║
╚══════════════════════════════════════════════════════════╝
""")

    threading.Thread(target=capture_thread,   daemon=True, name="Capture").start()
    threading.Thread(target=inference_thread, daemon=True, name="Inference").start()

    time.sleep(1.5)   # let camera warm up

    url = "http://localhost:5000"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"[Flask] → {url}\n")

    app.run(host="0.0.0.0", port=5000, debug=False,
            threaded=True, use_reloader=False)