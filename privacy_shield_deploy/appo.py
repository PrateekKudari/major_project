"""
Privacy Shield — with Card/ID Detection
=========================================
Card detection added via OpenCV geometry (YOLO has no card class in COCO):
  - Canny edge → find contours → approxPolyDP → 4-sided shapes
  - Aspect ratio filter: 1.4 – 1.8  (credit card = 85.6×53.98 = 1.586)
  - Area filter: removes tiny noise and huge false positives
  - Duplicate suppression via IoU so overlapping boxes collapse
  - Runs alongside YOLO so all other categories still work
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

# ── COCO categories (YOLO handles these) ──────────────────────────────────────
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
    "Cards & IDs":        (0,   220, 255),   # cyan for cards
}

ALL_TARGET_IDS = sorted({cid for ids in CATEGORY_COCO_IDS.values() for cid in ids})
CLS_TO_CAT     = {cid: cat for cat, ids in CATEGORY_COCO_IDS.items() for cid in ids}

# ── Config ────────────────────────────────────────────────────────────────────
class Config:
    lock             = threading.Lock()
    shield_active    = True
    blur_strength    = 51
    confidence       = 0.25
    show_boxes       = True
    protect_main     = True
    detect_cards     = True          # ← NEW: card detection toggle
    fps              = 0.0
    category_enabled = {
        **{k: True for k in CATEGORY_COCO_IDS},
        "Cards & IDs": True,         # ← NEW category
    }
    detections = {
        **{k: 0 for k in CATEGORY_COCO_IDS},
        "Cards & IDs": 0,
    }

cfg = Config()

# ── Shared state ──────────────────────────────────────────────────────────────
_state_lock   = threading.Lock()
latest_raw    = None
latest_boxes  = []
latest_counts = {**{k: 0 for k in CATEGORY_COCO_IDS}, "Cards & IDs": 0}

raw_q = queue.Queue(maxsize=1)

def q_put_latest(q, item):
    try:   q.get_nowait()
    except queue.Empty: pass
    try:   q.put_nowait(item)
    except queue.Full:  pass


# ══════════════════════════════════════════════════════════════════════════════
#  CARD DETECTOR  (OpenCV geometry — no ML needed)
#
#  Why this works:
#    Credit cards / ID badges / bank cards are ALWAYS:
#      • Rectangular (4 straight edges, ~90° corners)
#      • Aspect ratio 1.4 – 1.8  (credit card = 1.586, ID badge ≈ 1.4–1.6)
#      • A certain physical size relative to the frame
#
#  Pipeline:
#    1. Convert to grayscale + blur (removes noise)
#    2. Adaptive threshold (handles varying lighting conditions)
#    3. Canny edge detection
#    4. Find external contours
#    5. Approximate each contour to a polygon
#    6. Keep only 4-sided polygons (rectangles)
#    7. Filter by aspect ratio + area
#    8. Suppress duplicates with IoU
# ══════════════════════════════════════════════════════════════════════════════

# Card aspect ratio range (width / height when card is landscape)
CARD_RATIO_MIN = 1.35
CARD_RATIO_MAX = 1.85

# Area as fraction of total frame area
CARD_AREA_MIN_FRAC = 0.003   # at least 0.3% of frame (no tiny glints)
CARD_AREA_MAX_FRAC = 0.30    # at most 30% of frame (not the whole scene)

# IoU threshold for duplicate suppression
CARD_IOU_THRESH = 0.40


def _iou(a, b):
    """Intersection over Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def detect_cards(frame: np.ndarray) -> list[dict]:
    """
    Return list of card boxes: [{x1,y1,x2,y2,cat,conf}, ...]
    conf is a synthetic 0-1 score based on how close the ratio is to 1.586.
    """
    h, w   = frame.shape[:2]
    frame_area = w * h

    # ── 1. Pre-process ─────────────────────────────────────────────────────
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Bilateral filter: smooths flat regions but keeps edges sharp
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)

    # ── 2. Multi-scale edge detection ──────────────────────────────────────
    # Run two Canny passes (tight + loose) and OR them together.
    # This catches both high-contrast and low-contrast card edges.
    edges_tight = cv2.Canny(smooth, 50, 150)
    edges_loose = cv2.Canny(smooth, 20,  80)
    edges       = cv2.bitwise_or(edges_tight, edges_loose)

    # Also try adaptive threshold as a second edge source
    thresh = cv2.adaptiveThreshold(
        smooth, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    thresh_edges = cv2.Canny(thresh, 30, 90)
    edges = cv2.bitwise_or(edges, thresh_edges)

    # Dilate to close small gaps in card border
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    # ── 3. Find contours ───────────────────────────────────────────────────
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_frac = area / frame_area

        # Quick area pre-filter before expensive polygon approximation
        if area_frac < CARD_AREA_MIN_FRAC or area_frac > CARD_AREA_MAX_FRAC:
            continue

        # ── 4. Approximate to polygon ──────────────────────────────────────
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

        # Must be a quadrilateral (4 vertices)
        if len(approx) != 4:
            continue

        # ── 5. Must be convex ──────────────────────────────────────────────
        if not cv2.isContourConvex(approx):
            continue

        # ── 6. Aspect ratio check ──────────────────────────────────────────
        # Use bounding rect (works for slightly rotated cards too)
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0:
            continue

        ratio = max(bw, bh) / min(bw, bh)   # always ≥ 1

        if ratio < CARD_RATIO_MIN or ratio > CARD_RATIO_MAX:
            continue

        # ── 7. Corner angle check ─────────────────────────────────────────
        # All 4 angles should be close to 90°
        pts = approx.reshape(4, 2).astype(np.float32)
        # Sort: top-left, top-right, bottom-right, bottom-left
        s = pts.sum(axis=1);  d = np.diff(pts, axis=1)
        rect_pts = np.array([
            pts[np.argmin(s)],   pts[np.argmin(d)],
            pts[np.argmax(s)],   pts[np.argmax(d)],
        ], dtype=np.float32)

        angles_ok = True
        for i in range(4):
            p0 = rect_pts[i]
            p1 = rect_pts[(i-1) % 4]
            p2 = rect_pts[(i+1) % 4]
            v1 = p1 - p0;  v2 = p2 - p0
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm == 0:
                angles_ok = False
                break
            cos_a = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_a))
            if not (60 < angle < 120):   # allow ±30° from 90
                angles_ok = False
                break

        if not angles_ok:
            continue

        # ── 8. Box coordinates (clamped) ──────────────────────────────────
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w-1, x + bw)
        y2 = min(h-1, y + bh)

        # Synthetic confidence: how close ratio is to ideal 1.586
        ideal_ratio = 1.586
        conf = max(0.0, 1.0 - abs(ratio - ideal_ratio) / ideal_ratio)

        candidates.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                               cat="Cards & IDs", conf=round(conf, 2),
                               area_frac=area_frac))

    # ── 9. NMS / duplicate suppression ────────────────────────────────────
    # Sort by confidence descending, suppress overlapping boxes
    candidates.sort(key=lambda c: c["conf"], reverse=True)
    kept = []
    for cand in candidates:
        box = [cand["x1"], cand["y1"], cand["x2"], cand["y2"]]
        suppress = False
        for k in kept:
            if _iou(box, [k["x1"], k["y1"], k["x2"], k["y2"]]) > CARD_IOU_THRESH:
                suppress = True
                break
        if not suppress:
            kept.append(cand)

    return kept


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
        q_put_latest(raw_q, frame)

    cap.release()


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 2 — Inference  (YOLO + Card detector)
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
            shield      = cfg.shield_active
            conf_th     = cfg.confidence
            cat_en      = dict(cfg.category_enabled)
            prot_mp     = cfg.protect_main
            do_cards    = cfg.detect_cards

        if not shield:
            with _state_lock:
                latest_boxes  = []
                latest_counts = {k: 0 for k in cfg.detections}
            continue

        parsed     = []
        det_counts = {k: 0 for k in cfg.detections}

        # ── A) YOLO detections ─────────────────────────────────────────────
        results = model(
            frame,
            classes = ALL_TARGET_IDS,
            conf    = conf_th,
            iou     = 0.45,
            max_det = 100,
            verbose = False,
        )[0]

        if results.boxes is not None and len(results.boxes) > 0:
            boxes_np  = results.boxes.xyxy.cpu().numpy()
            clsids_np = results.boxes.cls.cpu().numpy().astype(int)
            confs_np  = results.boxes.conf.cpu().numpy()

            # Largest person = YOU → skip
            main_person_idx = None
            if prot_mp:
                p_idx = [i for i, c in enumerate(clsids_np) if c == 0]
                if p_idx:
                    areas = [(boxes_np[i][2]-boxes_np[i][0]) *
                             (boxes_np[i][3]-boxes_np[i][1]) for i in p_idx]
                    main_person_idx = p_idx[int(np.argmax(areas))]

            for i in range(len(boxes_np)):
                cls_id = clsids_np[i]
                if cls_id == 0 and i == main_person_idx:
                    continue
                cat = CLS_TO_CAT.get(cls_id)
                if not cat or not cat_en.get(cat, True):
                    continue
                x1 = max(0,     int(boxes_np[i][0]))
                y1 = max(0,     int(boxes_np[i][1]))
                x2 = min(w - 1, int(boxes_np[i][2]))
                y2 = min(h - 1, int(boxes_np[i][3]))
                if x2 - x1 < 4 or y2 - y1 < 4:
                    continue
                det_counts[cat] += 1
                parsed.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                                   cat=cat, conf=float(confs_np[i])))

        # ── B) Card / ID badge detector ────────────────────────────────────
        if do_cards and cat_en.get("Cards & IDs", True):
            card_boxes = detect_cards(frame)
            for cb in card_boxes:
                det_counts["Cards & IDs"] += 1
                parsed.append(cb)

        # Publish
        with _state_lock:
            latest_boxes  = parsed
            latest_counts = det_counts

        # FPS
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
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return
    rh, rw = roi.shape[:2]
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    scale   = max(2, min(rw, rh) // 12)
    sw, sh  = max(1, rw // scale), max(1, rh // scale)
    small   = cv2.resize(blurred, (sw, sh), interpolation=cv2.INTER_AREA)
    pixel   = cv2.resize(small,   (rw, rh), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixel


# ══════════════════════════════════════════════════════════════════════════════
#  MJPEG generator
# ══════════════════════════════════════════════════════════════════════════════
def generate_mjpeg():
    TARGET_FPS  = 30
    FRAME_DELAY = 1.0 / TARGET_FPS

    while True:
        t0 = time.time()

        with _state_lock:
            raw   = latest_raw
            boxes = list(latest_boxes)

        with cfg.lock:
            shield = cfg.shield_active
            blur_k = max(5, cfg.blur_strength) | 1
            show_b = cfg.show_boxes

        if raw is None:
            time.sleep(0.02)
            continue

        out = raw.copy()

        if shield and boxes:
            for box in boxes:
                apply_blur(out, box["x1"], box["y1"], box["x2"], box["y2"], blur_k)
                if show_b:
                    color = CATEGORY_COLORS_BGR.get(box["cat"], (200, 200, 200))
                    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                    label = f"{box['cat']}  {box['conf']:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    ly1 = max(0, y1 - th - 10)
                    lx2 = min(out.shape[1], x1 + tw + 8)
                    cv2.rectangle(out, (x1, ly1), (lx2, y1), color, cv2.FILLED)
                    cv2.putText(out, label, (x1+3, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 0, 0), 1, cv2.LINE_AA)

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
        if "detect_cards" in data: cfg.detect_cards   = bool(data["detect_cards"])
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
╔═══════════════════════════════════════════════════════════╗
║  🛡️  Privacy Shield  — Card + Multi-Object Detection     ║
╠═══════════════════════════════════════════════════════════╣
║  YOLO  → People, Phones, Docs, Keyboards                 ║
║  OpenCV geometry → Credit cards, ID badges, Bank cards   ║
║                    (aspect ratio 1.4–1.8, 4-sided shape) ║
╚═══════════════════════════════════════════════════════════╝
""")

    threading.Thread(target=capture_thread,   daemon=True, name="Capture").start()
    threading.Thread(target=inference_thread, daemon=True, name="Inference").start()

    time.sleep(1.5)

    url = "http://localhost:5000"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"[Flask] → {url}\n")

    app.run(host="0.0.0.0", port=5000, debug=False,
            threaded=True, use_reloader=False)
