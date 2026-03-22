"""
Privacy Shield — Fixed for Windows MSMF error
===============================================
Fix 1: MSMF error -1072875772
  → Use DirectShow (CAP_DSHOW) backend on Windows instead of MSMF.
  → Try camera indices 0, 1, 2 automatically.

Fix 2: Internal Server Error on /video_feed
  → generate_mjpeg() now yields a placeholder frame while camera
    is starting rather than crashing Flask with a bare exception.

Fix 3: Camera locked by another app
  → Clear error message + auto-retry every 5 seconds.
"""

import cv2
import numpy as np
import threading
import queue
import time
import sys
import os
from flask import Flask, render_template, Response, jsonify, request

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] pip install ultralytics flask opencv-python Pillow")
    sys.exit(1)

try:
    from flask_socketio import SocketIO, emit as ws_emit
    HAS_SOCKETIO = True
except ImportError:
    HAS_SOCKETIO = False

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["SECRET_KEY"] = "privacy-shield-2024"

if HAS_SOCKETIO:
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                        max_http_buffer_size=10 * 1024 * 1024)

# ── COCO categories ────────────────────────────────────────────────────────────
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
    "Cards & IDs":        (0,   220, 255),
}
ALL_TARGET_IDS = sorted({cid for ids in CATEGORY_COCO_IDS.values() for cid in ids})
CLS_TO_CAT     = {cid: cat for cat, ids in CATEGORY_COCO_IDS.items() for cid in ids}

# ── Config ─────────────────────────────────────────────────────────────────────
class Config:
    lock             = threading.Lock()
    shield_active    = True
    blur_strength    = 51
    confidence       = 0.25
    show_boxes       = True
    protect_main     = True
    detect_cards     = True
    fps              = 0.0
    camera_ok        = False   # NEW: track camera state for UI
    category_enabled = {
        **{k: True for k in CATEGORY_COCO_IDS},
        "Cards & IDs": True,
    }
    detections = {**{k: 0 for k in CATEGORY_COCO_IDS}, "Cards & IDs": 0}

cfg = Config()

# ── Shared state ────────────────────────────────────────────────────────────────
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
#  FIX 1 — Camera opener with DirectShow on Windows
# ══════════════════════════════════════════════════════════════════════════════
def _open_camera():
    """
    Try backends in order until one works.
    Windows MSMF gives -1072875772 on many laptops.
    DirectShow (CAP_DSHOW) is far more reliable on Windows.
    """
    if sys.platform == "win32":
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),   # most reliable on Windows
            (cv2.CAP_MSMF,  "MSMF"),
            (cv2.CAP_ANY,   "Auto"),
        ]
    elif sys.platform == "darwin":
        backends = [
            (cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (cv2.CAP_ANY,          "Auto"),
        ]
    else:
        backends = [
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY,  "Auto"),
        ]

    for idx in [0, 1, 2]:
        for backend, name in backends:
            try:
                print(f"[Camera] Trying index {idx} via {name}…")
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                # Sanity check: actually grab a frame
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"[Camera] ✓ Opened camera {idx} via {name}")
                    return cap
                cap.release()
            except Exception as e:
                print(f"[Camera] {name} index {idx}: {e}")

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 1 — Capture
# ══════════════════════════════════════════════════════════════════════════════
def capture_thread():
    global latest_raw

    while True:
        cap = _open_camera()

        if cap is None:
            print("[Camera] ✗ No camera found. Common fixes:")
            print("         1. Close Zoom / Teams / Meet — they lock the camera")
            print("         2. Windows: Settings → Privacy → Camera → Allow apps")
            print("         3. Check Device Manager for camera driver issues")
            print("[Camera] Retrying in 5 seconds…")
            with cfg.lock:
                cfg.camera_ok = False
            time.sleep(5)
            continue

        # Lower resolution first — more stable on problem cameras
        cap.set(cv2.CAP_PROP_BUFFERSIZE,  1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Try upgrading to 1280×720 if camera supports it
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"[Camera] Resolution: {int(actual_w)}×{int(actual_h)}")

        with cfg.lock:
            cfg.camera_ok = True

        failures = 0
        while failures < 60:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                failures += 1
                time.sleep(0.01)
                continue
            failures = 0
            frame = cv2.flip(frame, 1)
            with _state_lock:
                latest_raw = frame
            q_put_latest(raw_q, frame)

        print("[Camera] Too many failures — reopening…")
        cap.release()
        with cfg.lock:
            cfg.camera_ok = False
        time.sleep(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Card detector (OpenCV geometry — COCO has no card class)
# ══════════════════════════════════════════════════════════════════════════════
CARD_RATIO_MIN   = 1.35
CARD_RATIO_MAX   = 1.85
CARD_AREA_MIN    = 0.003
CARD_AREA_MAX    = 0.30
CARD_IOU_THRESH  = 0.40

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def detect_cards(frame):
    h, w   = frame.shape[:2]
    farea  = w * h
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    e1     = cv2.Canny(smooth, 50, 150)
    e2     = cv2.Canny(smooth, 20,  80)
    thr    = cv2.adaptiveThreshold(smooth, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    e3     = cv2.Canny(thr, 30, 90)
    edges  = cv2.bitwise_or(cv2.bitwise_or(e1, e2), e3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges  = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates  = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        frac = area / farea
        if frac < CARD_AREA_MIN or frac > CARD_AREA_MAX:
            continue
        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        x, y, bw, bh = cv2.boundingRect(approx)
        if bh == 0: continue
        ratio = max(bw, bh) / min(bw, bh)
        if ratio < CARD_RATIO_MIN or ratio > CARD_RATIO_MAX:
            continue
        x1 = max(0, x);     y1 = max(0, y)
        x2 = min(w-1,x+bw); y2 = min(h-1,y+bh)
        conf = max(0.0, 1.0 - abs(ratio - 1.586) / 1.586)
        candidates.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                               cat="Cards & IDs", conf=round(conf, 2)))

    candidates.sort(key=lambda c: c["conf"], reverse=True)
    kept = []
    for c in candidates:
        box = [c["x1"], c["y1"], c["x2"], c["y2"]]
        if not any(_iou(box, [k["x1"],k["y1"],k["x2"],k["y2"]]) > CARD_IOU_THRESH for k in kept):
            kept.append(c)
    return kept


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD 2 — Inference
# ══════════════════════════════════════════════════════════════════════════════
def inference_thread():
    global latest_boxes, latest_counts

    print("[Inference] Loading YOLOv8n …")
    model = YOLO("yolov8n.pt")
    print("[Inference] ✓ Ready")

    t_prev, frame_n = time.time(), 0

    while True:
        try:
            frame = raw_q.get(timeout=2.0)
        except queue.Empty:
            continue

        h, w = frame.shape[:2]

        with cfg.lock:
            shield   = cfg.shield_active
            conf_th  = cfg.confidence
            cat_en   = dict(cfg.category_enabled)
            prot_mp  = cfg.protect_main
            do_cards = cfg.detect_cards

        if not shield:
            with _state_lock:
                latest_boxes  = []
                latest_counts = {k: 0 for k in cfg.detections}
            continue

        parsed     = []
        det_counts = {k: 0 for k in cfg.detections}

        try:
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

                main_idx = None
                if prot_mp:
                    pidx = [i for i, c in enumerate(clsids_np) if c == 0]
                    if pidx:
                        areas    = [(boxes_np[i][2]-boxes_np[i][0])*(boxes_np[i][3]-boxes_np[i][1]) for i in pidx]
                        main_idx = pidx[int(np.argmax(areas))]

                for i in range(len(boxes_np)):
                    cls_id = clsids_np[i]
                    if cls_id == 0 and i == main_idx:
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
        except Exception as e:
            print(f"[Inference] YOLO error: {e}")

        if do_cards and cat_en.get("Cards & IDs", True):
            try:
                for cb in detect_cards(frame):
                    det_counts["Cards & IDs"] += 1
                    parsed.append(cb)
            except Exception as e:
                print(f"[Inference] Card detect error: {e}")

        with _state_lock:
            latest_boxes  = parsed
            latest_counts = det_counts

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
def apply_blur(frame, x1, y1, x2, y2, k):
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return
    rh, rw  = roi.shape[:2]
    blurred = cv2.GaussianBlur(roi, (k, k), 0)
    scale   = max(2, min(rw, rh) // 12)
    sw, sh  = max(1, rw//scale), max(1, rh//scale)
    small   = cv2.resize(blurred, (sw, sh), interpolation=cv2.INTER_AREA)
    pixel   = cv2.resize(small,   (rw, rh), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = pixel


# ══════════════════════════════════════════════════════════════════════════════
#  FIX 2 — MJPEG stream never crashes; shows placeholder if camera not ready
# ══════════════════════════════════════════════════════════════════════════════
def _placeholder(msg="Starting camera…", colour=(16, 20, 30)):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = colour
    cv2.putText(img, "PRIVACY SHIELD", (150, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 220), 2, cv2.LINE_AA)
    cv2.putText(img, msg, (40, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 110, 140), 1, cv2.LINE_AA)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

# Pre-build common placeholders once
_PH_STARTING = _placeholder("Starting camera…")
_PH_NO_CAM   = _placeholder("No camera found — see terminal", (30, 10, 10))

def generate_mjpeg():
    TARGET_FPS  = 30
    FRAME_DELAY = 1.0 / TARGET_FPS

    while True:
        t0 = time.time()
        try:
            with _state_lock:
                raw   = latest_raw
                boxes = list(latest_boxes)
            with cfg.lock:
                shield   = cfg.shield_active
                blur_k   = max(5, cfg.blur_strength) | 1
                show_b   = cfg.show_boxes
                cam_ok   = cfg.camera_ok

            # ── Camera not ready → send placeholder ───────────────────────────
            if raw is None:
                ph = _PH_STARTING if not cam_ok else _placeholder("Camera warming up…")
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + ph + b"\r\n")
                time.sleep(0.2)
                continue

            out = raw.copy()

            if shield and boxes:
                for box in boxes:
                    try:
                        apply_blur(out, box["x1"], box["y1"], box["x2"], box["y2"], blur_k)
                        if show_b:
                            color = CATEGORY_COLORS_BGR.get(box["cat"], (200, 200, 200))
                            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
                            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                            label = f"{box['cat']}  {box['conf']:.0%}"
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                            ly1 = max(0, y1-th-10)
                            lx2 = min(out.shape[1], x1+tw+8)
                            cv2.rectangle(out, (x1, ly1), (lx2, y1), color, cv2.FILLED)
                            cv2.putText(out, label, (x1+3, y1-4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)
                    except Exception:
                        pass

            txt = "SHIELD ON" if shield else "SHIELD OFF"
            col = (30, 160, 30) if shield else (30, 30, 200)
            cv2.rectangle(out, (10, 10), (165, 40), col, cv2.FILLED)
            cv2.putText(out, txt, (16, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255,255,255), 2, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", out,
                                   [cv2.IMWRITE_JPEG_QUALITY, 82,
                                    cv2.IMWRITE_JPEG_OPTIMIZE, 1])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")

        except GeneratorExit:
            return
        except Exception as e:
            print(f"[Stream] Error: {e}")
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + _placeholder("Stream error — retrying…") + b"\r\n")
            time.sleep(0.5)

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
            camera_ok  = cfg.camera_ok,
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

@app.route("/health")
def health():
    return jsonify(status="ok", model="yolov8n")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import webbrowser

    print("""
╔══════════════════════════════════════════════════╗
║  🛡️  Privacy Shield — Starting                  ║
╠══════════════════════════════════════════════════╣
║  Windows: using DirectShow (fixes MSMF error)   ║
║  Camera:  auto-detected (tries 0, 1, 2)         ║
║  Stream:  placeholder shown while camera starts ║
╚══════════════════════════════════════════════════╝
""")

    threading.Thread(target=capture_thread,   daemon=True, name="Capture").start()
    threading.Thread(target=inference_thread, daemon=True, name="Inference").start()

    time.sleep(2.0)   # give camera time to open before browser launches

    port = int(os.environ.get("PORT", 5000))
    url  = f"http://localhost:{port}"
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    print(f"\n[Flask] → {url}\n")

    if HAS_SOCKETIO:
        socketio.run(app, host="0.0.0.0", port=port, debug=False, use_reloader=False)
    else:
        app.run(host="0.0.0.0", port=port, debug=False,
                threaded=True, use_reloader=False)
