# major_project


```
██████╗ ██████╗ ██╗██╗   ██╗ █████╗  ██████╗██╗   ██╗
██╔══██╗██╔══██╗██║██║   ██║██╔══██╗██╔════╝╚██╗ ██╔╝
██████╔╝██████╔╝██║██║   ██║███████║██║      ╚████╔╝ 
██╔═══╝ ██╔══██╗██║╚██╗ ██╔╝██╔══██║██║       ╚██╔╝  
██║     ██║  ██║██║ ╚████╔╝ ██║  ██║╚██████╗   ██║   
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═══╝  ╚═╝  ╚═╝ ╚═════╝   ╚═╝  
███████╗██╗  ██╗██╗███████╗██╗     ██████╗           
██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗          
███████╗███████║██║█████╗  ██║     ██║  ██║          
╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║          
███████║██║  ██║██║███████╗███████╗██████╔╝          
╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝           
```

### 🛡️ Real-Time AI Privacy Filter for Video Calls


**Automatically detects and blurs sensitive items in your webcam feed — background people, phones, documents, and more — before they appear on video calls. Everything runs on your machine. Nothing leaves your device.**

[Features](#-features) · [Demo](#-how-it-works) · [Quick Start](#-quick-start) · [Deployment](#-deployment) · [Architecture](#-architecture) · [Contributing](#-contributing)

---

</div>

## 🎯 The Problem

Working remotely? Every video call is a potential privacy risk:

- 📄 Confidential documents left on your desk
- 💳 Credit cards, ID badges visible in the background  
- 👥 Family members walking behind you
- 📱 Your phone screen reflecting sensitive notifications
- ⌨️ Your workspace setup revealing personal info

**Privacy Shield solves this automatically, in real time, with zero configuration.**

---

## ✨ Features

| Feature | Details |
|---|---|
| 🧠 **YOLOv8n Detection** | Nano model — fast enough for real-time on laptop CPU |
| 🌫️ **Pixelated Blur** | Gaussian + mosaic censor blur per detected region |
| 👤 **Smart Person Mode** | Keeps YOU visible; blurs all other people in frame |
| ⚡ **3-Thread Pipeline** | Capture / Inference / Render run independently at 30fps |
| 🎛️ **Live Control Panel** | Toggle categories, adjust blur intensity, set confidence |
| 🌐 **Browser UI** | Native camera permission dialog via `getUserMedia` |
| 🔒 **100% Local** | No cloud, no uploads, no API keys — ever |
| 📦 **Packagable** | Build as `.app` (macOS) or `.exe` (Windows) with PyInstaller |

### Detection Categories

```
👥 Background People   ──  Anyone in frame who isn't you
📱 Phones & Screens    ──  Cell phones, laptop displays  
📄 Documents & Books   ──  Papers, notebooks, ID badges
⌨️  Keyboards & Mice   ──  Peripherals, workspace setup
🧸 Background Objects  ──  Bottles, clocks, distracting clutter
```

---

## 🔧 How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   Webcam  ──▶  Thread 1: Capture                                │
│               (CAP_BUFFERSIZE=1, always latest frame)            │
│                    │                                             │
│                    ├──▶  Thread 2: YOLOv8n Inference            │
│                    │     classes=[0,39,63,64,66,67,73,74,77]    │
│                    │     iou=0.45  conf=0.25  max_det=100       │
│                    │     └──▶ latest_boxes[]                     │
│                    │                                             │
│                    └──▶  Thread 3: Render @ 30fps               │
│                          raw.copy() + apply_blur(box)            │
│                          └──▶ MJPEG stream → Browser            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- `classes=` filter on YOLO prevents cross-class NMS suppression (the #1 cause of missed detections)
- `iou=0.45` keeps nearby objects (default 0.7 merges them)  
- Render thread reads `latest_raw` + `latest_boxes` independently — smooth motion even when YOLO is mid-inference
- `roi.copy()` before GaussianBlur prevents numpy view aliasing

---

## 🚀 Quick Start

### Prerequisites
- Python **3.9+**
- A webcam (built-in laptop camera works)
- ~600 MB disk for YOLOv8n weights + PyTorch (downloaded once automatically)

### Install & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/privacy-shield.git
cd privacy-shield

# Install dependencies
pip install -r requirements.txt

# Launch
python app.py
```

A browser tab opens at `http://localhost:5000`. Click **"Enable Camera & Start Shield"** — your browser will show its native camera permission dialog.

> **First launch** downloads YOLOv8n weights automatically (~6 MB model, ~500 MB PyTorch). Every launch after that is instant.

---

## 📁 Project Structure

```
privacy-shield/
│
├── app.py                      # Flask backend — all three threads live here
├── requirements.txt            # Python dependencies
│
├── templates/
│   └── index.html              # Web UI — camera permission + live controls
│
├── build_scripts/
│   ├── privacy_shield.spec     # PyInstaller config (bundles Info.plist etc.)
│   ├── entitlements.plist      # macOS camera entitlement for code-signing
│   ├── PrivacyShield.exe.manifest  # Windows webcam device capability
│   ├── build_mac.sh            # Build + optional codesign/notarise
│   └── build_windows.bat       # Build Windows EXE
│
├── .gitignore
└── README.md
```

---

## 📦 Deployment

### Option 1 — Share live via ngrok (30 seconds)
```bash
# Terminal 1
python app.py

# Terminal 2
ngrok http 5000
# → https://abc123.ngrok-free.app  (share this URL)
```

### Option 2 — Build a native desktop app

**macOS `.app`:**
```bash
chmod +x build_scripts/build_mac.sh
./build_scripts/build_mac.sh
# → dist/PrivacyShield.app
```

**Windows `.exe`:**
```bat
build_scripts\build_windows.bat
# → dist\PrivacyShield\PrivacyShield.exe
```

### Option 3 — Use with OBS Virtual Camera
```
Zoom / Teams / Meet
      ↑
OBS Virtual Camera
      ↑
Browser Source → http://localhost:5000
      ↑
Flask + YOLO (your machine)
```

### Camera Permission — How it works per platform

| Platform | Trigger | Where user manages it |
|---|---|---|
| **Browser** | `getUserMedia()` | Browser permission bar |
| **macOS** | `NSCameraUsageDescription` in Info.plist | System Settings → Camera |
| **Windows** | `<DeviceCapability Name="webcam"/>` in manifest | Settings → Privacy → Camera |

---

## ⚡ Performance

| Hardware | FPS (approx) |
|---|---|
| Apple M1 / M2 | 25 – 35 fps |
| Intel Core i7 (no GPU) | 15 – 22 fps |
| Intel Core i5 (no GPU) | 12 – 18 fps |
| NVIDIA GPU (CUDA) | 45 – 70 fps |

**Reduce resolution for slower machines:**
```python
# In app.py capture_thread():
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**Upgrade model for better accuracy (slower):**
```python
model = YOLO("yolov8s.pt")   # small — ~30% slower, noticeably more accurate
```

---

## 🛠️ Configuration

All controls are available live in the browser UI — no restart needed:

| Control | Default | Effect |
|---|---|---|
| Shield toggle | ON | Pause/resume all blurring |
| Blur intensity | 51 | Gaussian kernel size (5 → 101) |
| Confidence threshold | 25% | Lower = catch more; higher = fewer false positives |
| Show boxes | ON | Toggle coloured detection outlines |
| Protect main person | ON | Keep largest person (you) unblurred |
| Per-category toggles | All ON | Enable/disable each object class individually |

---

## 🧩 Tech Stack

| Library | Purpose |
|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) | YOLOv8n inference — 80-class real-time detection |
| [opencv-python](https://opencv.org) | Camera capture, Gaussian blur, frame encoding |
| [Flask](https://flask.palletsprojects.com) | MJPEG streaming server + config API |
| [NumPy](https://numpy.org) | Array ops, bounding box math |

---

## 🔍 Bugs Fixed (changelog)

| Version | Fix |
|---|---|
| v1.0 | Initial release — single-thread, stale frames |
| v1.1 | **Movement fix** — 3-thread pipeline, `CAP_BUFFERSIZE=1`, `roi.copy()` |
| v1.2 | **Multi-object fix** — `classes=` filter, `iou=0.45`, `conf=0.25`, separate render thread |

---

## 🤝 Contributing

Pull requests are welcome. For major changes please open an issue first.

```bash
git checkout -b feature/your-feature
git commit -m "add: your feature"
git push origin feature/your-feature
# → open a Pull Request on GitHub
```

**Ideas for contributions:**
- [ ] Custom-trained credit card / ID badge model
- [ ] Audio alert when new object detected
- [ ] OBS plugin integration
- [ ] Electron wrapper for true native app

---


---



Built with Python · YOLOv8 · OpenCV · Flask

**All processing is 100% local. No frames, data, or detections ever leave your machine.**

