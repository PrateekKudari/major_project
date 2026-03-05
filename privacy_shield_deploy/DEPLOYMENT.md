# 🛡️ Privacy Shield — Deployment Guide

Three ways to deploy Privacy Shield so it **requests camera permission** like a proper app:

---

## Option A — Web App (Recommended ⭐)
*Browser handles the camera permission dialog automatically.*

```
pip install -r requirements.txt
python app.py
```

A browser tab opens at `http://localhost:5000`.
The browser shows its native **"Allow camera access?"** dialog on the first visit.

### Use in Video Calls via OBS Virtual Camera
```
Zoom / Teams / Meet
        ↑
  OBS Virtual Camera
        ↑
  Browser Source → http://localhost:5000
        ↑
   Flask (YOLO processing)
        ↑
  Laptop webcam
```

1. Install [OBS Studio](https://obsproject.com)
2. Add **Browser Source** → URL: `http://localhost:5000`
3. Click **Start Virtual Camera** in OBS
4. In your video call app → select **OBS Virtual Camera**

---

## Option B — Native macOS App (.app)

### Build
```bash
chmod +x build_scripts/build_mac.sh
./build_scripts/build_mac.sh
```

Output: `dist/PrivacyShield.app`

### How camera permission works on macOS
The `NSCameraUsageDescription` key in `Info.plist` (set in `privacy_shield.spec`) triggers the **system camera-permission dialog** the first time the app accesses the camera:

```
"PrivacyShield" would like to access the camera.
"Privacy Shield uses your camera to detect and blur
 sensitive items during video calls..."
   [ Don't Allow ]  [ OK ]
```

Users can review/revoke at any time: **System Settings → Privacy & Security → Camera**.

### Optional: Code-sign for Gatekeeper
```bash
SIGN_ID="Developer ID Application: Your Name (TEAMID)" \
APPLE_ID="you@email.com" \
APP_PASSWORD="xxxx-xxxx-xxxx-xxxx" \
./build_scripts/build_mac.sh
```

Without signing, users right-click → Open the first time.

---

## Option C — Native Windows App (.exe)

### Build
```
build_scripts\build_windows.bat
```

Output: `dist\PrivacyShield\PrivacyShield.exe`

### How camera permission works on Windows
The `PrivacyShield.exe.manifest` declares `<DeviceCapability Name="webcam"/>`, which registers the app with Windows' camera privacy system.

On **Windows 10/11**, users can control access at:
**Settings → Privacy & Security → Camera → "Let apps access your camera"**

The app also triggers the standard Windows consent flow on first launch.

---

## Folder Structure

```
privacy_shield_deploy/
├── app.py                          ← Flask backend (YOLO + streaming)
├── requirements.txt
├── templates/
│   └── index.html                  ← Web UI (camera permission via getUserMedia)
└── build_scripts/
    ├── privacy_shield.spec         ← PyInstaller config (Info.plist + bundle)
    ├── entitlements.plist          ← macOS camera entitlement
    ├── PrivacyShield.exe.manifest  ← Windows webcam device capability
    ├── build_mac.sh                ← macOS build + optional codesign/notarise
    └── build_windows.bat           ← Windows build script
```

---

## Camera Permission Cheat Sheet

| Platform | How permission is declared | Where user sees it |
|---|---|---|
| **Browser** | `navigator.mediaDevices.getUserMedia()` | Browser permission bar |
| **macOS** | `NSCameraUsageDescription` in `Info.plist` | System dialog sheet |
| **Windows 10/11** | `<DeviceCapability Name="webcam"/>` in manifest | Settings → Privacy → Camera |
| **Linux** | No OS-level permission prompt (udev rules) | N/A |

---

## Troubleshooting

**macOS: "PrivacyShield" is not from an identified developer**
→ Right-click the .app → Open → Open anyway (first time only).
→ Or: `xattr -cr dist/PrivacyShield.app`

**Windows: Camera blocked by Settings**
→ Settings → Privacy & Security → Camera → Enable for PrivacyShield.

**Low FPS**
→ Reduce resolution in `app.py`:
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

**Port already in use**
→ Change `port=5000` to `port=5001` in `app.py`.
