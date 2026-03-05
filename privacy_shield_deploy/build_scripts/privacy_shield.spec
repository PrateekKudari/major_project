# privacy_shield.spec
# ─────────────────────────────────────────────────────────────────────────────
# PyInstaller spec — produces a single-folder app with launcher EXE / .app
#
# Build:
#   pip install pyinstaller
#   pyinstaller privacy_shield.spec
#
# Output:
#   dist/PrivacyShield/           ← folder app (Windows / Linux)
#   dist/PrivacyShield.app/       ← macOS bundle (after build_mac.sh)
# ─────────────────────────────────────────────────────────────────────────────

import sys, os

block_cipher = None

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=[
        # Bundle the templates folder
        ("templates", "templates"),
        # Bundle any pre-downloaded YOLO weights if present
        *([("yolov8n.pt", ".")] if os.path.exists("yolov8n.pt") else []),
    ],
    hiddenimports=[
        "ultralytics",
        "ultralytics.models",
        "ultralytics.models.yolo",
        "ultralytics.nn.tasks",
        "flask",
        "flask_socketio",
        "engineio",
        "socketio",
        "cv2",
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        "numpy",
        "torch",
        "torchvision",
        "tkinter",
        "tkinter.ttk",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PrivacyShield",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no black terminal window on Windows
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Windows icon:
    icon="assets/icon.ico" if os.path.exists("assets/icon.ico") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PrivacyShield",
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="PrivacyShield.app",
        icon="assets/icon.icns" if os.path.exists("assets/icon.icns") else None,
        bundle_identifier="com.yourname.privacyshield",
        info_plist={
            # ← This is the KEY line that triggers macOS camera permission dialog
            "NSCameraUsageDescription":
                "Privacy Shield uses your camera to detect and blur sensitive items "
                "during video calls. All processing is local — nothing leaves your device.",

            "NSMicrophoneUsageDescription":
                "Privacy Shield may access the microphone for future audio features.",

            "CFBundleName":            "Privacy Shield",
            "CFBundleDisplayName":     "Privacy Shield",
            "CFBundleVersion":         "1.0.0",
            "CFBundleShortVersionString": "1.0",
            "LSMinimumSystemVersion":  "10.15",
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,   # support dark mode
        },
    )
