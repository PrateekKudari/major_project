#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
#  build_mac.sh — Build PrivacyShield.app for macOS
#
#  Usage:
#    chmod +x build_scripts/build_mac.sh
#    ./build_scripts/build_mac.sh
#
#  Optional (code-signing + notarisation):
#    SIGN_ID="Developer ID Application: YOUR NAME (TEAMID)"
#    APPLE_ID="you@email.com"
#    APP_PASSWORD="xxxx-xxxx-xxxx-xxxx"   # app-specific password
#    ./build_scripts/build_mac.sh
# ─────────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")/.."   # always run from project root

echo ""
echo "════════════════════════════════════════════"
echo "  🛡️  Privacy Shield — macOS Build Script"
echo "════════════════════════════════════════════"
echo ""

# 1. Dependencies
echo "[1/4] Installing Python dependencies…"
pip3 install pyinstaller ultralytics flask flask-socketio opencv-python Pillow

# 2. Pre-download YOLO weights so they're bundled into the app
echo "[2/4] Pre-fetching YOLOv8n weights…"
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
cp ~/.cache/ultralytics/yolov8n.pt . 2>/dev/null || true

# 3. PyInstaller build
echo "[3/4] Building .app bundle with PyInstaller…"
pyinstaller build_scripts/privacy_shield.spec --clean --noconfirm

echo ""
echo "✅  Build complete →  dist/PrivacyShield.app"

# 4. Optional: code-sign + notarise
if [ -n "$SIGN_ID" ]; then
  echo ""
  echo "[4/4] Code-signing with: $SIGN_ID"
  codesign --force --deep --sign "$SIGN_ID" \
           --entitlements build_scripts/entitlements.plist \
           --options runtime \
           dist/PrivacyShield.app

  echo "✅  Signed."

  if [ -n "$APPLE_ID" ] && [ -n "$APP_PASSWORD" ]; then
    echo "     Submitting for notarisation…"
    ditto -c -k --keepParent dist/PrivacyShield.app dist/PrivacyShield.zip
    xcrun notarytool submit dist/PrivacyShield.zip \
          --apple-id "$APPLE_ID" \
          --password "$APP_PASSWORD" \
          --team-id "$(echo "$SIGN_ID" | grep -oE '\([A-Z0-9]+\)' | tr -d '()')" \
          --wait
    xcrun stapler staple dist/PrivacyShield.app
    echo "✅  Notarised & stapled."
  fi
else
  echo ""
  echo "ℹ️  Skipping code-sign (set SIGN_ID env var to enable)."
  echo "   Users will need to right-click → Open the first time (Gatekeeper)."
  echo "   Camera permission dialog will still appear automatically."
fi

echo ""
echo "════════════════════════════════════════════"
echo "  App ready:  dist/PrivacyShield.app"
echo "  Camera permission is declared in Info.plist"
echo "  → NSCameraUsageDescription key is set ✓"
echo "════════════════════════════════════════════"
echo ""
