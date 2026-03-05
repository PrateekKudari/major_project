@echo off
:: ─────────────────────────────────────────────────────────────────
::  build_windows.bat — Build PrivacyShield.exe for Windows
::
::  Run from project root:
::    build_scripts\build_windows.bat
::
::  Output: dist\PrivacyShield\PrivacyShield.exe
::  The Windows app manifest declares webcam device capability so
::  Windows 10/11 shows the camera-permission consent dialog.
:: ─────────────────────────────────────────────────────────────────

title Privacy Shield — Windows Build
echo.
echo ============================================
echo   Privacy Shield - Windows Build Script
echo ============================================
echo.

cd /d "%~dp0.."

echo [1/4] Installing dependencies...
pip install pyinstaller ultralytics flask flask-socketio opencv-python Pillow

echo.
echo [2/4] Pre-fetching YOLOv8n weights...
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
copy "%USERPROFILE%\.cache\ultralytics\yolov8n.pt" . 2>nul

echo.
echo [3/4] Building EXE with PyInstaller + Windows manifest...
pyinstaller build_scripts\privacy_shield.spec ^
    --clean --noconfirm ^
    --manifest build_scripts\PrivacyShield.exe.manifest

echo.
echo [4/4] Embedding manifest into EXE (requires Windows SDK)...
where mt.exe >nul 2>&1
if %errorlevel%==0 (
    mt.exe -manifest build_scripts\PrivacyShield.exe.manifest ^
           -outputresource:dist\PrivacyShield\PrivacyShield.exe;1
    echo    Manifest embedded.
) else (
    echo    mt.exe not found - manifest already set by PyInstaller.
)

echo.
echo ============================================
echo   Build complete!
echo   Output: dist\PrivacyShield\PrivacyShield.exe
echo.
echo   Windows camera permission dialog will
echo   appear automatically on first launch.
echo   (Settings > Privacy > Camera > PrivacyShield)
echo ============================================
echo.
pause
