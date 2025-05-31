@echo off
chcp 65001 >nul
echo ========================================
echo YOLO11 + Kalman Filter Object Tracking System Installation
echo ========================================
echo.

echo Checking Python version...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found, please install Python 3.8+
    pause
    exit /b 1
)

echo.
echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo Installation failed, trying with China mirror...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
)

echo.
echo Checking CUDA support...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Version:', torch.version.cuda if torch.cuda.is_available() else 'None')"

echo.
echo Testing module imports...
python -c "from ultralytics import YOLO; print('YOLO import successful')"
python -c "import cv2; print('OpenCV import successful')"
python -c "from filterpy.kalman import KalmanFilter; print('FilterPy import successful')"

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Usage:
echo 1. Camera tracking: python demo.py
echo 2. Video file: python demo.py --source video.mp4
echo 3. Save output: python demo.py --output result.mp4
echo.
echo Press any key to exit...
pause 