@echo off
chcp 65001 >nul
echo ========================================
echo YOLO11 + Kalman Filter + SIFT Tracking
echo ========================================
echo.
echo Please select an option:
echo 1. Start camera tracking
echo 2. Open configuration GUI
echo 3. View current configuration
echo 4. Test hover functionality
echo 5. Test prediction configuration
echo 6. Test camera settings
echo 7. Test enhanced features
echo 8. List available cameras
echo 9. Reset configuration
echo 0. Exit
echo.
set /p choice="Enter your choice (0-9): "

if "%choice%"=="1" (
    echo Starting camera tracking...
    python demo.py
) else if "%choice%"=="2" (
    echo Opening configuration GUI...
    python config_gui.py
) else if "%choice%"=="3" (
    echo Viewing current configuration...
    python demo.py --config
) else if "%choice%"=="4" (
    echo Testing hover functionality...
    python test_hover.py
) else if "%choice%"=="5" (
    echo Testing prediction configuration...
    python test_prediction_config.py
) else if "%choice%"=="6" (
    echo Testing camera settings...
    python test_camera_settings.py
) else if "%choice%"=="7" (
    echo Testing enhanced features...
    python test_enhanced_features.py
) else if "%choice%"=="8" (
    echo Listing available cameras...
    python demo.py --list-cameras
) else if "%choice%"=="9" (
    echo Resetting configuration...
    python demo.py --reset-config
) else if "%choice%"=="0" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice. Please try again.
    pause
    goto :eof
)

echo.
echo Press any key to return to menu...
pause >nul
start.bat 