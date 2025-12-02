@echo off
chcp 65001 >nul
title Download de Modelos YOLO

echo ============================================
echo   Download de Modelos para Detecção de Placas
echo ============================================
echo.

cd /d "%~dp0models"

echo [1/5] Baixando yolo11n.pt (5.6MB)...
if not exist "yolo11n.pt" (
    curl -L -o yolo11n.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
    echo      OK!
) else (
    echo      Já existe, pulando...
)

echo [2/5] Baixando yolo11s.pt (19MB)...
if not exist "yolo11s.pt" (
    curl -L -o yolo11s.pt "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
    echo      OK!
) else (
    echo      Já existe, pulando...
)

echo [3/5] Baixando license-plate-v1n.pt (5.4MB)...
if not exist "license-plate-v1n.pt" (
    curl -L -o license-plate-v1n.pt "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1n.pt"
    echo      OK!
) else (
    echo      Já existe, pulando...
)

echo [4/5] Baixando license-plate-v1s.pt (19MB)...
if not exist "license-plate-v1s.pt" (
    curl -L -o license-plate-v1s.pt "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1s.pt"
    echo      OK!
) else (
    echo      Já existe, pulando...
)

echo [5/5] Baixando license-plate-v1x.pt (114MB)...
if not exist "license-plate-v1x.pt" (
    curl -L -o license-plate-v1x.pt "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1x.pt"
    echo      OK!
) else (
    echo      Já existe, pulando...
)

echo.
echo ============================================
echo   Download concluído!
echo ============================================
echo.
echo Modelos disponíveis:
dir /b *.pt
echo.
pause
