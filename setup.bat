@echo off
cd /d "%~dp0"

echo ==== Criando/Ativando ambiente virtual ====
if exist venv (
    echo Ambiente virtual existente encontrado.
) else (
    echo Criando ambiente virtual com Python 3.11...
    py -3.11 -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar ambiente virtual.
        echo Certifique-se de que Python 3.11 esta instalado.
        pause
        exit /b 1
    )
)

call venv\Scripts\activate.bat

echo ==== Instalando dependencias ====
pip install --upgrade pip

echo ==== Instalando PyTorch com CUDA 12.4 ====
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo ==== Instalando demais dependencias ====
pip install ultralytics opencv-python numpy pillow easyocr lapx filterpy pyqt5 mss

echo.
echo ==== Verificando instalacao do PyTorch ====
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA disponivel: {torch.cuda.is_available()}'); print(f'Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo.
echo ==== Ambiente pronto! ====
pause
