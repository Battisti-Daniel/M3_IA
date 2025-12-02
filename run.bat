@echo off
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
) else (
    echo [AVISO] Ambiente virtual nao encontrado. Executando com Python global.
)

python main.py
pause
