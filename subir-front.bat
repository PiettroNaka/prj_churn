@echo off
REM Duplo clique aqui: sobe o front de churn inteiro -- dado, campeao, API e
REM Streamlit -- e abre o navegador.
REM
REM %~dp0 e a pasta DESTE arquivo, entao nao importa de onde ele for chamado.
REM Deixe a janela aberta enquanto estiver usando: fechar encerra os dois
REM servidores.
cd /d "%~dp0"

echo Preparando o ambiente (uv sync)...
echo.
uv sync --group api --group app
if errorlevel 1 (
    echo.
    echo O uv sync falhou -- a mensagem esta logo acima.
    echo Se disser que "uv" nao foi encontrado, instale com:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo e reabra esta janela.
    pause
    exit /b 1
)

uv run --group api --group app python scripts/subir_front.py
if errorlevel 1 (
    echo.
    echo Algo falhou -- a mensagem esta logo acima.
    echo Copie e cole para pedir ajuda.
    pause
    exit /b 1
)

echo.
echo Servidores encerrados.
pause
