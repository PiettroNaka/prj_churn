@echo off
REM Duplo clique aqui: prepara o que faltar e sobe a UI do MLflow.
REM
REM Esta janela FICA ABERTA de proposito -- e ela que mantem o servidor de pe.
REM Fechar a janela derruba a UI. Ctrl+C encerra.
cd /d "%~dp0.."

echo Preparando o dataset e o treino, se faltarem...
echo Quando aparecer "abra: http://localhost:5000", abra esse endereco no navegador.
echo NAO feche esta janela enquanto estiver usando a UI.
echo.
uv run python scripts/preparar.py
if errorlevel 1 (
    echo.
    echo Algo falhou -- a mensagem esta logo acima.
    pause
    exit /b 1
)
pause
