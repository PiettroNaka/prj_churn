@echo off
REM Duplo clique aqui: gera o relatorio dos runs e abre no navegador.
REM
REM %~dp0 e a pasta DESTE arquivo, entao nao importa de onde ele for chamado --
REM o "cd errado" deixa de existir. E o relatorio e um HTML estatico: abre por
REM file://, sem servidor, sem porta e sem proxy no caminho.
cd /d "%~dp0.."

echo Gerando o relatorio dos runs...
echo.
uv run python scripts/relatorio.py --html
if errorlevel 1 (
    echo.
    echo Algo falhou -- a mensagem esta logo acima.
    echo Copie e cole para pedir ajuda.
    pause
    exit /b 1
)

start "" "artifacts\runs.html"
echo.
echo Pronto: o relatorio abriu no navegador.
pause
