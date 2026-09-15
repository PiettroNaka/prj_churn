@echo off
REM Duplo clique aqui quando o Git travar: merge pela metade, branch divergida,
REM "local changes would be overwritten". Este script desfaz a bagunca e deixa a
REM pasta igual a main do GitHub.
REM
REM Por que isso acontece: o projeto e escrito no GitHub e so lido aqui. Quando a
REM branch de trabalho e reescrita la, o `git pull` daqui vira um conflito que
REM nao existe para ninguem -- e nao ha nada local que valha a pena salvar.
REM Resolver a mao custa cinco comandos; este arquivo e os cinco.
cd /d "%~dp0"

echo Estado atual desta pasta:
echo.
git status --short --branch
echo.
echo  ---------------------------------------------------------------
echo   Isto descarta TUDO o que estiver modificado aqui e deixa a
echo   pasta identica a main do GitHub.
echo.
echo   NAO sao apagados:  data\churn.csv   (o dataset)
echo                      mlflow.db        (os runs ja treinados)
echo                      artifacts\       (os relatorios)
echo  ---------------------------------------------------------------
echo.

set RESPOSTA=
set /p RESPOSTA="Continuar? (S para sim, qualquer outra tecla cancela) "
if /i not "%RESPOSTA%"=="S" (
    echo.
    echo Cancelado. Nada foi alterado.
    pause
    exit /b 0
)

echo.
echo [1/4] encerrando merge ou rebase pela metade, se houver
git merge --abort >nul 2>&1
git rebase --abort >nul 2>&1

echo [2/4] buscando o GitHub
git fetch origin
if errorlevel 1 (
    echo.
    echo Nao consegui falar com o GitHub -- a mensagem esta acima.
    echo Sem rede, ou o proxy do trabalho no caminho.
    pause
    exit /b 1
)

echo [3/4] alinhando esta pasta com a main
git checkout -f -B main origin/main
if errorlevel 1 (
    echo.
    echo O checkout falhou -- a mensagem esta acima.
    echo Copie e cole para pedir ajuda.
    pause
    exit /b 1
)
REM arquivos soltos que nao pertencem ao projeto. Os ignorados ficam -- dataset,
REM banco e relatorios nao sao tocados, porque -x nao foi passado.
git clean -fd >nul 2>&1

echo [4/4] instalando as dependencias travadas
uv sync
if errorlevel 1 (
    echo.
    echo O uv sync falhou -- a mensagem esta acima.
    echo Se disser que "uv" nao foi encontrado, instale com:
    echo   powershell -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo e reabra a janela.
    pause
    exit /b 1
)

echo.
echo  ---------------------------------------------------------------
echo   Pronto. A pasta esta na main, limpa e com o ambiente instalado:
echo.
git log --oneline -1
echo.
echo   Agora:  ver-runs.bat      ve os modelos ja treinados
echo           abrir-mlflow.bat  sobe a interface em localhost:5000
echo  ---------------------------------------------------------------
echo.
pause
