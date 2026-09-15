#!/usr/bin/env bash
# Sobe a UI do MLflow para ver os runs e os modelos — na SUA máquina.
#
#   ./scripts/mlflow_ui.sh          # http://localhost:5000
#   ./scripts/mlflow_ui.sh 5001     # outra porta
#
# Faz o que falta antes de abrir, nesta ordem:
#   1. o dataset, se não estiver no disco (scripts/fetch_data.sh)
#   2. um treino, se ainda não houver nenhum run — UI vazia não mostra modelo
#   3. a UI, apontada para o mesmo backend que o treino usa
#
# O backend é sqlite:///mlflow.db, e não o ./mlruns do material: o MLflow 3
# recusa o file store. Respeita CHURN_MLFLOW_TRACKING_URI, igual ao resto do
# projeto.
set -euo pipefail

cd "$(dirname "$0")/.."

PORTA="${1:-5000}"
URI="${CHURN_MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"

# O MLflow 3 manda telemetria para config.mlflow-telemetry.io. Num projeto de
# curso isso não serve para nada; desligado aqui, à vista e não em silêncio.
export MLFLOW_DISABLE_TELEMETRY="${MLFLOW_DISABLE_TELEMETRY:-true}"

if [ ! -f data/churn.csv ]; then
    echo "==> dataset ausente; materializando"
    ./scripts/fetch_data.sh
fi

# Só dá para checar a existência de runs sem abrir o banco quando o backend é um
# arquivo local. Para um servidor remoto, quem manda é o servidor.
banco=""
case "$URI" in
    sqlite:////*) banco="/${URI#sqlite:////}" ;;
    sqlite:///*) banco="${URI#sqlite:///}" ;;
esac

if [ -n "$banco" ] && [ ! -s "$banco" ]; then
    echo "==> nenhum run ainda ($banco não existe); treinando uma vez"
    uv run --frozen python -m churn.model
fi

echo
echo "==> MLflow UI"
echo "    backend: $URI"
echo "    abra:    http://localhost:${PORTA}"
echo "    (Ctrl+C encerra)"
echo
exec uv run --frozen mlflow ui --backend-store-uri "$URI" --port "$PORTA"
