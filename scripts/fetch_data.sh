#!/usr/bin/env bash
# Materializa data/churn.csv — o arquivo que o DVC rastreia e o Git não guarda mais.
#
# Ordem de tentativa:
#   1. já está no disco e bate com o hash rastreado  -> não faz nada
#   2. `dvc pull`                                    -> o caminho normal
#   3. blob do Git anterior ao Encontro 5            -> a rede de segurança
#
# O passo 3 existe porque o remote padrão é um diretório local (`../dvc-storage`):
# quem clona o repositório numa máquina nova não tem esse diretório, e o CI muito
# menos. Como o CSV esteve versionado no Git até o Encontro 5, os bytes continuam
# no histórico — e o hash do .dvc é quem confirma que são os bytes certos.
set -euo pipefail

cd "$(dirname "$0")/.."

DATASET="data/churn.csv"
DVC_FILE="${DATASET}.dvc"
# Blob de data/churn.csv no commit 049e49b (Encontro 5, antes de o DVC assumir).
# Hash de conteúdo: não muda se os commits forem reescritos.
BLOB="eb8a7d540c8e7316e3c4011211c1920d379bffac"

md5_de() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"  # macOS
    fi
}

esperado=$(sed -n 's/^[[:space:]-]*md5:[[:space:]]*\([0-9a-f]\{32\}\).*/\1/p' "$DVC_FILE" | head -1)
if [ -z "$esperado" ]; then
    echo "ERRO: não consegui ler o md5 de $DVC_FILE" >&2
    exit 1
fi

confere() { [ -f "$DATASET" ] && [ "$(md5_de "$DATASET")" = "$esperado" ]; }

if confere; then
    echo "$DATASET já está no disco e bate com $DVC_FILE ($esperado)."
    exit 0
fi

# O dvc pode estar no PATH ou só dentro do ambiente do uv (grupo `data`).
# --no-sync: usar o que já está instalado, nunca disparar um download surpresa.
dvc_pull() {
    if command -v dvc >/dev/null 2>&1; then
        dvc pull "$DVC_FILE"
    elif command -v uv >/dev/null 2>&1 \
        && uv run --frozen --no-sync dvc --version >/dev/null 2>&1; then
        uv run --frozen --no-sync dvc pull "$DVC_FILE"
    else
        return 127
    fi
}

echo "Tentando dvc pull..."
status=0
dvc_pull || status=$?

if [ "$status" -eq 0 ] && confere; then
    echo "$DATASET veio do remote do DVC."
    exit 0
elif [ "$status" -eq 127 ]; then
    echo "dvc não está instalado aqui (uv sync --group data instala)."
else
    echo "dvc pull não deixou o arquivo no lugar (remote indisponível?)."
fi

echo "Recuperando do histórico do Git (blob ${BLOB:0:7})..."
if git cat-file -e "$BLOB" 2>/dev/null; then
    git cat-file blob "$BLOB" > "$DATASET"
else
    echo "ERRO: o blob $BLOB não está neste clone." >&2
    echo "      Num clone raso, rode: git fetch --unshallow" >&2
    exit 1
fi

if ! confere; then
    obtido=$(md5_de "$DATASET")
    echo "ERRO: o arquivo recuperado não bate com $DVC_FILE." >&2
    echo "      esperado: $esperado" >&2
    echo "      obtido:   $obtido" >&2
    exit 1
fi

echo "$DATASET recuperado do histórico e conferido contra $DVC_FILE ($esperado)."
echo "Dica: 'dvc push' manda esses bytes para o seu remote."
