#!/usr/bin/env bash
# Alterna o dataset entre as versões registradas — tarefa do Encontro 5.
#
# São dois movimentos, e é isso que o encontro quer que fique claro:
#
#   git checkout <commit> -- data/churn.csv.dvc   # o Git traz o PONTEIRO
#   dvc checkout                                  # o DVC traz os BYTES
#
# O Git sozinho troca 100 bytes de texto; sem o segundo comando o CSV no disco
# continua sendo o da versão anterior — e aí o repositório mente sobre qual dado
# está ali. `dvc status` acusa; este script não deixa acontecer.
#
#   ./scripts/alternar_versao.sh v1
#   ./scripts/alternar_versao.sh v2
set -euo pipefail

cd "$(dirname "$0")/.."

PONTEIRO="data/churn.csv.dvc"
DATASET="data/churn.csv"

# As versões pelo que elas são — o hash do conteúdo — e não por número de
# commit, que muda a cada rebase.
V1="d390bd07b5514a2256a2396993b8b0e3"  # 7043 clientes, o Telco como veio
V2="f150a64ab5537f24b16770ba76ad7b95"  # 7143 clientes, v1 + scripts/gerar_v2.py

case "${1:-}" in
    v1) alvo="$V1" ;;
    v2) alvo="$V2" ;;
    *)
        echo "uso: $0 v1|v2" >&2
        echo "  v1  $V1  (7043 clientes)" >&2
        echo "  v2  $V2  (7143 clientes)" >&2
        exit 1
        ;;
esac

md5_de() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" | cut -d' ' -f1
    else
        md5 -q "$1"  # macOS
    fi
}

dvc_checkout() {
    if command -v dvc >/dev/null 2>&1; then
        dvc checkout "$PONTEIRO"
    elif command -v uv >/dev/null 2>&1 \
        && uv run --frozen --no-sync dvc --version >/dev/null 2>&1; then
        uv run --frozen --no-sync dvc checkout "$PONTEIRO"
    else
        return 127
    fi
}

# Qual commit declara esse hash? Percorre só os commits que tocaram o ponteiro.
commit=""
for candidato in $(git log --format=%H -- "$PONTEIRO"); do
    if git show "$candidato:$PONTEIRO" 2>/dev/null | grep -q "$alvo"; then
        commit="$candidato"
        break
    fi
done

if [ -z "$commit" ]; then
    echo "ERRO: nenhum commit tem um $PONTEIRO com md5 $alvo." >&2
    echo "      Num clone raso, rode: git fetch --unshallow" >&2
    exit 1
fi

antes=$([ -f "$DATASET" ] && md5_de "$DATASET" || echo "ausente")

echo "$1 → ponteiro do commit ${commit:0:7}"
git checkout "$commit" -- "$PONTEIRO"

# O ponteiro já mudou; agora o dado tem de segui-lo. dvc checkout resolve pelo
# cache local; sem DVC por perto, o fetch_data.sh reconstrói e confere o hash.
if ! dvc_checkout 2>/dev/null || [ "$(md5_de "$DATASET")" != "$alvo" ]; then
    ./scripts/fetch_data.sh >/dev/null
fi

depois=$(md5_de "$DATASET")
if [ "$depois" != "$alvo" ]; then
    echo "ERRO: o dado no disco ($depois) não é o do ponteiro ($alvo)." >&2
    exit 1
fi

echo "  dado:     ${antes:0:8} → ${depois:0:8}"
echo "  clientes: $(($(wc -l < "$DATASET") - 1))"
echo "  ponteiro e dado batem. Para voltar: $0 $([ "$1" = v1 ] && echo v2 || echo v1)"
