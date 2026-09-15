# Rodar tudo na sua máquina

Do clone até ver os modelos no MLflow. Cada comando aqui foi executado neste
projeto — não é roteiro de memória.

## 0. O que precisa estar instalado

| | por quê | conferir |
|---|---|---|
| **git** | pegar o código e recuperar o dataset do histórico | `git --version` |
| **uv** | o ambiente Python travado (Encontro 2) | `uv --version` |
| Docker *(opcional)* | só para o caminho em container | `docker --version` |

Se faltar o `uv`:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

> **Windows:** os scripts `.sh` **não** rodam no PowerShell nem no CMD. Use o
> **Git Bash**, que vem junto com o Git for Windows (botão direito na pasta →
> "Git Bash Here"). Todo o resto — `uv run …` — funciona em qualquer terminal.

## 1. O código

```bash
git clone https://github.com/PiettroNaka/prj_churn.git
cd prj_churn
```

Já tem o repositório? `git checkout main && git pull`.

## 2. O ambiente

```bash
uv sync
```

Instala exatamente as versões do `uv.lock`. Não precisa criar venv à mão, nem
ativar nada: todo comando daqui em diante começa com `uv run`.

## 3. O dataset

```bash
./scripts/fetch_data.sh
```

O CSV não está no Git — quem o versiona é o DVC (Encontro 5). O script tenta
`dvc pull` e, sem remote alcançável, reconstrói o arquivo do histórico do Git,
conferindo o md5 contra `data/churn.csv.dvc`. Deve terminar com:

```
data/churn.csv reconstruído e conferido contra data/churn.csv.dvc (f150a64ab…)
```

## 4. Treinar

```bash
uv run python -m churn.model
```

```
Modelo treinado. Métricas no teste:
    accuracy: 0.7609
   roc_auc: 0.8150
```

Isso cria o `mlflow.db` e o primeiro run. Para as nove combinações do grid:
`uv run python -m churn.experiments` (~1 minuto).

## 5. Ver os modelos

```bash
./scripts/mlflow_ui.sh          # abre em http://localhost:5000
./scripts/mlflow_ui.sh 5001     # se a 5000 estiver ocupada
```

O script materializa o dataset se faltar e treina se não houver run nenhum —
então, na prática, ele sozinho cobre os passos 3 a 5.

**Ao abrir, duas coisas confundem:**

1. A UI 3.16 **abre na aba GenAI** (traces, judges, playground), que não tem
   nada a ver com este projeto e aparece vazia. Seus runs estão em **"Model
   training"**, no canto superior esquerdo.
2. As **colunas de métrica não vêm ligadas**. Use o botão **Columns** para
   marcar `roc_auc`, `accuracy` e `data_md5`, ou vá direto ao endereço já
   ordenado:
   `http://localhost:5000/#/experiments/1/runs?orderByKey=metrics.roc_auc&orderByAsc=false`

## 6. Rodar o modelo

```bash
uv run python scripts/prever.py --n 10
```

```
customerID  tenure       Contract real churn  probabilidade  acertou
 7004-MAFB       8 Month-to-month  Yes   Yes          0.815     True
```

Como HTTP, em vez de lote:

```bash
uv run mlflow models serve -m runs:/<run_id>/model --env-manager local
```

## Tudo de uma vez

```bash
git clone https://github.com/PiettroNaka/prj_churn.git && cd prj_churn
uv sync
./scripts/mlflow_ui.sh
```

## Quando der errado

| Sintoma | O que é | Saída |
|---|---|---|
| `uv: command not found` | o instalador não entrou no PATH | feche e reabra o terminal, ou `source $HOME/.local/bin/env` |
| `./scripts/…: command not found` (Windows) | PowerShell não roda `.sh` | use o **Git Bash** |
| `Permission denied` ao rodar um script | bit de execução perdido no clone | `chmod +x scripts/*.sh` |
| Porta 5000 ocupada no **macOS** | é o AirPlay Receiver, não outro MLflow | `./scripts/mlflow_ui.sh 5001`, ou desligue em Ajustes → Geral → AirDrop e Handoff |
| UI abre vazia, sem run nenhum | banco novo, nada treinado ainda | `uv run python -m churn.model` |
| UI mostra só traces e "No data" | é a aba GenAI | clique em **"Model training"** |
| `dvc: command not found` | o DVC é opcional, fora do grupo padrão | `uv sync --group data` |
| `data/churn.csv não existe` | o dataset não foi materializado | `./scripts/fetch_data.sh` |
| O blob não está neste clone | clone raso | `git fetch --unshallow` |
| Artefato do run não abre na UI | run criado em container, UI no host (o caminho é absoluto no banco) | use o mesmo mundo nos dois: ou tudo local, ou `docker compose up mlflow` |

## Pelo container, se preferir

```bash
docker compose up dados     # traz o CSV por dvc pull
docker compose up treino    # treina
docker compose up mlflow    # a UI em localhost:5000
```

O dado nunca entra na imagem; chega por volume. É o entregável do Encontro 5.
