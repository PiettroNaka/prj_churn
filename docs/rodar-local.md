# Rodar tudo na sua máquina

Do clone até ver os modelos no MLflow. Cada comando aqui foi executado neste
projeto — não é roteiro de memória.

## Windows: o caminho de duplo clique

Se você está no Windows e não quer lidar com terminal, a pasta do projeto tem
dois arquivos para duplo clique:

| arquivo | o que faz |
|---|---|
| **`subir-front.bat`** | sobe tudo — dado, campeão, API e Streamlit — e abre o navegador |
| **`ver-runs.bat`** | gera o relatório dos runs e abre no navegador (HTML estático, sem servidor) |
| **`abrir-mlflow.bat`** | prepara o que faltar e sobe a UI em `localhost:5000` — deixe a janela aberta |
| **`atualizar.bat`** | traz a versão mais nova e desfaz qualquer enrosco do Git |

Os quatro se viram sozinhos quanto à pasta: rodam sempre a partir de onde estão,
então o erro de "abri o terminal na pasta errada" deixa de existir. Se algo
falhar, a janela fica aberta com a mensagem — é só copiar e pedir ajuda.

O `atualizar.bat` merece uma palavra, porque ele **descarta** o que estiver
modificado na pasta. Isso é de propósito: aqui o projeto é lido, não escrito — o
código é feito no GitHub. Quando a branch de trabalho é reescrita lá, o `git
pull` daqui vira um conflito sobre mudanças que não são suas, e sair dele à mão
custa cinco comandos. O script são os cinco, com uma confirmação antes. O que
ele **não** toca: `data\churn.csv`, `mlflow.db` e `artifacts\` — dataset, runs e
relatórios ficam onde estão.

Continua precisando do `uv` instalado (seção abaixo) e de ter clonado o projeto.

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

> **Windows:** os scripts `.sh` **não** rodam no PowerShell nem no CMD. Há dois
> caminhos, e os dois funcionam:
>
> - **PowerShell mesmo**, com o equivalente em Python:
>   `uv run python scripts/preparar.py` — faz dado + treino + UI de uma vez.
> - **Git Bash** (vem com o Git for Windows: botão direito na pasta → "Git Bash
>   Here"), onde os `.sh` rodam normalmente.
>
> Todo o resto — qualquer `uv run …` — funciona em qualquer terminal.

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

Para o ajuste fino — 24 candidatos em seis eixos, com validação cruzada —
`uv run python -m churn.tuning` (~2 minutos). Ele termina treinando o vencedor,
então o run que sai já pode ser promovido com `uv run python -m churn.registry`.

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

## 5b. Se o navegador não abrir o `localhost`

Máquina de trabalho com proxy, porta bloqueada, ou só vontade de não deixar
servidor de pé. Tudo o que a UI mostra sai sem navegador:

```bash
uv run python scripts/relatorio.py           # a tabela no terminal
uv run python scripts/relatorio.py --html    # artifacts/runs.html
```

O HTML é um arquivo só, sem CSS nem JS de fora: abre com **duplo clique**, por
`file://` — sem porta, sem rede, sem servidor.

```
run            roc_auc  accuracy  n_estim.  max_depth  dado (md5)   código
rf-n500-d8     0.8220   0.7749    500       8          f150a64ab5…  d7ee6e5
rf             0.8150   0.7609    200       None       f150a64ab5…  b4912fd
```

Antes de desistir da UI, porém, vale conferir **se o servidor está mesmo no ar** —
`ERR_CONNECTION_REFUSED` quase sempre é isso, não bloqueio:

```powershell
curl.exe http://127.0.0.1:5000     # 200 = no ar;  refused = não subiu
```

E tente `127.0.0.1:5000` no lugar de `localhost:5000`: alguns proxies corporativos
tratam os dois de forma diferente.

## 5c. A API e o front

```bash
uv run --group api uvicorn churn.api:app      # localhost:8000/docs
uv run --group app streamlit run app/Inicio.py  # localhost:8501
```

Os quatro passos de uma vez (é o que o `subir-front.bat` faz no Windows):

```bash
uv run --group api --group app python scripts/subir_front.py
uv run --group api --group app python scripts/subir_front.py --porta 8502
```

Separados, os dois precisam de um campeão promovido — `uv run python -m churn.tuning`
seguido de `uv run python -m churn.registry`. Sem isso a API sobe e responde
"sem modelo" em `/saude`, e o front mostra a mensagem em vez de calcular.

Para conferir que a API está servindo de verdade:

```bash
uv run python scripts/conferir_api.py
```

Tudo junto, em container:

```bash
docker compose up dados tuning promover   # dado, busca, promoção
docker compose up app                     # sobe a API e o front
```

> **Treinou no host e a API em Docker diz "sem modelo"?** É esperado. O MLflow
> grava o caminho do artefato absoluto no banco, então um run criado em
> `<repo>/mlruns` não resolve em `/app/mlruns`. Escolha um mundo: ou tudo local,
> ou tudo em compose.

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

**Linux, macOS ou Git Bash:**

```bash
git clone https://github.com/PiettroNaka/prj_churn.git && cd prj_churn
uv sync
./scripts/mlflow_ui.sh
```

**Windows (PowerShell), sem depender de bash:**

```powershell
git clone https://github.com/PiettroNaka/prj_churn.git
cd prj_churn
uv sync
uv run python scripts/preparar.py
```

`scripts/preparar.py` é o mesmo caminho dos dois scripts `.sh`, em Python puro:
materializa o dataset (do DVC ou do histórico do Git), treina se não houver run
e sobe a UI. `--porta 5001` troca a porta; `--so-dado` para depois do CSV.

## Quando der errado

| Sintoma | O que é | Saída |
|---|---|---|
| `uv: command not found` | o instalador não entrou no PATH | feche e reabra o terminal, ou `source $HOME/.local/bin/env` |
| `./scripts/…` não é reconhecido (Windows) | PowerShell não roda `.sh` | `uv run python scripts/preparar.py`, ou abra o **Git Bash** |
| `cd caminho/para/prj_churn` falha | `caminho/para` era um exemplo, não um caminho real | use a pasta onde você clonou; se o prompt já termina em `prj_churn`, você já está nela |
| `You have not concluded your merge (MERGE_HEAD exists)` | um `git pull` anterior parou no meio e nunca terminou | **`atualizar.bat`** |
| `Your local changes would be overwritten by checkout` | há arquivo modificado na pasta barrando a troca de branch | **`atualizar.bat`** |
| `have diverged, and have N and M different commits each` | a branch foi reescrita no GitHub; o histórico daqui não existe mais lá | **`atualizar.bat`** |
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
