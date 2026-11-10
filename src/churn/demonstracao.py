"""Modo demonstração — o front funcionando onde não há API nem Registry.

O desenho do projeto é claro: o front chama a API, a API carrega o campeão do
Registry. É o que separa as três coisas, e é o que permite trocar de modelo
movendo um alias.

Só que esse desenho pressupõe três processos e um banco de tracking. Num deploy
de vitrine — o Streamlit Community Cloud, por exemplo — existe **um** processo,
o repositório clonado, e nada mais: sem MLflow, sem API, e sem o CSV, que saiu
do Git no Encontro 5.

Este módulo é a saída para esse caso, e só para ele. São duas saídas, em ordem
de preferência:

    garantir_dataset()   materializa data/churn.csv, conferindo o md5
    subir_api_embutida() treina, promove e sobe a API AQUI, no mesmo container
    pipeline_local()     se nem isso der, treina e prevê sem API nenhuma

A primeira preserva o desenho: há API, há Registry, há alias, e o front continua
só perguntando. Um processo a mais no mesmo container é bem menos concessão do
que o front passar a carregar modelo.

A segunda é a rede de segurança. **Não é o caminho normal e não deve virar um.**
O front deixa à vista qual das duas respondeu — um número que sai de um modelo
treinado na hora não é o mesmo número que sai do campeão promovido, e quem olha
tem o direito de saber qual é qual.

O dado vem de onde ele sempre esteve: o CSV foi versionado pelo Git até o
Encontro 5, e o GitHub continua servindo aquele commit por HTTP. A disciplina do
DVC é mantida no que importa — os bytes só são aceitos se o md5 bater com o
ponteiro ``data/churn.csv.dvc``. Transporte diferente, mesma verificação.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from churn.config import Settings, settings
from churn.lineage import data_md5, ponteiro_de

# O ChurnPipeline é importado dentro de `pipeline_local`, não aqui: ele arrasta
# o mlflow junto, e quem só quer `garantir_dataset` — o scripts/preparar.py, por
# exemplo — não deveria pagar esse import para baixar um CSV.

#: O commit em que ``data/churn.csv`` ainda estava versionado pelo Git. É o
#: mesmo blob que o ``scripts/preparar.py`` recupera do histórico local; aqui
#: ele é buscado por HTTP, porque um clone raso não traz histórico.
COMMIT_COM_CSV = "049e49b130854881c096508a40b4bf6ae37b2596"

URL_DO_CSV = (
    f"https://raw.githubusercontent.com/PiettroNaka/prj_churn/{COMMIT_COM_CSV}"
    "/data/churn.csv"
)

ESPERA = 30


class DatasetIndisponivel(RuntimeError):
    """Não foi possível materializar o CSV, e a causa está na mensagem."""


def baixar(destino: Path, md5_esperado: str, url: str = URL_DO_CSV) -> None:
    """Baixa o CSV e **só grava se o md5 bater** com o ponteiro do DVC.

    A ordem importa: escreve num temporário, confere, e só então move. Gravar
    antes de conferir deixaria um arquivo errado no lugar certo — pior que
    nenhum arquivo, porque o próximo passo o usaria sem desconfiar.
    """
    destino.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=ESPERA) as resposta:
            conteudo = resposta.read()
    except OSError as erro:
        raise DatasetIndisponivel(f"não consegui baixar de {url}: {erro}") from erro

    obtido = hashlib.md5(conteudo).hexdigest()
    if obtido != md5_esperado:
        raise DatasetIndisponivel(
            f"o arquivo baixado não bate com o ponteiro do DVC "
            f"(esperado {md5_esperado[:10]}…, obtido {obtido[:10]}…)"
        )

    temporario = destino.with_suffix(destino.suffix + ".parcial")
    # bytes crus: nada de texto, ou o Windows troca LF por CRLF e o md5 muda
    temporario.write_bytes(conteudo)
    temporario.replace(destino)


def garantir_dataset(cfg: Settings = settings) -> Path:
    """Devolve o caminho do CSV, baixando-o se não estiver no disco."""
    caminho = Path(cfg.data_path)
    esperado = data_md5(ponteiro_de(caminho))

    if caminho.exists() and hashlib.md5(caminho.read_bytes()).hexdigest() == esperado:
        return caminho

    baixar(caminho, esperado)
    return caminho


def pipeline_local(cfg: Settings = settings) -> tuple["ChurnPipeline", dict[str, float]]:
    """Treina na hora e devolve o pipeline pronto para prever, com as métricas.

    Mesmo dado, mesmo split, mesma semente e mesmos hiperparâmetros do
    ``churn.model`` — o que falta aqui é só o registro. Então o modelo é o
    mesmo; o que não existe é o endereço estável dele.
    """
    from churn.model import ajustar
    from churn.serving import ChurnPipeline
    from churn.tuning import VENCEDOR

    garantir_dataset(cfg)
    # a configuração que a busca encontrou, e não o padrão: a vitrine não pode
    # mostrar um modelo pior do que o que o Registry promoveria
    modelo, encoder, metricas = ajustar(cfg.model_copy(update=VENCEDOR))
    pipeline = ChurnPipeline(
        modelo, encoder, cfg.id_column, cfg.target, cfg.positive_label
    )
    return pipeline, metricas


# --------------------------------------------------------------------------
# a API dentro do mesmo processo — para plataformas de um container só
# --------------------------------------------------------------------------


def _responde(url: str, espera: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=espera):
            return True
    except (OSError, ValueError):
        return False


def garantir_campeao(cfg: Settings = settings) -> str:
    """Treina e promove, se ainda não houver um ``@champion``. Devolve a versão.

    Idempotente de propósito: numa plataforma que reinicia o processo sozinha,
    isto roda de novo a cada subida, e treinar duas vezes o mesmo modelo seria
    desperdício puro.
    """
    from churn.model import train
    from churn.registry import promover, versao_do_alias
    from churn.tuning import VENCEDOR

    if (versao := versao_do_alias(cfg)) is not None:
        return str(versao.version)

    garantir_dataset(cfg)
    # a configuração que a busca encontrou — a mesma que o Registry promoveria
    train(cfg.model_copy(update=VENCEDOR))
    return str(promover(cfg))


def subir_api_embutida(
    porta: int = 8000, cfg: Settings = settings, espera: int = 90
) -> subprocess.Popen | None:
    """Sobe a API neste mesmo container e devolve o processo.

    Para o Streamlit Community Cloud, que roda **um** processo: em vez de o
    front passar a carregar modelo — o que apagaria a fronteira que o projeto
    inteiro defende —, a API sobe ao lado dele, em 127.0.0.1, e o front
    continua apenas perguntando.

    Devolve ``None`` se não deu, em vez de levantar: quem chama tem a
    :func:`pipeline_local` como rede de segurança, e derrubar a tela inteira
    porque a API não subiu seria trocar um problema por um pior.
    """
    try:
        garantir_campeao(cfg)
    except Exception:  # noqa: BLE001 — sem campeão não há API para subir
        return None

    processo = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "churn.api:app",
            "--host", "127.0.0.1", "--port", str(porta),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    alvo = f"http://127.0.0.1:{porta}/saude"
    for _ in range(espera):
        if _responde(alvo):
            return processo
        if processo.poll() is not None:
            return None
        time.sleep(1)

    processo.terminate()
    return None
