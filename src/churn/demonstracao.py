"""Modo demonstração — o front funcionando onde não há API nem Registry.

O desenho do projeto é claro: o front chama a API, a API carrega o campeão do
Registry. É o que separa as três coisas, e é o que permite trocar de modelo
movendo um alias.

Só que esse desenho pressupõe três processos e um banco de tracking. Num deploy
de vitrine — o Streamlit Community Cloud, por exemplo — existe **um** processo,
o repositório clonado, e nada mais: sem MLflow, sem API, e sem o CSV, que saiu
do Git no Encontro 5.

Este módulo é a saída para esse caso, e só para ele:

    garantir_dataset()   materializa data/churn.csv, conferindo o md5
    pipeline_local()     treina e devolve o ChurnPipeline, sem MLflow

**Não é o caminho normal e não deve virar um.** Quando há API, o front fala com
ela; este módulo só entra quando não há. O front deixa isso à vista em vez de
esconder — um número que sai de um modelo treinado na hora não é o mesmo número
que sai do campeão promovido, e quem olha tem o direito de saber qual é qual.

O dado vem de onde ele sempre esteve: o CSV foi versionado pelo Git até o
Encontro 5, e o GitHub continua servindo aquele commit por HTTP. A disciplina do
DVC é mantida no que importa — os bytes só são aceitos se o md5 bater com o
ponteiro ``data/churn.csv.dvc``. Transporte diferente, mesma verificação.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from churn.config import Settings, settings
from churn.evaluate import evaluate  # noqa: F401  (reexport por conveniência)
from churn.lineage import data_md5, ponteiro_de
from churn.serving import ChurnPipeline

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


def pipeline_local(cfg: Settings = settings) -> tuple[ChurnPipeline, dict[str, float]]:
    """Treina na hora e devolve o pipeline pronto para prever, com as métricas.

    Mesmo dado, mesmo split, mesma semente e mesmos hiperparâmetros do
    ``churn.model`` — o que falta aqui é só o registro. Então o modelo é o
    mesmo; o que não existe é o endereço estável dele.
    """
    from churn.model import ajustar
    from churn.tuning import VENCEDOR

    garantir_dataset(cfg)
    # a configuração que a busca encontrou, e não o padrão: a vitrine não pode
    # mostrar um modelo pior do que o que o Registry promoveria
    modelo, encoder, metricas = ajustar(cfg.model_copy(update=VENCEDOR))
    pipeline = ChurnPipeline(
        modelo, encoder, cfg.id_column, cfg.target, cfg.positive_label
    )
    return pipeline, metricas
