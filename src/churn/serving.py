"""O modelo que roda com um cliente de verdade.

O estimador sozinho não serve para nada fora do notebook: ele foi treinado com
features já derivadas e categóricas já codificadas, e não faz ideia do que fazer
com uma linha como ela sai do sistema de origem — com ``customerID``, com
``TotalCharges`` em texto, sem ``gasto_por_mes``.

    >>> modelo.predict(cliente_cru)
    ValueError: The feature names should match those that were passed during fit.

:class:`ChurnPipeline` fecha esse buraco: limpeza, features, encoder e estimador
num objeto só, logado como ``pyfunc``. É o que torna ``mlflow models serve``
utilizável — e é o mesmo objeto que o Encontro 7 vai promover no Registry.
"""
from __future__ import annotations

from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin

import churn
from churn.config import Settings
from churn.data import coerce_total_charges
from churn.features import CategoricalEncoder, add_derived_features
from churn.schema import SIM_NAO

#: Nome do modelo dentro do run. ``runs:/<run_id>/model`` e ``models:/<id>``
#: apontam para cá.
NOME_DO_MODELO = "model"


class ChurnPipeline(mlflow.pyfunc.PythonModel):
    """Do cliente cru à previsão, no mesmo caminho que o treino usou.

    A ordem importa e é a mesma do ``data.py``/``features.py``: derrubar o
    identificador, coagir ``TotalCharges``, derivar ``gasto_por_mes`` e só então
    codificar. Qualquer divergência aqui é *training/serving skew* — o modelo
    veria em produção um dado diferente do que viu no treino.
    """

    def __init__(
        self,
        modelo: ClassifierMixin,
        encoder: CategoricalEncoder,
        id_column: str,
        target: str,
        positive_label: str,
    ) -> None:
        self._modelo = modelo
        self._encoder = encoder
        self._id_column = id_column
        self._target = target
        self._positive_label = positive_label
        # o rótulo negativo sai do domínio declarado no contrato, não de um
        # "não " colado na frente do positivo: a saída fala o vocabulário do dado
        self._negative_label = next(v for v in SIM_NAO if v != positive_label)

    def entrada_de_servico(self, entrada: pd.DataFrame) -> pd.DataFrame:
        """A linha como ela chega em produção: sem identificador e sem alvo.

        As duas colunas são *toleradas* — quem chama manda a linha que tem, não
        a que o modelo gostaria de receber — mas nenhuma das duas é entrada. O
        ``customerID`` não é feature, e o ``Churn`` é o que se quer descobrir.
        """
        descartaveis = [
            c for c in (self._id_column, self._target) if c in entrada.columns
        ]
        return entrada.drop(columns=descartaveis)

    def preparar(self, entrada: pd.DataFrame) -> pd.DataFrame:
        """Põe a linha crua no formato que o estimador espera."""
        df = self.entrada_de_servico(entrada.copy())
        return self._encoder.transform(add_derived_features(coerce_total_charges(df)))

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        """Devolve o rótulo e a probabilidade de churn, uma linha por cliente."""
        preparado = self.preparar(model_input)
        rotulos = self._modelo.predict(preparado)
        probabilidades = self._modelo.predict_proba(preparado)[:, 1]
        return pd.DataFrame(
            {
                "churn": [
                    self._positive_label if r == 1 else self._negative_label
                    for r in rotulos
                ],
                "probabilidade": probabilidades,
            },
            index=model_input.index,
        )


def log_pipeline(
    modelo: ClassifierMixin,
    encoder: CategoricalEncoder,
    cfg: Settings,
    exemplo_cru: pd.DataFrame,
) -> mlflow.models.model.ModelInfo:
    """Loga o pipeline no run corrente, pronto para carregar e para servir.

    A assinatura sai do exemplo **sem identificador e sem alvo**, não do CSV
    cru inteiro. A diferença não é cosmética: o MLflow trata a assinatura como
    contrato e exige toda coluna declarada. Inferir do CSV inteiro produzia um
    modelo que recusava a chamada com ``Model is missing inputs ['customerID',
    'Churn']`` — isto é, para prever o churn era preciso informar o churn. O
    pipeline sempre descartou as duas; só a assinatura não sabia disso.
    """
    pipeline = ChurnPipeline(
        modelo, encoder, cfg.id_column, cfg.target, cfg.positive_label
    )
    exemplo = pipeline.entrada_de_servico(exemplo_cru)
    return mlflow.pyfunc.log_model(
        name=NOME_DO_MODELO,
        python_model=pipeline,
        signature=infer_signature(exemplo, pipeline.predict(None, exemplo)),
        input_example=exemplo,
        # o pacote vai dentro do artefato: sem isto o modelo só carrega onde o
        # projeto já estiver instalado, que é o oposto de "servir em qualquer
        # lugar"
        code_paths=[str(Path(churn.__file__).parent)],
    )


def carregar(uri: str) -> mlflow.pyfunc.PyFuncModel:
    """Carrega o pipeline de um ``runs:/...`` ou ``models:/...``."""
    return mlflow.pyfunc.load_model(uri)
