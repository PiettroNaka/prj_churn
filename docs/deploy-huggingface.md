# Publicar no Hugging Face Spaces

> **Atenção: isto deixou de ser gratuito.** O Hugging Face passou a exigir plano
> pago (PRO ou créditos) para Spaces com Docker ou Gradio; só os **Static
> Spaces** continuam livres, e eles não rodam Python. Este guia fica porque a
> imagem funciona e é testada no CI a cada push — mas para um deploy sem custo,
> o caminho é o **[Streamlit Community Cloud](../README.md#publicar-de-graça-com-a-api-junto)**,
> que roda a mesma pilha completa.

Deploy com URL pública rodando a **pilha inteira**, não só o front.

```
você  →  Streamlit (porta 7860)  →  FastAPI (127.0.0.1:8000)  →  modelo @champion
                        └──────── o mesmo container ────────┘
```

O desenho do projeto é front → API → Registry, e ele é mantido aqui. O Spaces
expõe uma porta só, então os dois processos dividem um container em vez de o
front passar a carregar modelo — o que descaracterizaria o projeto.

> Diferente do Streamlit Community Cloud, aqui **não** é modo demonstração: a
> API está no ar e serve o campeão promovido. A barra lateral do front vai dizer
> "API · modelo v1".

## O que você precisa

- Uma conta no [huggingface.co](https://huggingface.co) (gratuita, sem cartão).
- Nada instalado: dá para fazer tudo pelo navegador.

## Passo a passo

### 1. Criar o Space

Em **huggingface.co → seu perfil → New Space**:

| campo | valor |
|---|---|
| Space name | `prj-churn` (ou o que preferir) |
| License | MIT |
| **Space SDK** | **Docker** → *Blank* |
| Hardware | CPU basic (gratuito) |
| Visibility | Public |

### 2. Pôr os dois arquivos

O repositório do Space precisa de **dois arquivos**, e os dois estão aqui em
`deploy/spaces/`:

| arquivo | o que é |
|---|---|
| `Dockerfile` | constrói a imagem: clona este repositório, instala, treina e promove |
| `README.md` | o cabeçalho que o Spaces lê (`sdk: docker`, `app_port: 7860`) |

Pelo navegador, no seu Space: **Files → Add file → Create a new file**, uma vez
para cada um, colando o conteúdo.

O `README.md` do Space **substitui** o que o Spaces cria sozinho. Ele precisa
começar com o bloco entre `---`; é dali que a plataforma lê a porta.

### 3. Esperar a construção

A aba **Logs** mostra o progresso. A primeira construção leva alguns minutos:
ela clona o projeto, instala 105 pacotes, baixa o dataset, treina o modelo e o
promove no Registry.

Quando terminar, o Space abre sozinho na página **Início**.

## Por que o código vem por clone

O `Dockerfile` do Space não copia arquivos — ele faz `git clone` deste
repositório. Duas consequências, e as duas são boas aqui:

- o repositório do Space fica com **dois arquivos**, em vez de uma cópia do
  projeto inteiro que envelheceria em paralelo;
- a fonte da verdade continua sendo um lugar só.

O preço: o Space **não** acompanha os seus pushes sozinho. Para atualizar, use
**Settings → Factory rebuild** no Space.

Para publicar de outra branch, mude o `ARG REF=main` no `Dockerfile`.

## O modelo é treinado na construção, não na subida

```dockerfile
RUN ... python scripts/preparar.py --so-dado \
    && ... python -m churn.model \
    && ... python -m churn.registry
```

São uns 20 segundos que ninguém precisa esperar olhando uma tela branca. O Space
nasce com `churn v1 @champion` pronto, e a API responde na primeira requisição.

A configuração treinada é a que a busca encontrou (`churn.tuning.VENCEDOR`),
fixada em variáveis `CHURN_*` no `Dockerfile` — `roc_auc` 0,8206 e `recall`
0,7609.

O dataset não está no Git (Encontro 5) e o clone é raso, então o histórico
também não vem. Ele é buscado do commit em que ainda era versionado e **só é
aceito se o md5 bater com o ponteiro `.dvc`** — transporte diferente, mesma
verificação.

## Quando der errado

| Sintoma | O que é | Saída |
|---|---|---|
| "No application file" ou o Space não inicia | falta o cabeçalho `---` no `README.md` do Space, ou `sdk: docker` | recrie o `README.md` com o bloco inteiro, incluindo as duas linhas de `---` |
| o build falha em `git clone` | o repositório está privado | deixe-o público, ou troque o `ARG REPO` por uma URL com token |
| o build falha ao escrever | a plataforma roda como usuário 1000 | o `Dockerfile` já cria esse usuário; confira que o `USER usuario` não foi removido |
| a página abre mas a barra diz "Modo demonstração" | a API interna não subiu | **Logs** do Space; `scripts/subir_front.py` anuncia cada passo |
| mudou o código e o Space não mudou | ele não acompanha pushes | **Settings → Factory rebuild** |

## O que o CI garante

O job `spaces` do `.github/workflows/docker.yml` constrói esta imagem a cada
push, sobe o container, espera o front responder e roda
`scripts/conferir_api.py` **dentro dele** — provando que a API interna está no
ar e servindo o campeão promovido, não um modelo treinado na hora.

Se o Space falhar, o CI falha antes.
