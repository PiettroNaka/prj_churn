# Deploy em AWS EC2

O arranjo pedido: **dois containers numa instância EC2** — um com a aplicação
(front + API), outro com o Postgres. O dataset deixa de ser arquivo e passa a
ser tabela.

```
  ┌─ instância EC2 ─────────────────────────────────────────────┐
  │                                                             │
  │   ┌─ web ───────────────┐          ┌─ postgres ──────────┐  │
  │   │  Streamlit  :8501   │ ───────▶ │                     │  │
  │   │  FastAPI    :8000   │ ◀─────── │   tabela clientes   │  │
  │   │  o CSV, só na carga │ ──carga─▶│   7.043 linhas      │  │
  │   └─────────────────────┘          └─────────────────────┘  │
  │          ▲                           sem porta publicada    │
  └──────────┼──────────────────────────────────────────────────┘
             │ :8501
       o navegador
```

Só o front sai para a internet. A API responde em `127.0.0.1` dentro do
container, e o banco só existe na rede interna do compose — ele não tem porta
publicada, então nem o grupo de segurança precisa protegê-lo.

> **Antes de começar:** a AWS é nuvem paga. Leia a seção
> [Quanto custa, e como não pagar demais](#quanto-custa-e-como-não-pagar-demais)
> **antes** de criar a instância, não depois.

---

## O caminho curto: deixar a instância fazer tudo sozinha

Há dois caminhos. Este é o recomendado — você não precisa entrar por SSH para
nada.

A AWS tem um campo **User data** na tela de criar a instância; o que estiver
nele roda como root no primeiro boot. O arquivo
[`deploy/ec2/user-data.sh`](../deploy/ec2/user-data.sh) faz a sequência inteira:
espera o `apt` liberar, instala o Docker, cria swap **se** a instância tiver
pouca memória, clona o repositório, gera a senha do banco e sobe os dois
containers — conferindo no fim que a aplicação ficou saudável e que as 7.043
linhas entraram na tabela.

1. **Launch instance** → Ubuntu Server 24.04 LTS, `t3.small` (ou `t3.micro` do
   free tier), disco **20 GB**.
2. Grupo de segurança: libere a **8501** para `0.0.0.0/0` e o **22** só para o
   seu IP (detalhes no passo 2 abaixo).
3. **Advanced details** → role até **User data** → cole o conteúdo de
   `deploy/ec2/user-data.sh`.
4. **Launch**. Espere de 8 a 20 minutos conforme a instância.
5. Abra **`http://<IP-PÚBLICO>:8501`**.

Para acompanhar, se quiser (não é necessário):

```bash
ssh -i sua-chave.pem ubuntu@<IP>
cat /var/log/churn-estado        # em que passo está; a última linha é a atual
sudo tail -f /var/log/churn-deploy.log
```

O estado termina em `PRONTO` quando deu certo, ou numa linha começando com
`FALHOU:` dizendo o quê. Se falhar, o resto do caminho longo abaixo serve para
consertar à mão.

> O script não tem segredo nenhum dentro dele: a senha do banco é **gerada na
> instância** e fica só lá, em `/opt/prj_churn/.env` com permissão 600.

---

## O caminho longo: passo a passo manual

Use este se o User data falhar, ou se quiser ver cada peça entrando no lugar.


## 1. Escolher a instância

| Tipo | RAM | Serve? |
|---|---|---|
| `t2.micro` / `t3.micro` | 1 GB | **só com swap** — ver passo 4. O build instala Streamlit, MLflow e scikit-learn, e 1 GB não aguenta sem ajuda |
| `t3.small` | 2 GB | **sim**, é o que eu recomendo. Fora do free tier: ~US$ 15/mês ligada |
| `t3.medium` | 4 GB | folgado, ~US$ 30/mês |

O *free tier* dá 750 horas/mês de `t2.micro` ou `t3.micro` nos primeiros 12
meses — uma instância ligada o mês inteiro cabe nisso. Se o seu objetivo é
apresentar e desligar, `t3.micro` com swap resolve e não custa nada.

**AMI:** Ubuntu Server 24.04 LTS (x86_64).
**Disco:** 20 GB gp3 — o padrão de 8 GB fica apertado com as imagens Docker.

## 2. Grupo de segurança

Duas regras de entrada, e só:

| Tipo | Porta | Origem | Para quê |
|---|---|---|---|
| SSH | 22 | **o seu IP** (`Meu IP` no console) | você entrar na máquina |
| TCP personalizado | 8501 | `0.0.0.0/0` | o professor abrir o front |

Não abra a 5432. O banco não publica porta nenhuma; abri-la no grupo de
segurança não o tornaria acessível, só daria a impressão de que é para estar.

Deixar o SSH em `0.0.0.0/0` é o erro clássico: põe o seu IP.

## 3. Entrar na máquina

```bash
chmod 400 ~/Downloads/sua-chave.pem
ssh -i ~/Downloads/sua-chave.pem ubuntu@<IP-PÚBLICO-DA-INSTÂNCIA>
```

No Windows, o PowerShell já tem `ssh`. O `chmod` não existe lá; se ele reclamar
da permissão da chave, use o WSL ou ajuste as permissões pelas propriedades do
arquivo.

## 4. Swap — só se a instância tiver 1 GB

Pule este passo numa `t3.small` ou maior.

```bash
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab   # sobrevive ao reboot
free -h                                                       # confere
```

Sem isto, o build morre no meio da instalação das dependências com um
`Killed` — que é o kernel matando o processo por falta de memória, e não diz
isso com todas as letras.

## 5. Instalar o Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker ubuntu
newgrp docker          # aplica o grupo sem precisar sair e entrar de novo
docker --version && docker compose version
```

## 6. Clonar e configurar

```bash
git clone https://github.com/PiettroNaka/prj_churn.git
cd prj_churn
```

A senha do banco vai num `.env` ao lado do compose. Ela **não** vai para o Git —
o `.gitignore` já cobre, e o compose recusa subir sem ela, de propósito:

```bash
printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -base64 24)" > .env
cat .env        # anote; é com ela que você entra no banco depois
```

## 7. Subir

```bash
docker compose -f compose.ec2.yml up -d --build
```

O build leva de 5 a 15 minutos conforme a instância — é a primeira vez que ela
baixa tudo. Depois disso, a aplicação ainda faz a sequência de partida sozinha:

1. materializa o CSV (tenta o DVC, cai para HTTP conferindo o md5)
2. carrega as 7.043 linhas no Postgres, **validando pelo contrato antes**
3. treina e promove um campeão, se ainda não houver (~2 min)
4. sobe a API, espera ela responder **com modelo carregado**, e então o front

Acompanhe:

```bash
docker compose -f compose.ec2.yml logs -f web
```

## 8. Conferir

```bash
# os dois containers, e a saúde deles
docker compose -f compose.ec2.yml ps

# o dataset está mesmo no banco?
docker compose -f compose.ec2.yml exec postgres \
  psql -U churn -d churn -c "SELECT count(*) FROM clientes;"
#  count
#  -------
#    7043

# a API está servindo o campeão promovido?
docker compose -f compose.ec2.yml exec web \
  /app/.venv/bin/python -c "import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:8000/modelo').read().decode())"
```

E no navegador: **`http://<IP-PÚBLICO>:8501`**

É `http`, não `https` — não há certificado aqui. O navegador vai avisar, e está
correto avisar; ver as [Ressalvas](#ressalvas) no fim.

## 9. Atualizar depois de um push

```bash
cd prj_churn
git pull
docker compose -f compose.ec2.yml up -d --build
```

O volume do banco sobrevive: a tabela continua lá e a carga não repete.

---

## Quanto custa, e como não pagar demais

A cobrança é **por hora ligada**, não por uso. Uma instância esquecida ligada
cobra o mês inteiro.

```bash
# parar: para de cobrar a instância, o disco continua (centavos/mês)
#        → faça isto quando terminar de apresentar
sudo poweroff        # ou pelo console: Instance state → Stop instance

# religar depois: pelo console → Start instance
#   ATENÇÃO: o IP público MUDA a cada parada/partida, a menos que você
#   associe um Elastic IP. Um Elastic IP associado a uma instância ligada é
#   grátis; associado a uma instância PARADA, é cobrado.
```

Para encerrar de vez: **Terminate instance** e, na aba Volumes, confira se o
disco foi junto. E crie um alerta de orçamento em Billing → Budgets, com valor
baixo — é a rede de segurança que avisa antes da fatura.

---

## Problemas comuns

**O build morre com `Killed`.** É falta de memória. Volte ao passo 4 e crie o
swap.

**`docker: permission denied`.** O `usermod` não foi aplicado à sessão atual.
Rode `newgrp docker`, ou saia e entre de novo no SSH.

**O container `web` reinicia num laço.** Veja o log:
`docker compose -f compose.ec2.yml logs web`. As duas causas prováveis são
memória (volte ao passo 4) e o banco não ter ficado pronto — o `depends_on` já
espera o healthcheck, e o código ainda espera a conexão por até 30s, então isto
é raro.

**`required variable POSTGRES_PASSWORD is missing`.** Falta o `.env` do passo 6.
É proposital: melhor recusar do que subir com uma senha que todo mundo conhece.

**A página não abre, mas `docker compose ps` diz `healthy`.** É o grupo de
segurança: falta a regra de entrada na 8501 (passo 2).

**O Streamlit abre e diz "Modo demonstração".** A API não subiu dentro do
container. O front cai para treinar na sessão em vez de quebrar, e avisa na
barra lateral — mas aí não está passando pelo Registry. Veja o log do `web`.

---

## Ressalvas

Dito antes que alguém descubra sozinho:

- **Não há HTTPS.** O tráfego vai em claro, incluindo o que for digitado no
  formulário. Para um trabalho de curso com dado público, tudo bem; para
  qualquer outra coisa, precisaria de um proxy com certificado (Caddy ou nginx
  com Let's Encrypt) e de um domínio.
- **Não há autenticação.** Quem tiver o IP e a porta usa o front e a API.
- **O backup do banco não existe.** O volume sobrevive a `down` e a reboot, mas
  não a um `down -v` nem à destruição da instância. Aqui isso é aceitável: o
  dataset é recarregável a partir do CSV, conferido pelo md5. Com dado que não
  se recupera, faltaria `pg_dump` agendado.
- **Uma instância, sem redundância.** Se ela cair, o serviço cai.
- **O MLflow continua em SQLite num volume**, não num servidor de tracking.
  Serve a uma instância; não serviria a um time.
- **A senha do banco fica em texto no `.env` da instância.** O certo seria o
  AWS Secrets Manager; para um trabalho de curso, o `.env` com permissão
  restrita é a escolha consciente.
