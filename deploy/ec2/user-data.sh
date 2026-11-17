#!/bin/bash
# Sobe o projeto inteiro numa EC2 nova, sozinho.
#
# COMO USAR: na tela de criar a instância, em "Advanced details" (Detalhes
# avançados), cole este arquivo inteiro no campo "User data". A AWS o executa
# como root no primeiro boot, e no fim o front está no ar na porta 8501.
#
# Não é preciso entrar por SSH para nada disto. O SSH serve para acompanhar ou
# para consertar se algo falhar — ver o fim deste arquivo.
#
# Requisitos da instância: Ubuntu Server 24.04 LTS, disco de 20 GB, e o grupo
# de segurança liberando a porta 8501.
#
# O progresso fica em /var/log/churn-deploy.log e o estado em
# /var/log/churn-estado (uma palavra por linha, a última é onde ele está).

set -o pipefail
exec > >(tee -a /var/log/churn-deploy.log) 2>&1

REPO="https://github.com/PiettroNaka/prj_churn.git"
DESTINO="/opt/prj_churn"
ESTADO="/var/log/churn-estado"

marco() { echo "$1" >> "$ESTADO"; echo "=== [$(date -Is)] $1 ==="; }

desisto() {
  marco "FALHOU: $1"
  echo "Veja /var/log/churn-deploy.log. O log do cloud-init tem o resto:"
  echo "  sudo tail -50 /var/log/cloud-init-output.log"
  exit 1
}

marco "comecando"

# ---------------------------------------------------------------- 1. apt
# No primeiro boot o unattended-upgrades costuma estar segurando o lock do
# apt. Sem esperar, o install falha com "Could not get lock" — que parece erro
# de rede e não é.
marco "esperando o apt ficar livre"
for _ in $(seq 1 60); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  sleep 5
done

marco "instalando docker e git"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y || desisto "apt-get update"
apt-get install -y docker.io docker-compose-v2 git openssl \
  || desisto "nao consegui instalar docker/git"
systemctl enable --now docker || desisto "o docker nao subiu"

# quem entrar por SSH depois usa docker sem sudo
usermod -aG docker ubuntu 2>/dev/null || true

# ---------------------------------------------------------------- 2. swap
# A imagem instala Streamlit, MLflow e scikit-learn. Numa instância de 1 GB
# (t2.micro/t3.micro do free tier) o build morre com "Killed", que é o kernel
# matando o processo por falta de memória — e não diz isso com todas as letras.
memoria_mb=$(free -m | awk '/^Mem:/{print $2}')
echo "memória da instância: ${memoria_mb} MB"
if [ "$memoria_mb" -lt 1900 ] && [ ! -f /swapfile ]; then
  marco "memoria curta: criando 2 GB de swap"
  fallocate -l 2G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=2048
  chmod 600 /swapfile
  mkswap /swapfile && swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  free -h
else
  echo "memória suficiente (ou swap já existe): seguindo sem criar swap"
fi

# ---------------------------------------------------------------- 3. código
marco "clonando o repositorio"
rm -rf "$DESTINO"
git clone --depth 1 "$REPO" "$DESTINO" || desisto "o clone falhou"
cd "$DESTINO" || desisto "nao achei $DESTINO"

# ---------------------------------------------------------------- 4. senha
# O compose recusa subir sem ela, de propósito. Gerada aqui e guardada só no
# disco da instância, com permissão restrita.
marco "gerando a senha do banco"
if [ ! -f .env ]; then
  printf 'POSTGRES_PASSWORD=%s\n' "$(openssl rand -base64 24 | tr -d '/+=')" > .env
  chmod 600 .env
fi

# ---------------------------------------------------------------- 5. subir
# O build leva de 5 a 15 minutos conforme a instância. Depois dele a aplicação
# ainda materializa o CSV, carrega o Postgres validando pelo contrato, treina e
# promove um campeão — tudo sozinho, antes de o front abrir.
marco "construindo e subindo os dois containers"
docker compose -f compose.ec2.yml up -d --build || desisto "o compose nao subiu"

marco "esperando a aplicacao ficar saudavel"
for _ in $(seq 1 90); do
  cid=$(docker compose -f compose.ec2.yml ps -q web 2>/dev/null)
  if [ -n "$cid" ]; then
    saude=$(docker inspect --format '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo partindo)
    [ "$saude" = "healthy" ] && break
    [ "$saude" = "unhealthy" ] && desisto "a aplicacao subiu mas nao ficou saudavel"
  fi
  sleep 20
done

saude=$(docker inspect --format '{{.State.Health.Status}}' \
  "$(docker compose -f compose.ec2.yml ps -q web)" 2>/dev/null || echo desconhecida)
[ "$saude" = "healthy" ] || desisto "passou do tempo e a saude ficou '$saude'"

# ---------------------------------------------------------------- 6. conferir
marco "conferindo"
docker compose -f compose.ec2.yml ps
linhas=$(docker compose -f compose.ec2.yml exec -T postgres \
  psql -U churn -d churn -tAc "SELECT count(*) FROM clientes" 2>/dev/null | tr -d ' ')
echo "linhas na tabela clientes: ${linhas:-?}"
[ "$linhas" = "7043" ] || echo "AVISO: esperava 7043 linhas; veio '${linhas:-vazio}'"

# O IP vem do serviço de metadados. Precisa de token: as instâncias novas vêm
# com IMDSv2 obrigatório, e a consulta antiga (um GET simples) volta vazia sem
# erro nenhum — foi o que deixou esta mensagem sair como "http://:80".
token=$(curl -sX PUT --max-time 5 "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null)
ip=$(curl -s --max-time 5 -H "X-aws-ec2-metadata-token: ${token}" \
  "http://169.254.169.254/latest/meta-data/public-ipv4" 2>/dev/null)
[ -n "$ip" ] || ip="<IP-DA-INSTANCIA>"

porta="${PORTA_PUBLICA:-80}"
endereco="http://${ip}"
[ "$porta" = "80" ] || endereco="http://${ip}:${porta}"

marco "PRONTO"
echo
echo "   O front está no ar em:  ${endereco}"
echo
echo "   Se não abrir, falta liberar a porta ${porta} no grupo de segurança."

# ---------------------------------------------------------------------------
# PARA ACOMPANHAR OU DEPURAR, por SSH:
#
#   sudo tail -f /var/log/churn-deploy.log     # este script, ao vivo
#   cat /var/log/churn-estado                  # em que passo ele está/parou
#   cd /opt/prj_churn && docker compose -f compose.ec2.yml logs -f web
#
# PARA ATUALIZAR depois de um push:
#   cd /opt/prj_churn && sudo git pull && sudo docker compose -f compose.ec2.yml up -d --build
#
# LEMBRE DE DESLIGAR a instância quando terminar: a AWS cobra por hora ligada.
