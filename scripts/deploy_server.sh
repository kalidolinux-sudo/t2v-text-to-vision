#!/usr/bin/env bash
# Deploy the T2V demo on your own Linux server (Ubuntu/Debian) with Docker.
#
#   curl -fsSL https://raw.githubusercontent.com/kalidolinux-sudo/t2v-text-to-vision/main/scripts/deploy_server.sh | sudo bash
#
# It installs Docker if needed, builds the image, asks for the AMD TokenFactory key (typed, never stored in
# the image or in git), and runs the app on port 80 with automatic restart. Re-run it to update.
# Optional: PORT=8080 (host port), DOMAIN=t2v.example.com (adds HTTPS through Caddy; point the domain's
# A record at this server first).
set -euo pipefail

REPO=https://github.com/kalidolinux-sudo/t2v-text-to-vision
DIR=/opt/t2v-text-to-vision
PORT=${PORT:-80}
DOMAIN=${DOMAIN:-}
ENV_FILE=/etc/t2v.env

if [ "$(id -u)" -ne 0 ]; then echo "run as root (sudo)"; exit 1; fi

echo "== 1/5 Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker >/dev/null 2>&1 || true

echo "== 2/5 Code"
if [ -d "$DIR/.git" ]; then git -C "$DIR" pull --ff-only; else
  command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
  git clone --depth 1 "$REPO" "$DIR"
fi

echo "== 3/5 TokenFactory key"
if [ ! -s "$ENV_FILE" ]; then
  KEY=${TOKENFACTORY_API_KEY:-}
  if [ -z "$KEY" ]; then read -r -s -p "AMD TokenFactory API key (input hidden, Enter to skip): " KEY < /dev/tty; echo; fi
  umask 077
  printf 'TOKENFACTORY_API_KEY=%s\n' "$KEY" > "$ENV_FILE"
fi

echo "== 4/5 Build image (first time: several minutes)"
docker build -t t2v "$DIR"

echo "== 5/5 Run"
docker rm -f t2v >/dev/null 2>&1 || true
if [ -n "$DOMAIN" ]; then
  docker network create t2v-net >/dev/null 2>&1 || true
  docker run -d --name t2v --restart unless-stopped --network t2v-net --env-file "$ENV_FILE" \
    -v t2v-checkpoints:/app/checkpoints -v t2v-hf:/root/.cache/huggingface t2v
  docker rm -f t2v-caddy >/dev/null 2>&1 || true
  docker run -d --name t2v-caddy --restart unless-stopped --network t2v-net -p 80:80 -p 443:443 \
    -v t2v-caddy:/data caddy:2 caddy reverse-proxy --from "$DOMAIN" --to t2v:7860
  URL="https://$DOMAIN"
else
  docker run -d --name t2v --restart unless-stopped -p "$PORT:7860" --env-file "$ENV_FILE" \
    -v t2v-checkpoints:/app/checkpoints -v t2v-hf:/root/.cache/huggingface t2v
  IP=$(curl -fsS https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')
  URL="http://$IP$([ "$PORT" = 80 ] || echo ":$PORT")"
fi

echo
echo "Starting. The first start downloads the models (~800 MB); follow it with:  docker logs -f t2v"
echo "Then open: $URL"
