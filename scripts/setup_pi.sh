#!/usr/bin/env bash
# One-command setup for Raspberry Pi 5 (4 GB) / Raspberry Pi OS Bookworm (64-bit),
# or any Debian/Ubuntu machine. Run from the project folder:
#
#   bash scripts/setup_pi.sh                  base install (online voice mode works after this)
#   bash scripts/setup_pi.sh --models         + offline speech models and voices (~330 MB)
#   bash scripts/setup_pi.sh --llm            + local AI: Ollama + qwen2.5:1.5b (~1 GB)
#   bash scripts/setup_pi.sh --service te     + start Aria hands-free at boot (language en|te|hi)
#   bash scripts/setup_pi.sh --all te         everything
#
# Safe to re-run: each step skips work that's already done.
set -euo pipefail

WITH_MODELS=0; WITH_LLM=0; SERVICE_LANG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --models)  WITH_MODELS=1 ;;
    --llm)     WITH_LLM=1 ;;
    --service|--all)
      [[ "$1" == "--all" ]] && { WITH_MODELS=1; WITH_LLM=1; }
      SERVICE_LANG="en"
      # The language is optional; only consume the next argument if it is one.
      if [[ "${2:-}" =~ ^(en|te|hi)$ ]]; then SERVICE_LANG="$2"; shift; fi ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 1 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."
PROJECT_DIR="$(pwd)"
step() { echo; echo "==> $*"; }

step "Checking the machine"
ARCH="$(uname -m)"
MEM_MB=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)
echo "architecture: $ARCH, RAM: ${MEM_MB} MB"
[[ "$ARCH" == "aarch64" || "$ARCH" == "x86_64" ]] || echo "warning: untested architecture $ARCH (need 64-bit OS)"
command -v python3 >/dev/null || { echo "python3 not found"; exit 1; }
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
  || { echo "Python 3.10+ required (found $(python3 --version))"; exit 1; }

step "System packages (audio, mp3 playback, fallback voice)"
sudo apt-get update -qq
sudo apt-get install -y -qq python3-venv python3-dev libportaudio2 mpg123 espeak-ng curl

step "Python environment"
[[ -d venv ]] || python3 -m venv venv
# shellcheck disable=SC1091
. venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

step "Self-test"
python -m pytest -q tests
python -m tools.evaluate | head -14

if [[ $WITH_MODELS -eq 1 ]]; then
  step "Offline speech models and voices"
  python -m tools.download_models
fi

if [[ $WITH_LLM -eq 1 ]]; then
  step "Local AI (Ollama)"
  if (( MEM_MB < 3500 )); then
    echo "only ${MEM_MB} MB RAM - a 1.5B model needs ~1.1 GB on top of the bot; using qwen2.5:0.5b instead"
    MODEL="qwen2.5:0.5b"
    grep -q '^OLLAMA_MODEL=' .env 2>/dev/null || echo "OLLAMA_MODEL=$MODEL" >> .env
  else
    MODEL="qwen2.5:1.5b"
  fi
  if ! command -v ollama >/dev/null; then
    # Ollama's official installer (https://ollama.com/download/linux); runs CPU-only on a Pi.
    curl -fsSL https://ollama.com/install.sh | sh
  fi
  ollama pull "$MODEL"
fi

if [[ -n "$SERVICE_LANG" ]]; then
  step "Autostart service (hands-free, language: $SERVICE_LANG)"
  sudo tee /etc/systemd/system/aria.service >/dev/null <<EOF
[Unit]
Description=Aria customer support voice bot
After=network-online.target sound.target
Wants=network-online.target

[Service]
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/python main.py --hands-free --lang $SERVICE_LANG
# "bye" ends a customer's session; restarting greets the next customer.
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now aria.service
  echo "service running - logs: journalctl -u aria -f   stop: sudo systemctl stop aria"
fi

step "Done"
python -m tools.download_models --status
cat <<EOF

Try it:
  . venv/bin/activate
  python main.py                        # text chat
  python main.py --mode voice           # voice (press Enter, then speak)
  python main.py --hands-free --lang te # voice, no keyboard needed
  python -m asr.speech_to_text --levels # check the microphone
EOF
