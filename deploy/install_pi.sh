#!/usr/bin/env bash
# Install Samantha as a service on a Raspberry Pi (64-bit Raspberry Pi OS / Debian).
# Run from the repo root:  bash deploy/install_pi.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP_USER="$(id -un)"
APP_UID="$(id -u)"

echo "Installing system packages (needs sudo)..."
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev libportaudio2 libatlas-base-dev ffmpeg libgl1

echo "Creating virtual environment in $APP_DIR/Samantha.venv..."
python3 -m venv "$APP_DIR/Samantha.venv"
"$APP_DIR/Samantha.venv/bin/python" -m pip install --upgrade pip
# CPU-only torch keeps the download small on a Pi
"$APP_DIR/Samantha.venv/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$APP_DIR/Samantha.venv/bin/python" -m pip install -r "$APP_DIR/requirements.txt"

if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo
  echo ">>> Created .env from .env.example. Fill in your API keys and set"
  echo ">>> CAMERA_ENABLED=true / CAMERA_SOURCE, then re-run this script."
  exit 0
fi

# Let the service user use the camera and audio devices
sudo usermod -aG video,audio "$APP_USER"

echo "Installing systemd service..."
sed -e "s|__USER__|$APP_USER|g" -e "s|__APP_DIR__|$APP_DIR|g" -e "s|__UID__|$APP_UID|g" \
  "$APP_DIR/deploy/samantha.service" | sudo tee /etc/systemd/system/samantha.service > /dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now samantha

echo
echo "Done. Samantha starts on boot and restarts if it crashes."
echo "  Status:  sudo systemctl status samantha"
echo "  Logs:    journalctl -u samantha -f"
