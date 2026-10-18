#!/usr/bin/env bash
set -e

# Run from the directory containing this script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Install VS Code CLI
mkdir -p "$HOME/vscode-cli"

curl -fL 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
  -o "$HOME/vscode-cli/vscode_cli.tar.gz"

tar -xzf "$HOME/vscode-cli/vscode_cli.tar.gz" \
  -C "$HOME/vscode-cli"

# Create and activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies, including gdown
python -m pip install -r requirement.txt

# Download data
gdown --folder \
  'https://drive.google.com/drive/folders/1SG80k8JHM5Uq1HQ0Zs0S-Tq7JrfQH-sF?usp=sharing'