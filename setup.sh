#!/usr/bin/env bash
set -e

# Run from the directory containing this script
cd "$(dirname "${BASH_SOURCE[0]}")"

# Install VS Code CLI
# mkdir -p "$HOME/vscode-cli"

# curl -fL 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' \
#   -o "$HOME/vscode-cli/vscode_cli.tar.gz"

# tar -xzf "$HOME/vscode-cli/vscode_cli.tar.gz" \
#   -C "$HOME/vscode-cli"

# Create and activate venv
# python3 -m venv .venv
# source .venv/bin/activate

# if python > 3.11 does not exist
# install python 

# Install uv into /workspace/tools/uv
curl -LsSf https://astral.sh/uv/install.sh \
  | env UV_INSTALL_DIR=/workspace/tools/uv sh

# Tell uv where to store Python
export UV_PYTHON_INSTALL_DIR=/workspace/tools/python

# Make uv available in this terminal
export PATH="/workspace/tools/uv:$PATH"

# Install Python 3.11
uv python install 3.11

#update shell
uv python update-shell
# then restart terminal

# create venv using python 3.11

cd /workspace/segmentation

uv venv --python 3.11 --seed .venv311
source .venv311/bin/activate

python --version



# Install dependencies, including gdown
python -m pip install \
    torch torchvision torchaudio \
    -c constraints.txt \
    --index-url https://download.pytorch.org/whl/cu128

python -m pip install -r requirement.txt -c constraints.txt
python -m pip check

python -c "import torch; assert torch.cuda.is_available(), 'GPU unavailable'; print('GPU:', torch.cuda.get_device_name(0))"

# Download data
# gdown --folder \
#   'https://drive.google.com/drive/folders/1SG80k8JHM5Uq1HQ0Zs0S-Tq7JrfQH-sF?usp=sharing'


# mkdir -p data/input data/output

# tar --exclude='.DS_Store' --exclude='._*' \
#     -xvf example_quadra_21/example_quadra_21.tar \
#     -C data/input --strip-components=1