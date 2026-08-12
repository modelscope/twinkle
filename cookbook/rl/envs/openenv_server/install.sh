#!/bin/sh
set -eu
OPENENV_SRC="${OPENENV_SRC:-$HOME/OpenEnv}"

echo "==> Installing the openenv client"
pip install openenv

echo "==> Installing coding_env from source"
# server_app.py imports PythonCodeActEnv and PyExecutor from this package; its
# dependencies pull in smolagents, which is the AST interpreter that actually
# runs the model's code.
if [ -d "$OPENENV_SRC/.git" ]; then
    echo "    reusing $OPENENV_SRC"
    git -C "$OPENENV_SRC" pull --ff-only
else
    git clone https://github.com/huggingface/OpenEnv.git "$OPENENV_SRC"
fi
pip install -e "$OPENENV_SRC/envs/coding_env"

echo
echo "Done. Start the server:"
echo "    sh serve.sh                  # binds 0.0.0.0:8000"
echo "    HOST=127.0.0.1 sh serve.sh   # same-host training, no network exposure"
