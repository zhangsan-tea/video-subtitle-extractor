#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if ! command -v python3.12 >/dev/null 2>&1; then
  echo "未找到 python3.12。请先安装：brew install python@3.12" >&2
  exit 1
fi

if [ ! -d "videoEnv" ]; then
  python3.12 -m venv videoEnv
fi

# shellcheck disable=SC1091
source videoEnv/bin/activate

python -m pip install --upgrade pip

# 安装 PaddlePaddle（macOS CPU 版）
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# 安装项目依赖
python -m pip install -r requirements.txt

# GUI 依赖（tkinter）
if ! python - <<'PY'
import tkinter  # noqa: F401
print("tkinter OK")
PY
then
  echo "缺少 tkinter，请执行：brew install python-tk@3.12" >&2
  exit 1
fi

echo "环境安装完成。运行：source videoEnv/bin/activate && python gui.py"
