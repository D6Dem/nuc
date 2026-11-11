
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

mkdir -p blocks memory workspace
python -m nucleus.build_demo_core --out=./blocks/core.nblk

echo ""
echo "=== Установка готова ==="
echo "Запустите smoke-тесты: pytest -v"
echo "Затем: python -m nucleus.main --scan"
