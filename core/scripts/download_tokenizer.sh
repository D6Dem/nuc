
#!/usr/bin/env bash
# Скачивает токенизатор Qwen2.5 (не веса!)
set -e
DEST="${1:-./blocks/qwen_tokenizer}"
mkdir -p "$DEST"

pip install huggingface-hub -q
huggingface-cli download Qwen/Qwen2.5-0.5B \
  vocab.json merges.txt tokenizer_config.json \
  --local-dir "$DEST"

echo "Токенизатор в $DEST"
