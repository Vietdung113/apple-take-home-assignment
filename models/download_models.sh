#!/bin/bash
# Download GGUF model for Qwen3-0.6B base model from Google Drive

set -e

MODELS_DIR="$(dirname "$0")"
FILE_ID="19dLrBVM1nIvPZ25XBhsvYk0oZdZnqCtO"
OUTPUT="$MODELS_DIR/finetuned-qwen3-0.6b.Q4_K_M.gguf"

echo ""
echo "========================================================================"
echo "Downloading Qwen3-0.6B GGUF (Base Model) from Google Drive"
echo "========================================================================"
echo "Size: ~400MB (Q4_K_M quantization)"
echo ""

if [ -f "$OUTPUT" ]; then
  echo "File already exists: $OUTPUT"
  ls -lh "$OUTPUT"
  echo ""
  read -p "Re-download? [y/N] " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipped."
    exit 0
  fi
fi

# Download from Google Drive (handles large file confirmation)
echo "Downloading from Google Drive (file ID: $FILE_ID)..."
if command -v gdown &> /dev/null; then
  gdown "$FILE_ID" -O "$OUTPUT"
else
  echo "gdown not found, using curl..."
  # Google Drive direct download URL for large files
  CONFIRM=$(curl -sc /tmp/gdrive_cookie \
    "https://drive.google.com/uc?export=download&id=$FILE_ID" \
    | grep -o 'confirm=[^&]*' | head -1)

  if [ -n "$CONFIRM" ]; then
    curl -Lb /tmp/gdrive_cookie \
      "https://drive.google.com/uc?export=download&$CONFIRM&id=$FILE_ID" \
      -o "$OUTPUT"
  else
    curl -L \
      "https://drive.google.com/uc?export=download&id=$FILE_ID" \
      -o "$OUTPUT"
  fi
  rm -f /tmp/gdrive_cookie
fi

echo ""
echo "Download complete!"
echo ""
echo "Model downloaded to:"
ls -lh "$OUTPUT"
echo ""
echo "Usage with llama.cpp:"
echo "  llama-server --model $OUTPUT --port 8080"
