#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="./data/models"
HF_BASE="https://huggingface.co/mlx-community"

check_downloaded() {
    local dir="$BASE_DIR/$1"
    if [ -f "$dir/model.safetensors" ] && [ -d "$dir/voice_embedding" ]; then
        printf "\033[32m[downloaded]\033[0m"
    else
        printf "            "
    fi
}

printf "\033[34mAvailable Voxtral TTS models:\033[0m\n\n"
printf "  1. Voxtral-4B-TTS-2603-mlx-4bit   (~2.5 GB, fastest, RTF <1.0x)  %s\n" "$(check_downloaded Voxtral-4B-TTS-2603-mlx-4bit)"
printf "  2. Voxtral-4B-TTS-2603-mlx-6bit   (~3.5 GB, balanced, RTF ~1.1x)  %s\n" "$(check_downloaded Voxtral-4B-TTS-2603-mlx-6bit)"
printf "  3. Voxtral-4B-TTS-2603-mlx-bf16   (~8.0 GB, highest quality, RTF ~6.3x)  %s\n" "$(check_downloaded Voxtral-4B-TTS-2603-mlx-bf16)"
printf "\n"

read -rp "Select model [1-3]: " choice

case "$choice" in
    1) MODEL_NAME="Voxtral-4B-TTS-2603-mlx-4bit" ;;
    2) MODEL_NAME="Voxtral-4B-TTS-2603-mlx-6bit" ;;
    3) MODEL_NAME="Voxtral-4B-TTS-2603-mlx-bf16" ;;
    *)
        printf "\033[31mInvalid choice.\033[0m\n"
        exit 1
        ;;
esac

MODEL_DIR="$BASE_DIR/$MODEL_NAME"
BASE_URL="$HF_BASE/$MODEL_NAME/resolve/main"

mkdir -p "$MODEL_DIR/voice_embedding"

FILES=(
    config.json
    model.safetensors
    model.safetensors.index.json
    params.json
    tekken.json
    voice_embedding/ar_male.safetensors
    voice_embedding/casual_female.safetensors
    voice_embedding/casual_male.safetensors
    voice_embedding/cheerful_female.safetensors
    voice_embedding/de_female.safetensors
    voice_embedding/de_male.safetensors
    voice_embedding/es_female.safetensors
    voice_embedding/es_male.safetensors
    voice_embedding/fr_female.safetensors
    voice_embedding/fr_male.safetensors
    voice_embedding/hi_female.safetensors
    voice_embedding/hi_male.safetensors
    voice_embedding/it_female.safetensors
    voice_embedding/it_male.safetensors
    voice_embedding/neutral_female.safetensors
    voice_embedding/neutral_male.safetensors
    voice_embedding/nl_female.safetensors
    voice_embedding/nl_male.safetensors
    voice_embedding/pt_female.safetensors
    voice_embedding/pt_male.safetensors
)

printf "\n\033[34mDownloading %s to %s\033[0m\n\n" "$MODEL_NAME" "$MODEL_DIR"

for file in "${FILES[@]}"; do
    dest="$MODEL_DIR/$file"
    printf "\033[34mDownloading %s...\033[0m\n" "$file"
    curl -L --fail -C - --progress-bar -o "$dest" "$BASE_URL/$file"
done

printf "\n\033[32mDone. Model saved to %s\033[0m\n" "$MODEL_DIR"
printf "\nUpdate config.yaml model_dir to:\n  model_dir: %s\n" "$MODEL_DIR"
