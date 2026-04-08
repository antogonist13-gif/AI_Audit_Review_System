#!/bin/bash
# Запуск AI-assisted Audit Review System
cd "$(dirname "$0")"
source .venv/bin/activate

# Использовать только локальный кэш моделей, не обращаться к HuggingFace
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

streamlit run app.py
