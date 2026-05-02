#!/bin/bash

# Exit immediately if any command fails
set -e

source activate venv_audio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
 
if [ "$1" == "iemocap" ]; then
    echo "Running IEMOCAP only..."
    "$VENV_AUDIO" preprocess_dataset/IEMOCAP_preprocess_pipeline.py

elif [ "$1" == "msp" ]; then
    echo "Running MSP only..."
    "$VENV_AUDIO" preprocess_dataset/MSP_preprocess_pipeline.py
else
    echo "Running all pipelines..."
    "$VENV_AUDIO" preprocess_dataset/IEMOCAP_preprocess_pipeline.py
    "$VENV_AUDIO" preprocess_dataset/MSP_preprocess_pipeline.py
fi
