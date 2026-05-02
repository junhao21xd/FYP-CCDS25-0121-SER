#!/bin/bash

# Exit immediately if any command fails
set -e

source activate venv_audio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==============================================================================
# 1. USER CONFIGURATION
# Modify the variables below to change the behavior of the Python scripts.
# ==============================================================================
# --- Args ---
DATASET="iemocap"
#DATASET="msp"
ORIGINAL_CSV="../data/${DATASET}_dataset/${DATASET}_dataset_original.csv"

# --- Args for run_gender_classifer.py ---
# GENDER_CLASSIFIER_PATH="../checkpoints/wav2vec2-gender-best-model_${DATASET}"
GENDER_INPUT_CSV=$ORIGINAL_CSV
GENDER_OUTPUT_CSV="../data/${DATASET}_dataset/${DATASET}_gender_predictions.csv"

# --- Args for vad regressor ---
VAD_INPUT_CSV=$ORIGINAL_CSV
VAD_OUTPUT_PATH="../data/${DATASET}_dataset/${DATASET}_audeering_data_multiple"

# --- Args for combine_file ---
COMBINED_OUTPUT_CSV="../data/${DATASET}_dataset/${DATASET}_dataset_processed.csv"

# --- Args for egemaps extractor ---
# FEATURE_EXTRACTION_INPUT_CSV=$COMBINED_OUTPUT_CSV
FEATURE_EXTRACTION_INPUT_CSV=$ORIGINAL_CSV
FEATURE_INPUT_CSV="../data/${DATASET}_dataset/${DATASET}_dataset_egemaps_features.csv"
FEATURE_OUTPUT_PATH="../data/${DATASET}_dataset"

# --- Args for run_asr.py ---
ASR_MODEL_ID="openai/whisper-large-v3-turbo"
# ASR_TRAIN_INPUT_JSON="../data/${DATASET}_dataset/train.json"
# ASR_TRAIN_OUTPUT_JSON="../data/${DATASET}_dataset/train.json"
ASR_TEST_INPUT_JSON="../data/${DATASET}_dataset/test.json"
ASR_TEST_OUTPUT_JSON=$ASR_TEST_INPUT_JSON

# ==============================================================================
# 2. SCRIPT EXECUTION
# ==============================================================================
python main_preprocess_pipeline.py \
    --dataset "$DATASET" \
    --original_csv "$ORIGINAL_CSV" \
    --train_gender_classifier \
    --eval_gender_classifier \
    --gender_input_csv "$GENDER_INPUT_CSV" \
    --gender_output_csv "$GENDER_OUTPUT_CSV" \
    --preprocess_vad_data \
    --train_vad_regressor \
    --eval_vad_regressor \
    --vad_input_csv "$VAD_INPUT_CSV" \
    --vad_output_path "$VAD_OUTPUT_PATH" \
    --combine_files \
    --combined_output_csv "$COMBINED_OUTPUT_CSV" \
    --run_extractor \
    --feature_extraction_input_csv "$FEATURE_EXTRACTION_INPUT_CSV" \
    --process_audio_feature \
    --feature_input_csv "$FEATURE_INPUT_CSV" \
    --feature_output_path "$FEATURE_OUTPUT_PATH" \
    --run_asr \
    --asr_model_id "$ASR_MODEL_ID" \
    --asr_test_input_json "$ASR_TEST_INPUT_JSON" \
    --asr_test_output_json "$ASR_TEST_OUTPUT_JSON"