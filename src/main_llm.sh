#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$SCRIPT_DIR"
cd "$SCRIPT_DIR"

export PYTHONPATH="$SRCIPT_DIR:$PYTHONPATH"

export CC=$(which gcc)
export CXX=$(which g++)
# export CUDA_VISIBLE_DEVICES=0

source activate venv_llm

# ------  select basemodel ----------
MODEL_NAME='LLaMA3-instruct'

#  ------ select the dataset ------ 
# dataset='iemocap'
# dataset='msp'
dataset=${1:?"Error: Dataset name (arg 1) MUST be provided."}

# ------ select the experiment ------------
# Experiments_setting='zero_shot'
# Experiments_setting='lora_training'
# Experiments_setting='inference'
Experiments_setting=${2:?"Error: Experiment setting (arg 2) MUST be provided."}

# ------  training setting ------ 
if [ "$dataset" == "msp" ]; then
    LORA_LR=1e-5
    LORA_DIM=32
    LORA_ALPHA=32
    LORA_DROPOUT_PROB=0.1
    MAX_MASK_PROB=0.1
    historical_window=0

elif [ "$dataset" == "iemocap" ]; then
    LORA_LR=1e-4
    LORA_DIM=16
    LORA_ALPHA=16
    LORA_DROPOUT_PROB=0.05
    MAX_MASK_PROB=0.0
    historical_window=8

else
    echo "ERROR: Invalid or missing dataset."
fi

SEED=11
num_train_epochs=15
accumulations=8
graphics_card=1
mini_batch_size=1
BS=$((accumulations * graphics_card * mini_batch_size))

data_percent=1.0

# set the port for deepspeed: use different ports if running in parallel
port=26000

# name the experiment (your choice)
task="${dataset}_structured_numeric_mask_run"

# Path that contains train/test json file for training/inference
DATA_PATH="../data/${dataset}_dataset/"

# Provide checkpoint_dir to load model weight to continue training or inference
# checkpoint_dir="../models/experiments/LLaMA3-instruct/lora/msp/window_8/LR_1e-5_BS_8_per_1.0_des_context_msp_vad_pred_vad_structured_proposed_mask_new_prob_lora32_class5_11/best"

# -----------------------------------------------------------------------------

case ${MODEL_NAME} in
'LLaMA3-instruct'|'LLaMA3-instruct-70b')
    case ${Experiments_setting} in
    'zero_shot'|'lora_training'|'inference')
        case ${dataset} in
        'iemocap'|'msp')
            echo "******************************************************************************************"
            echo "All parameters are valid."
            echo "The dataset you have selected is: ${dataset} !"
            echo "The base model you have selected is ${MODEL_NAME}!"
            echo "The model's SFT method you have selected: ${Experiments_setting}!"
            echo "******************************************************************************************"
            ;;
        *)
            echo "The dataset parameter is invalid. CHECK IT OUT!"
            FLAG=0
            ;;
        esac
        ;;
    *)
        echo "The Experiments_setting parameter is invalid. CHECK IT OUT!"
        FLAG=0
        ;;
    esac
    ;;
*)
    echo "The MODEL_NAME parameter is invalid. CHECK IT OUT!"
    FLAG=0
    ;;
esac


if [ ${FLAG} = 1 ]
then

    if [ ${dataset} = 'iemocap' ]    
    then
        # MAX_LENGTH=1200
        MAX_LENGTH=2500

    elif [ ${dataset} = 'msp' ]
    then
        #MAX_LENGTH=1024
        MAX_LENGTH=2048

    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo "******************************************************************************************"
    echo -e "Your choose ${dataset}! The max_context_length will be set as ${MAX_LENGTH}!"
    echo "******************************************************************************************"


    if [ ${MODEL_NAME} = 'LLaMA3-instruct' ]
    then
        MODEL_PATH='meta-llama/Meta-Llama-3-8B-Instruct'
    elif [ ${MODEL_NAME} = 'LLaMA3-instruct-70b' ]
    then
        MODEL_PATH='LLaMA3-instruct-70b MODELPATH'
    else
        echo -e "Your choose is not in MY candidations! Please check your Model name!"
    fi
    echo -e "Your choose ${MODEL_NAME}! Model Parameters should be initialized in the path \n ${MODEL_PATH}"


    if [ ${Experiments_setting} = 'zero_shot' ]
    then
        DO_EVAL=True
        DO_TRAIN=False
        LORA=False
        LR=0
	    ZERO_SHOT=True
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as ZERO_SHOT model"
    elif [ ${Experiments_setting} = 'lora_training' ]
    then
        DO_EVAL=True
        DO_TRAIN=True
        LORA=True
        LR=${LORA_LR}
        ZERO_SHOT=False
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as LORA model"
    elif [ ${Experiments_setting} = 'inference' ]
    then
        DO_EVAL=True
        DO_TRAIN=False
        LORA=True
        LR=${LORA_LR}
        ZERO_SHOT=False
        echo -e "Your choose ${Experiments_setting}! The experiment will be set as inferencing"
    else
        echo -e "Your choose is not in MY candidations! Please CHECK your Experiments Setting!"
    fi

    echo "Processed Data_Path: $DATA_PATH"
    deepspeed --master_port=${port} LLM_code/main.py \
    --dataset ${dataset} \
    --model_name_or_path ${MODEL_PATH} \
    --data_dir ${DATA_PATH} \
    --output_dir ../experiments/${MODEL_NAME}/${Experiments_setting}/${dataset}/evaluation_${task}_${SEED} \
    --max_length ${MAX_LENGTH} \
    --batch_size ${BS} \
    --deepspeed_config ./LLM_code/llm_data_utils/deepspeed_config.json \
    --gradient_accumulation_steps ${accumulations} \
    --eval_batch_size 8 \
    --num_train_epochs ${num_train_epochs} \
    --save_steps 100 \
    --lora ${LORA}\
    --learning_rate ${LR} \
    --lora_dim ${LORA_DIM} \
    --lora_alpha ${LORA_ALPHA} \
    --lora_dropout ${LORA_DROPOUT_PROB} \
    --max_mask_prob ${MAX_MASK_PROB} \
    --do_train ${DO_TRAIN} \
    --do_eval ${DO_EVAL} \
    --data_percent ${data_percent} \
    --seed ${SEED} \
    --zero_shot ${ZERO_SHOT}
    #--checkpoint_dir ${checkpoint_dir}
fi
