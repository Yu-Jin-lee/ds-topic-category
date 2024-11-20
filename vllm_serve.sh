#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES="1"

CONDA_BASE_PATH=/data1/share/anaconda3/envs/vllm
PYTHON_PATH=${CONDA_BASE_PATH}/bin/python
MODEL="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"

# model name만 추출
MODEL_NAME=$(echo ${MODEL} | cut -d'/' -f2)
PORT=8000

nohup ${PYTHON_PATH} -m vllm.entrypoints.openai.api_server \
    --trust-remote-code \
    --model ${MODEL} \
    --port ${PORT} \
    --dtype=half \
    # --max_model_len 4096 > ${MODEL_NAME}.log 2>&1 &
    --max_model_len 4096 > /dev/null 2>&1 &