#!/bin/bash

CONDA_BASE_PATH=/data1/share/anaconda3/envs/vllm
PYTHON_PATH=${CONDA_BASE_PATH}/bin/python
MODEL="Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
HOST=10.10.200.9
MODEL_PORT=8000
MODEL_DEVICE=1

export CUDA_VISIBLE_DEVICES=${MODEL_DEVICE}

MODEL_NAME=$(echo ${MODEL} | cut -d'/' -f2)
# UI_PORT는 MODEL_PORT + 1
PORT=$((${MODEL_PORT} + 1))

nohup ${PYTHON_PATH} -m ui -m ${MODEL} --model-url http://localhost:${MODEL_PORT}/v1 --host ${HOST} --port ${PORT} > ${MODEL_NAME}_gpu${MODEL_DEVICE}_gradio_ui.out &