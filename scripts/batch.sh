#!/bin/bash
cd ..
CONDA_BASE_PATH=/data1/anaconda3/envs/vllm
PYTHON_PATH=${CONDA_BASE_PATH}/bin/python

# ko
nohup python -m batch --language ko --dataset_path ./data/entity/all/ko_entity_all_volume_sorted.csv --state_path ./results/entity_category_infer-retrieve-rank_02/program_state.json --save_path ./results_batch/entity_category_infer-retrieve-rank_02_ko > nohup_ko.out &

# ja
nohup python -m batch --language ja --dataset_path ./data/entity/all/ja_entity_all_volume_sorted.csv --state_path ./results/entity_category_infer-retrieve-rank_02_ja/program_state.json --save_path ./results_batch/entity_category_infer-retrieve-rank_02_ja > nohup_ja.out &