#!/bin/bash

model_name=DeepSeek-R1-Distill-Qwen-7B
model_path=deepseek-ai/$model_name
task=aime24
export CUDA_VISIBLE_DEVICES=0
python evaluation/eval_math.py \
    --exp_name "evaluation" \
    --output_dir "./logs/$model_path/$task" \
    --dataset $task \
    --base_dir "./outputs/$model_path/$task" 

python evaluation/length_eval.py \
    --exp_name "evaluation" \
    --model_name $model_path \
    --output_dir "./logs/$model_path/$task" \
    --dataset $task \
    --base_dir "./outputs/$model_path/$task" 