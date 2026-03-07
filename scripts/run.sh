run_task() {
    id=$1              # GPU id
    task=$2            # dataset name (without extension)
    method=$3          # compression/eviction method
    kv_budget=$4       # KV budget
    S_threshold=$5     # threshold
    eval_batch_size=$6 # batch size
    nsamples=$7

    mkdir -p logs/multibatch/${model_name}/
    mkdir -p ./outputs/${model_path}

    CUDA_VISIBLE_DEVICES=$id python3 ./run_math.py \
        --dataset_path ./data/${task}.jsonl \
        --save_path ./outputs/${model_path}/$task/output_rank_${nsamples}_${task}_${method}_${kv_budget}_${S_threshold}_${eval_batch_size}_${time}.jsonl \
        --kv_budget ${kv_budget} \
        --model_path ${model_path} \
        --nsamples $nsamples \
        --eval_batch_size ${eval_batch_size} \
        --S_threshold ${S_threshold} \
        --method ${method} 2>&1 | tee -a logs/multibatch/${model_name}/${nsamples}_${task}_${kv_budget}_${method}_${eval_batch_size}_${time}.txt
}

run_seal() {
    id=$1              # GPU id
    task=$2            # dataset name (without extension)
    method=$3          # compression/eviction method
    kv_budget=$4       # KV budget
    S_threshold=$5     # threshold
    eval_batch_size=$6 # batch size
    steering_coef=$7   # steering coef
    gamma=$8  # steering gamma
    steering_layer=$9 # steering layer
    nsamples=${10}


    steering=SEAL

    mkdir -p logs/multibatch/${model_name}/
    mkdir -p ./outputs/${model_path}

    CUDA_VISIBLE_DEVICES=$id python3 ./run_math.py \
        --dataset_path ./data/${task}.jsonl \
        --save_path ./outputs/$model_path/$task/output_rank_${nsamples}_${task}_${method}_${kv_budget}_${steering}_${steering_layer}_${steering_coef}_${gamma}_${S_threshold}_${eval_batch_size}.jsonl \
        --model_path $model_path \
        --eval_batch_size $eval_batch_size \
        --kv_budget $kv_budget \
        --use_chat_format \
        --remove_bos \
        --S_threshold $S_threshold \
        --nsamples $nsamples \
        --method $method \
        --steering $steering \
        --steering_gamma $gamma \
        --steering_vector steering_vectors/$model_name/layer_${steering_layer}_transition_reflection_steervec.pt \
        --steering_layer $steering_layer \
        --steering_coef $steering_coef 2>&1 | tee -a logs/multibatch/${model_name}/${nsamples}_${task}_${kv_budget}_${steering}_${steering_layer}_${method}_${steering_coef}_${gamma}_${S_threshold}_${eval_batch_size}.txt
}


time="exp_$(date +%Y%m%d_%H%M%S)"

### qwen-7b
### 
model_name=DeepSeek-R1-Distill-Qwen-7B
model_path=deepseek-ai/$model_name
batch_size=10
nsamples=-1

# task=math
# run_task 0 $task fullkv 768 0.95 $batch_size $nsamples 
# run_task 0 $task rkv 768 0.95 $batch_size $nsamples 
# run_task 1 $task rkv 1024 0.95 $batch_size $nsamples 
# run_task 2 $task rkv 1260 0.95 $batch_size $nsamples 
# run_task 3 $task rkv 1536 0.95 $batch_size $nsamples 
# run_seal 4 $task skipkv 768 0.96 $batch_size -1.25 0.02 20 $nsamples
# run_seal 4 $task skipkv 1024 0.99 $batch_size -1.25 0.02 20 $nsamples
# run_seal 4 $task skipkv 1260 0.99 $batch_size -1.25 0.02 20 $nsamples
# run_seal 4 $task skipkv 1536 0.95 $batch_size -1.25 0.02 20 $nsamples

task=aime24
# run_task 4 $task fullkv 1536 0.95 $batch_size $nsamples &
# run_task 5 $task rkv 1536 0.95 $batch_size $nsamples &
# run_task 1 $task rkv 2220 0.95 $batch_size $nsamples &
# run_task 2 $task rkv 3072 0.95 $batch_size $nsamples &
# run_task 3 $task rkv 4096 0.95 $batch_size $nsamples &
run_seal 6 $task skipkv 1536 0.98 $batch_size -1.25 0.02 20 $nsamples &
# run_seal 3 $task skipkv 2220 0.98 $batch_size -1.25 0.02 20 $nsamples 
# run_seal 3 $task skipkv 3072 0.98 $batch_size -1.25 0.02 20 $nsamples 
# run_seal 3 $task skipkv 4096 0.99 $batch_size -1.25 0.02 20 $nsamples 

# task=gsm8k
# run_seal 2 $task skipkv 512 0.95 200 -1.25 0.02 20 $nsamples 
# run_seal 2 $task skipkv 512 0.95 100 -1.25 0.02 20 $nsamples 
# run_seal 2 $task skipkv 512 0.95 10 -1.25 0.02 20 $nsamples 


# ### qwen-14b
# ### 
# model_name=DeepSeek-R1-Distill-Qwen-14B
# model_path=deepseek-ai/$model_name
# batch_size=10
# nsamples=-1

# task=math
# run_task 0 $task fullkv 768 0.95 $batch_size $nsamples 
# run_task 0 $task rkv 768 0.95 $batch_size $nsamples 
# run_task 1 $task rkv 1024 0.95 $batch_size $nsamples 
# run_task 2 $task rkv 1536 0.95 $batch_size $nsamples 
# run_task 3 $task rkv 2048 0.95 $batch_size $nsamples 
# run_seal 4 $task skipkv 768 0.98 $batch_size -1 0.02 35 $nsamples 
# run_seal 5 $task skipkv 1024 0.97 $batch_size -1 0.02 35 $nsamples 
# run_seal 6 $task skipkv 1536 0.98 $batch_size -1 0.02 35 $nsamples 
# run_seal 7 $task skipkv 2048 0.98 $batch_size -1 0.02 35 $nsamples 

# task=aime24
# run_task 0 $task fullkv 1536 0.95 $batch_size $nsamples 
# run_task 0 $task rkv 1024 0.95 $batch_size $nsamples 
# run_task 1 $task rkv 1536 0.95 $batch_size $nsamples 
# run_task 2 $task rkv 3072 0.95 $batch_size $nsamples 
# run_task 3 $task rkv 4096 0.95 $batch_size $nsamples 
# run_seal 4 $task skipkv 1024 0.96 $batch_size -1.25 0.02 35 $nsamples 
# run_seal 5 $task skipkv 1536 0.96 $batch_size -1.25 0.02 35 $nsamples 
# run_seal 6 $task skipkv 3072 0.97 $batch_size -1.25 0.02 35 $nsamples 
# run_seal 7 $task skipkv 4096 0.96 $batch_size -1.25 0.02 35 $nsamples 



# ### llama-8b
# ### 
# model_name=DeepSeek-R1-Distill-Llama-8B
# model_path=deepseek-ai/$model_name
# batch_size=10
# nsamples=-1

# task=math
# run_task 0 $task fullkv 768 0.95 $batch_size $nsamples 
# run_task 0 $task rkv 768 0.95 $batch_size $nsamples 
# run_task 1 $task rkv 1024 0.95 $batch_size $nsamples 
# run_task 2 $task rkv 1536 0.95 $batch_size $nsamples 
# run_task 3 $task rkv 2048 0.95 $batch_size $nsamples 
# run_seal 4 $task skipkv 768 0.95 $batch_size -1.25 0.02 20 $nsamples
# run_seal 5 $task skipkv 1024 0.97 $batch_size -1.25 0.02 20 $nsamples
# run_seal 6 $task skipkv 1536 0.96 $batch_size -1.25 0.02 20 $nsamples
# run_seal 7 $task skipkv 2048 0.99 $batch_size -1.25 0.02 20 $nsamples

# task=aime24
# run_task 0 $task fullkv 1536 0.95 $batch_size $nsamples 
# run_task 0 $task rkv 2048 0.95 $batch_size $nsamples 
# run_task 1 $task rkv 2700 0.95 $batch_size $nsamples 
# run_task 2 $task rkv 3072 0.95 $batch_size $nsamples 
# run_task 3 $task rkv 4096 0.95 $batch_size $nsamples 
# run_seal 4 $task skipkv 2048 0.95 $batch_size -1 0.02 20 $nsamples
# run_seal 5 $task skipkv 2700 0.97 $batch_size -1 0.02 20 $nsamples
# run_seal 6 $task skipkv 3072 0.99 $batch_size -1 0.02 20 $nsamples
# run_seal 7 $task skipkv 4096 0.99 $batch_size -1 0.02 20 $nsamples

# ### Qwen3-8B
# ### 
# model_name=Qwen3-8B
# model_path=Qwen/$model_name
# batch_size=10
# nsamples=-1
# task=aime24
# run_seal 4 $task skipkv 1536 0.95 $batch_size -1 0.02 25 $nsamples &
# run_seal 5 $task skipkv 2220 0.95 $batch_size -1 0.02 25 $nsamples &
# run_seal 6 $task skipkv 3072 0.95 $batch_size -1 0.02 25 $nsamples &
# run_seal 7 $task skipkv 4096 0.95 $batch_size -1 0.02 25 $nsamples &
