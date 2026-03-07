# model_name=DeepSeek-R1-Distill-Qwen-14B
# model_path=deepseek-ai/$model_name
# model_path=qwen-14b


run_task() {
    id=$1              # GPU id
    method=$2          # compression/eviction method
    kv_budget=$3       # KV budget
    S_threshold=$4     # threshold
    batch_size=$5 # batch size
    nsamples=$6


    CUDA_VISIBLE_DEVICES=$id python3 ./eval_code.py \
        --model_name_or_path $model_path \
        --kv_budget $kv_budget \
        --S_threshold $S_threshold \
        --method $method \
        --release release_v1 \
        --use_chat_format \
        --batch_size $batch_size \
        --max_tokens 10000 \
        --remove_bos \
        --max_examples $nsamples \
        --save_dir result/${model_path}/${method}/${kv_budget}/${S_threshold} 2>&1 | tee -a logs/${model_path}/livecodebench_${kv_budget}_${method}_${batch_size}.txt 
}

run_seal() {
    id=$1              # GPU id
    method=$2          # compression/eviction method
    kv_budget=$3       # KV budget
    S_threshold=$4     # threshold
    batch_size=$5 # batch size
    steering_coef=$6   # steering coef
    gamma=$7  # steering gamma
    steering_layer=$8 # steering layer
    nsamples=$9


    steering=SEAL

    CUDA_VISIBLE_DEVICES=$id python3 ./eval_code.py \
        --model_name_or_path $model_path \
        --kv_budget $kv_budget \
        --release release_v1 \
        --batch_size $batch_size \
        --use_chat_format \
        --remove_bos \
        --S_threshold $S_threshold \
        --max_examples $nsamples \
        --max_tokens 10000 \
        --method $method \
        --steering $steering \
        --steering_gamma $gamma \
        --steering_vector ./SEAL/results/MATH_train/$model_name/baseline_10000/vector_500_500/layer_${steering_layer}_transition_reflection_steervec.pt \
        --steering_layer $steering_layer \
        --steering_coef $steering_coef \
        --save_dir result/${model_path}/$steering/${method}/${steering_coef}/${kv_budget}/${S_threshold} 2>&1 | tee -a logs/${model_path}/livecodebench_${kv_budget}_${steering}_${method}_${steering_coef}.txt
}

### qwen-7b
model_name=DeepSeek-R1-Distill-Qwen-7B
model_path=deepseek-ai/$model_name
method=skipkv

run_seal 0 $method 1536 0.95 10 -1 0.02 20
run_seal 1 $method 2000 0.95 10 -1 0.02 20
run_seal 2 $method 3072 0.95 10 -1 0.02 20
run_seal 3 $method 3854 0.95 10 -1 0.02 20

### qwen-14b
model_name=DeepSeek-R1-Distill-Qwen-14B
model_path=deepseek-ai/$model_name

run_seal 0 $method 1536 0.95 10 -1.25 0.02 35
run_seal 1 $method 2000 0.95 10 -1.25 0.02 35
run_seal 2 $method 3072 0.95 10 -1 0.02 35
run_seal 3 $method 3854 0.95 10 -1 0.02 53

### llama-8b
model_name=DeepSeek-R1-Distill-Qwen-14B
model_path=deepseek-ai/$model_name

run_seal 0 $method 1536 0.95 10 -1 0.02 35
run_seal 1 $method 2000 0.95 10 -1 0.02 35
run_seal 2 $method 3072 0.95 10 -1 0.02 35
run_seal 3 $method 3854 0.95 10 -1 0.02 53
