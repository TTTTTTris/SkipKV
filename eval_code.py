import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from collections import Counter
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from tqdm import trange

import sys
import os
import gc
from code_evaluation import codegen_metrics, load_code_generation_dataset, get_deepseekcode_question_template_answer, extract_code, extract_instance_results
from skipkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3, replace_qwen2_steering, replace_llama_steering

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_name_to_gamma = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.27, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.46, "Qwen/QwQ-32B": 0.5}
model_name_to_steering_vectors = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "steering_vectors_qwen7b.pt", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "steering_vectors_llama8b.pt", "Qwen/QwQ-32B":"steering_vectors_qwq32b.pt"}
model_name_to_layer_index = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 20, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":20, "Qwen/QwQ-32B": 57}

def main(args):
    random.seed(42)

    print("Loading data...")

    benchmark = load_code_generation_dataset(release_version=args.release)

    if args.start:
        benchmark = benchmark[args.start:]
    
    if args.max_examples and len(benchmark) > args.max_examples:
        benchmark = benchmark[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for i, example in enumerate(benchmark):
        prompt =  get_deepseekcode_question_template_answer(example)
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]
        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    # ====== build compression config ======
    compression_config = {
        "method": args.method,
        "method_config": {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
            "S_threshold": args.S_threshold,
            "record_kept_token_indices": args.record_kept_token_indices
        },
        "compression": None,
        "update_kv": args.update_kv
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
    }
    

    # apply monkey patch
    if args.method.lower() != "fullkv":
        if "llama" in args.model_name_or_path.lower():
            replace_llama(compression_config)
        elif "qwen3" in args.model_name_or_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in args.model_name_or_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_name_or_path}")

    if args.steering=='SEAL':
        if "llama" in args.model_name_or_path.lower():
            replace_llama_steering()
        elif "qwen" in args.model_name_or_path.lower():
            replace_qwen2_steering()
        
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    
    model.eval()

    model.config.update(model_config)


    # add ASC
    if args.steering=='ASC': 
        steering_vec = torch.load("./vectors/"+model_name_to_steering_vectors[args.model_name_or_path])
        steering_vec = steering_vec.mean(dim=0)
        steering_vec = steering_vec.to(model.device).to(model.dtype)
        steering_str =model_name_to_gamma[args.model_name_or_path]

    def add_steer(_, __, output):
        gamma = steering_str
        output[0][:,-1,:] =output[0][:,-1,:] - gamma * steering_vec.to(output[0][:,-1,:].device)
        return (output[0], *output[1:])

    if args.steering=='ASC':
        handle = model.model.layers[int(model_name_to_layer_index[args.model_name_or_path])].register_forward_hook(add_steer)

    # add SEAL
    if args.steering=='SEAL':
        vector_name_split = args.steering_vector.split("/")[-3:]
        vector_name_split[-1] = vector_name_split[-1].split(".")[0]
        name = "_".join(vector_name_split)
        # args.save_dir = os.path.join(args.save_dir, name, f"coef_{args.steering_coef}")
        
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        steer_vec = steer_vec.to(model.device)
        model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  
                                steer_coef=args.steering_coef, steer_gamma=args.steering_gamma, tokenizer=tokenizer)

    # Get punctuation token IDs individually
    newline_token_ids = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n", "?\n\n"]
    model.newline_token_ids = [tokenizer.encode(t)[-1] for t in newline_token_ids]

    # wait_tokens = ["Wait", "again"]
    wait_tokens = ["Alternatively", "Wait", "again"]
    wait_token_ids = [tokenizer.encode(t)[-1] for t in wait_tokens]

    if args.method.lower() in ["skipkv"]: 
        model.enable_wait_token_monitoring(wait_token_ids, model.newline_token_ids, tokenizer=tokenizer)

    import time
    
    # Start timing for total generation
    total_start_time = time.time()
    

    # Rank batches by length for multi-batch decoding (shortest → longest)
    prefill_lengths = []
    for p in prompts:
        tp = tokenizer(
            p,
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")
        prefill_len = tp["attention_mask"].sum(dim=1).item()
        prefill_lengths.append(prefill_len)
    order = sorted(range(len(prefill_lengths)), key=lambda i: prefill_lengths[i])
    prompts = [prompts[i] for i in order]
    benchmark = [benchmark[i] for i in order]

    outputs = []
    for i in trange(0, len(prompts), args.batch_size):
        # Reset model state for each new sample to ensure consistent results
        if hasattr(model, 'reset_for_new_sample'):
            model.reset_for_new_sample()

        if args.steering == 'SEAL':
            model.start_new_round(args.steering_coef)
        batch = prompts[i:i+args.batch_size]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}
        with torch.no_grad():
            output = model.generate(**tokenized_batch, do_sample=False, max_new_tokens=args.max_tokens,use_cache=True)
        prompt_len = tokenized_batch["input_ids"].shape[1]
        output = [tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output]
        outputs.extend(output)
    
    # Calculate total generation time
    total_end_time = time.time()
    total_generation_time = total_end_time - total_start_time
    print(f"\nTotal generation time: {total_generation_time/60:.2f} mins")
    
    outputs = [[o] for o in outputs]
    
    combined_results = [
        (
            outputs_list,
            [extract_code(output) for output in outputs_list],
        )
        for outputs_list in outputs
    ]

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmark, combined_results
        )
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        json.dump(save_results, f, indent=4)


    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=50,
    )

    print(metrics[0]["pass@1"])

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.save_dir, "code_eval.jsonl"), "w") as f:
        json.dump(save_eval_results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="release_v1",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    # method config
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["skipkv", "fullkv", "rkv", "snapkv", "streamingllm", "h2o"],
    )
    parser.add_argument("--kv_budget", type=int, default=1536)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", type=bool, default=True)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument(
        "--divide_method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument(
        "--compression_content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )

    # steering
    parser.add_argument(
        '--steering',
        type=str,
        default=None,
        choices=["ASC", "SEAL"],
        help='Enable steering if this flag is set.'
    )
    parser.add_argument(
        "--steering_vector",
        type=str,
        default=None
    )
    parser.add_argument(
        "--steering_layer",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--steering_coef",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--steering_gamma",
        type=float,
        default=0.0
    )
    
    # sentence-level
    parser.add_argument('--S_threshold', type=float, default=0.95)
    parser.add_argument('--record_kept_token_indices', action='store_true', help='Enable recording')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, "base")
    
    if args.remove_bos:
        args.save_dir = args.save_dir + "_remove_bos"

    if args.max_examples or args.start:
        start = 0 if args.start is None else args.start
        end = start + args.max_examples if args.max_examples is not None else -1
        args.save_dir = os.path.join(args.save_dir, f"{start}_{end}")

    print(args.save_dir)
    main(args)

        
