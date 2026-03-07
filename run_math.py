import json
import random
import argparse
import os
import numpy as np
from tqdm import tqdm
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from skipkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3, replace_qwen2_steering, replace_llama_steering, replace_qwen3_steering
from transformers.cache_utils import DynamicCache
import torch.nn.functional as F

dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 16384,
    "math": 8192,
}

def append_jsonl(data, file_path):
    """
    Append results from list to a .jsonl file.

    Parameters:
        data (list): List containing Python dictionaries to append.
        file_path (str): Target .jsonl file path.
    """
    with open(file_path, 'a', encoding='utf-8') as f:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"
model_name_to_gamma = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 0.27, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 0.46, "Qwen/QwQ-32B": 0.5}
model_name_to_steering_vectors = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "steering_vectors_qwen7b.pt", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "steering_vectors_llama8b.pt", "Qwen/QwQ-32B":"steering_vectors_qwq32b.pt"}
model_name_to_layer_index = {"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 20, "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":20, "Qwen/QwQ-32B": 57}


def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    fout = open(args.save_path, "w")

    prompts = []
    test_data = []
    questions_json = []

    with open(args.dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[args.dataset_name][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)
            questions_json.append(example)

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)
    
    if args.steering == 'SEAL':
        prefix="Answer the following questions. You should think step-by-step and put your final answer within \\boxed{}.\n"
        prompts = []
        for i, example in enumerate(test_data):
            prompt =  prefix+"Question: " + example["question"].strip()+"\nAnswer: "
            if args.use_chat_format:
                if  "deepseek" in args.model_path:
                    messages = [{"role": "user", "content": prefix + "Question: " + example["question"].strip()}]
                else:
                    messages = [{"role": "system", "content": prefix}, {"role": "user", "content": "Question: " + example["question"].strip()}]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                    prompt = prompt[len(tokenizer.bos_token):]
            prompts.append(prompt)
        with open(os.path.join("outputs/example_prompt.txt"), 'w') as fout_e:
            fout_e.write(prompts[0])

    # Randomly select samples if nsamples is specified
    if args.nsamples > 0 and args.nsamples < len(prompts):
        prompts = prompts[:args.nsamples]
        test_data = test_data[:args.nsamples]
    
    print(f"Processing {len(prompts)} samples")

    import time
    # Start timing for total generation
    total_start_time = time.time()

    # Rank batches by length for multi-batch decoding (shortest → longest)
    if args.batch_grouping:
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

    for i in tqdm(range(args.start_id, len(prompts), args.eval_batch_size)):
        # Reset model state for each new sample to ensure consistent results
        if hasattr(model, 'reset_for_new_sample'):
            model.reset_for_new_sample()
            
        if args.steering == 'SEAL':
            model.start_new_round(args.steering_coef)
        batch_prompts = prompts[i : i + args.eval_batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")

        prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            output = model.generate(
                **tokenized_prompts,
                max_length=args.max_length,
                do_sample=False,
                num_beams=1,
                use_cache=True
            )

        # output = model.generate(
        #     **tokenized_prompts,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.95,
        #     max_length=args.max_length,
        #     num_return_sequences=64,
        #     pad_token_id=tokenizer.eos_token_id,
        # )

        batch_token_stats = []
        for j in range(output.size(0)):
            total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

            prefill = prefill_lengths[j]
            output_tokens = total_tokens - prefill

            batch_token_stats.append(
                {
                    "sample_idx": order[i + j] if args.batch_grouping else (i+j),
                    "prefill_tokens": prefill,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

        batch_outputs = tokenizer.batch_decode(
            [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
            skip_special_tokens=True,
        )

        torch.cuda.empty_cache()

        for j in range(len(batch_outputs)):
            sample_idx = batch_token_stats[j]["sample_idx"]
            test_data[sample_idx]["prompt"] = batch_prompts[j]
            test_data[sample_idx]["output"] = batch_outputs[j]
            test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
            test_data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
            test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
            test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

            fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()
    # Calculate total generation time
    total_end_time = time.time()
    total_generation_time = total_end_time - total_start_time
    print(f"\nTotal generation time: {total_generation_time/60:.2f} mins")


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=-1, help="Number of samples to randomly select (-1 for all)")
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
    parser.add_argument("--update_kv", type=lambda x: x.lower() in ['true'], default=True)
    parser.add_argument("--batch-grouping", type=lambda x: x.lower() in ['true'], default=True)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )

    # model config
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
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
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
    parser.add_argument('--start_id', type=int, default=0, help='Start sample id for debugging')
    parser.add_argument('--S_threshold', type=float, default=0.95)
    parser.add_argument('--record_kept_token_indices', action='store_true', help='Enable recording')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    args.dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    if args.max_length == -1: args.max_length = dataset2max_length[args.dataset_name]

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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # apply monkey patch
    if args.method.lower() != "fullkv":
        if "llama" in args.model_path.lower():
            replace_llama(compression_config)
        elif "qwen3" in args.model_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in args.model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")

    if args.steering=='SEAL':
        if "llama" in args.model_path.lower():
            replace_llama_steering()
        elif "qwen" in args.model_path.lower():
            replace_qwen2_steering()
        elif "qwen3" in args.model_path.lower():
            replace_qwen3_steering()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        cache_dir=args.cache_dir,
        attn_implementation=args.attn_implementation,
    )

    model.eval()

    model.config.update(model_config)

    newline_token_ids = ["\n", ".\n", ")\n", "\n\n", ".\n\n", ")\n\n", "?\n\n"]
    model.newline_token_ids = [tokenizer.encode(t)[-1] for t in newline_token_ids]

    if args.method.lower() != "fullkv":
        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]

    # add ASC
    if args.steering=='ASC':
        steering_vec = torch.load("./vectors/"+model_name_to_steering_vectors[args.model_path])
        steering_vec = steering_vec.mean(dim=0)
        steering_vec = steering_vec.to(model.device).to(model.dtype)
        steering_str = model_name_to_gamma[args.model_path]

        def add_steer(_, __, output):
            gamma = steering_str
            output[0][:,-1,:] =output[0][:,-1,:] - gamma * steering_vec.to(output[0][:,-1,:].device)
            return (output[0], *output[1:])

        handle = model.model.layers[int(model_name_to_layer_index[args.model_path])].register_forward_hook(add_steer)

    # add SEAL
    if args.steering=='SEAL':
        vector_name_split = args.steering_vector.split("/")[-3:]
        vector_name_split[-1] = vector_name_split[-1].split(".")[0]
        name = "_".join(vector_name_split)
        
        steer_vec = torch.load(args.steering_vector, weights_only=True)
        steer_vec = steer_vec.to(model.device)
        model.set_steering_flag(steering_flag=True, steering_layer=args.steering_layer, steer_vec=steer_vec,  
                                steer_coef=args.steering_coef, steer_gamma=args.steering_gamma, tokenizer=tokenizer)
    
    # Enable token monitoring for non-execution thoughts
    wait_tokens = ["Alternatively", "Wait", "again"]
    wait_token_ids = [tokenizer.encode(t)[-1] for t in wait_tokens]

    if args.method.lower() in ["skipkv"]: 
        model.enable_wait_token_monitoring(wait_token_ids, model.newline_token_ids, tokenizer=tokenizer)
    main(args)
