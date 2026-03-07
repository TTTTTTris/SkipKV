import os
import argparse
import time
import json
import csv
from tqdm import tqdm
from transformers import AutoTokenizer

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data, load_data_vanilla
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

import pdb


def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment related parameters
    parser.add_argument("--exp_name", default="QwQ-32B-Preview", type=str)
    # Prompt type, such as cot, pal, etc.
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    # Output directory
    parser.add_argument("--output_dir", default="./output", type=str)
    # Base directory containing the deepseek folders
    parser.add_argument("--base_dir", default="./results", type=str,
                        help="Base directory containing the deepseek-r1-distill-llama-8b_* folders")
    parser.add_argument("--json_file", default=None, type=str,
                        help="file you want to evaluate")
    # Stop words list
    parser.add_argument("--stop_words", default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："], type=list)
    parser.add_argument("--dataset", default=None, type=str)
    # Model name for tokenizer
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-1B", type=str,
                        help="Model name for tokenizer to calculate token lengths")
    args = parser.parse_args()
    return args

def prepare_data(data_name, args):
    # Load the current JSON file using load_data_vanilla
    if args.dataset is None:
        examples = load_predictions_file(args.input_path)
    else:
        examples = load_data_vanilla(args.input_path)
    return examples

def load_predictions_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a predictions file that might be either JSON array or JSONL format.
    
    Args:
        file_path: Path to the predictions file
        
    Returns:
        List of prediction dictionaries
    """
    try:
        # First, try to load as JSON array
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # Check if it starts with '[' (JSON array format)
        if content.startswith('['):
            print(f"Loading {file_path} as JSON array...")
            data = json.loads(content)
            if isinstance(data, list):
                return data
            else:
                return [data]  # Wrap single object in list
                
        # If not JSON array, try loading as JSONL
        else:
            print(f"Loading {file_path} as JSONL...")
            predictions = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        predictions.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error loading line {line_num}: {e}")
                        print(f"Problematic line: {repr(line)}")
                        continue
            return predictions
            
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
    
def collect_token_lengths(examples, tokenizer=None):
    prefill_tokens_lengths = []
    output_tokens_lengths = []
    total_tokens_lengths = []
    for example in examples:
        if args.dataset is None:
            # Calculate actual token lengths using tokenizer
            question_title = example.get("question_title", "")
            question_content = example.get("question_content", "")
            output_list = example.get("output_list", [])
            code_list = example.get("code_list", [])
            
            print(f"Question title: {question_title}")
            
            if tokenizer:
                # Calculate prefill tokens (input context)
                prefill_text = question_title + " " + question_content
                prefill_tokens = len(tokenizer.encode(prefill_text))
                
                # Calculate output tokens (generated content)
                output_text = ""
                if isinstance(output_list, list) and output_list:
                    output_text += " ".join(str(item) for item in output_list)
                elif isinstance(output_list, str):
                    output_text += output_list
                    
                if isinstance(code_list, list) and code_list:
                    output_text += " " + " ".join(str(item) for item in code_list)
                elif isinstance(code_list, str):
                    output_text += " " + code_list
                    
                output_tokens = len(tokenizer.encode(output_text)) if output_text else 0
                total_tokens = prefill_tokens + output_tokens
            else:
                # Fallback to character count approximation (rough estimate: ~4 chars per token)
                prefill_text = question_title + " " + question_content
                prefill_tokens = len(prefill_text) // 4
                
                output_text = ""
                if isinstance(output_list, list) and output_list:
                    output_text += " ".join(str(item) for item in output_list)
                elif isinstance(output_list, str):
                    output_text += output_list
                    
                if isinstance(code_list, list) and code_list:
                    output_text += " " + " ".join(str(item) for item in code_list)
                elif isinstance(code_list, str):
                    output_text += " " + code_list
                    
                output_tokens = len(output_text) // 4 if output_text else 0
                total_tokens = prefill_tokens + output_tokens
        else:
            prefill_tokens = example.get("prefill_tokens", 0)
            output_tokens = example.get("output_tokens", 0)
            total_tokens = example.get("total_tokens", 0)
            
        print(f"Prefill tokens: {prefill_tokens}, Output tokens: {output_tokens}, Total tokens: {total_tokens}")
        prefill_tokens_lengths.append(prefill_tokens)
        output_tokens_lengths.append(output_tokens)
        total_tokens_lengths.append(total_tokens)

    return prefill_tokens_lengths, output_tokens_lengths, total_tokens_lengths

def save_token_lengths_to_csv(token_lengths, output_csv):
    header = []
    rows = []

    for model, model_data in token_lengths.items():
        row = [model]
        for dataset, token_data in model_data.items():
            prefill_tokens, output_tokens, total_tokens = token_data
            row.extend([prefill_tokens, output_tokens, total_tokens])
        rows.append(row)

    # Write to CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Writing the header
        header = ["Model"]  # First column is the model name
        for dataset in token_lengths[list(token_lengths.keys())[0]].keys():
            header.extend([f"{dataset}-pre", f"{dataset}-out", f"{dataset}-tot"])

        writer.writerow(header)

        for row in rows:
            writer.writerow(row)

def main(data_name, args):
    """
    Process a single JSON file for token lengths.
    """
    # Load tokenizer for accurate token counting
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print(f"Loaded tokenizer: {args.model_name}")
    except Exception as e:
        print(f"Failed to load tokenizer {args.model_name}: {e}")
        print("Using character-based approximation instead.")
    
    examples = prepare_data(data_name, args)
    # if len(examples) > 100:
    #     examples = examples[:100]
    
    # Collect token lengths
    prefill_tokens_lengths, output_tokens_lengths, total_tokens_lengths = collect_token_lengths(examples, tokenizer)

    return prefill_tokens_lengths, output_tokens_lengths, total_tokens_lengths

def main_all(args):
    """
    Traverse all deepseek folders and datasets under base_dir, process each JSON file and summarize results.
    """

    token_lengths = {}

    json_files = []
    if args.json_file:
        json_files.append(args.json_file + 'predictions.jsonl')
    else:
        for file in os.listdir(args.base_dir):
            filepath = os.path.join(args.base_dir, file)
            if os.path.isfile(filepath) and (
                file.endswith(".json") or file.endswith(".jsonl")
            ):
                json_files.append(filepath)

    if not json_files:
        print("No JSON/JSONL files found in the folder.")
        return
    

    for json_file in json_files:
        dataset = args.dataset
        args.input_path = json_file

        print(f"Processing: dataset={dataset}")

        try:
            prefill_tokens, output_tokens, total_tokens = main(dataset, args)
            # Store the token lengths for this model and dataset
            if json_file not in token_lengths:
                token_lengths[json_file] = {}
            token_lengths[json_file][dataset] = (sum(prefill_tokens)//len(prefill_tokens), sum(output_tokens)//len(output_tokens), sum(total_tokens)//len(total_tokens))
        except Exception as e:
            print(f"Error processing {dataset}: {e}")
            continue

    # Save the token lengths to CSV
    output_csv = os.path.join(args.output_dir, f"token_lengths_{args.dataset}.csv")
    save_token_lengths_to_csv(token_lengths, output_csv)
    print(f"Token lengths saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    main_all(args)
