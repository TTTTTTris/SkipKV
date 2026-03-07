import os
import argparse
import json
import csv
from tqdm import tqdm

from evaluate import evaluate
from utils import save_jsonl
from python_executor import PythonExecutor
from data_loader import load_data_vanilla
from parser import choice_answer_clean, parse_ground_truth, run_execute


def parse_args():
    parser = argparse.ArgumentParser()
    # Experiment related parameters
    parser.add_argument("--exp_name", default="QwQ-32B-Preview", type=str)
    # Prompt type, such as cot, pal, etc.
    parser.add_argument("--prompt_type", default="cot", type=str)
    # Specify the folder containing JSON files to be evaluated (not a single JSON file)
    parser.add_argument(
        "--base_dir",
        default="./results",
        type=str,
        help="Folder containing JSON/JSONL files to be evaluated",
    )
    # Output directory
    parser.add_argument("--output_dir", default="./output", type=str)
    # Stop words list
    parser.add_argument(
        "--stop_words",
        default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："],
        type=list,
    )
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    # Load current JSON file using load_data_vanilla
    examples = load_data_vanilla(args.input_path)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{args.exp_name}/{data_name}", exist_ok=True)

    # If there are sample deduplication or filtering operations, they can be implemented here
    processed_samples = []
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(data_name, args):
    """
    Process and evaluate a single JSON file
    data_name is the math dataset name (here using JSON filename without extension)
    """
    examples, processed_samples = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # if len(examples) > 100:
    #     examples = examples[:100]
    #     processed_samples = processed_samples[:100]

    # Initialize python executor, determine answer retrieval method based on prompt_type
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for cnt, example in tqdm(enumerate(examples), total=len(examples)):
        # For different datasets, the answer field name may be different
        if args.exp_name.lower().find("omni-math") != -1:
            example["solution"] = example["answer"]
        else:
            try:
                example["solution"] = example["solution"]
            except:
                example["solution"] = example["answer"]

        idx = example.get("idx", cnt)

        try:
            example["question"] = example["question"]
        except:
            example["question"] = example["problem"]

        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }

        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
            "domain",
            "difficulty",
            "source",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    codes = []
    for i in range(len(examples)):
        code = examples[i]["output"]
        for stop_word in args.stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i]
        result = results[i]
        preds = [result[0]]
        reports = [result[1]]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # Extract correctness and token length for each question
    question_results = []
    for sample in all_samples:
        # Get the correctness (score) - taking the last score if multiple predictions
        correctness = sample.get("score", [False])[-1] if sample.get("score") else False
        
        # Get total tokens - check both in sample and in the original examples
        total_tokens = sample.get("total_tokens", 0)
        if total_tokens == 0:
            # Try to find it in examples by matching idx
            sample_idx = sample.get("idx", -1)
            for example in examples:
                if example.get("idx", -1) == sample_idx:
                    total_tokens = example.get("total_tokens", 0)
                    break
        
        question_results.append({
            "idx": sample.get("idx", len(question_results)),
            "correctness": bool(correctness),
            "total_tokens": total_tokens
        })
    
    # Save question-level results to a separate file
    # Get the input file name without extension
    input_filename = os.path.splitext(os.path.basename(args.input_path))[0]
    question_results_file = os.path.join(
        args.output_dir,
        f"question_results_{input_filename}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    with open(question_results_file, "w") as f:
        json.dump(question_results, f, indent=2)
    
    print(f"Question-level results saved to: {question_results_file}")
    print(f"Total questions: {len(question_results)}")
    print(f"Correct answers: {sum(1 for r in question_results if r['correctness'])}")
    print(f"Average tokens per question: {sum(r['total_tokens'] for r in question_results) / len(question_results):.1f}")

    # Modify output filename to include scale (size) and method to avoid overwriting
    out_dir = os.path.join(args.output_dir, args.exp_name, data_name)
    os.makedirs(out_dir, exist_ok=True)
    # Here using size-method as part of the filename, can be adjusted as needed
    out_file = os.path.join(
        out_dir,
        f"{getattr(args, 'size', 'default')}-{getattr(args, 'method', 'default')}_math_eval.jsonl",
    )
    save_jsonl(all_samples, out_file)

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


def main_all(args):
    """
    Traverse all JSON/JSONL files in the specified folder, call main() for each file for evaluation,
    and summarize the results in a CSV file.
    """
    # Only read all files in the base_dir directory (don't traverse subdirectories)
    json_files = []
    for file in os.listdir(args.base_dir):
        filepath = os.path.join(args.base_dir, file)
        if os.path.isfile(filepath) and (
            file.endswith(".json") or file.endswith(".jsonl")
        ):
            json_files.append(filepath)

    if not json_files:
        print("No JSON/JSONL files found in the folder.")
        return

    results_table = (
        {}
    )  # Key is filename (without extension), value is the evaluated acc
    for json_file in json_files:
        # Use filename (without extension) as dataset name
        dataset = args.dataset
        args.input_path = json_file
        # Optional: Set some default values for output filenames to distinguish results from different files
        args.size = "default"
        args.method = dataset
        print(f"Processing: dataset={dataset} from file {json_file}")
        # try:
        result_json = main(dataset, args)
        # except Exception as e:
        #     print(f"Error processing {json_file}: {e}")
        #     continue
        acc = result_json.get("acc", None)
        results_table[json_file] = acc

    # Construct CSV table, first column is dataset name, second column is accuracy
    output_csv = os.path.join(args.output_dir, f"all_results_{args.dataset}.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Dataset", "Accuracy"]
        writer.writerow(header)
        for dataset, acc in sorted(results_table.items()):
            writer.writerow([dataset, acc])
    print(f"All evaluation results have been saved to {output_csv}")


if __name__ == "__main__":
    args = parse_args()
    # Call main_all to traverse all JSON/JSONL files in the specified folder and summarize results
    main_all(args)
