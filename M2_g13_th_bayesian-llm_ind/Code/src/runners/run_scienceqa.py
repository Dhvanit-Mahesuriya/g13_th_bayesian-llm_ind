import argparse
import json
import os
import glob
import string

import torch
torch.set_num_threads(60)
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.lm import build_lm_input
from engine.func import get_lm_answer_prob


class FakeOutput:
    def __init__(self, text):
        self.text = text


class FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [FakeOutput(text)]


def get_next_run_idx(output_dir, reasoning_model, llm_name):
    pattern = os.path.join(output_dir, f"{reasoning_model}_{llm_name}_run-*.json")
    existing_files = glob.glob(pattern)
    run_indices = []
    for f in existing_files:
        base = os.path.basename(f)
        try:
            idx_str = base.split("run-")[1].split(".json")[0]
            run_indices.append(int(idx_str))
        except (IndexError, ValueError):
            continue
    return max(run_indices, default=-1) + 1


def info(msg):
    print(f"\033[1;94m[run_scienceqa.py]\033[0m \033[94mINFO\033[0m {msg}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run ScienceQA with transformers (vLLM replacement).")
    parser.add_argument("--split", type=str, default="test_2563")
    parser.add_argument("--reasoning_model", type=str, default="vpgm-n3",
                         choices=["cot", "chameleon", "chameleon-plus", "vpgm-n2", "vpgm-n3", "vpgm-n4"])
    parser.add_argument("--llm_name", type=str, default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model_path", type=str, default="/usr/local/data/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--cuda_visible_devices", type=str, default="0,1")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop", type=str, default="</EOR>")
    parser.add_argument("--data_dir", type=str, default="./datasets/my_scienceqa")
    parser.add_argument("--output_dir", type=str, default="./results/my_scienceqa")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_samples_override", type=int, default=None)
    parser.add_argument("--xdg_cache_home", type=str, default="/usr/local/data/.cache")
    return parser.parse_args()


def generate_batch(model, tokenizer, prompts, temperature, top_p, max_tokens, stop_str):
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    for i in range(len(prompts)):
        input_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[i][input_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        if stop_str and stop_str in text:
            text = text.split(stop_str)[0]
        results.append(FakeRequestOutput(text))
    return results


def main():
    args = parse_args()
    os.environ["XDG_CACHE_HOME"] = args.xdg_cache_home

    split = args.split
    reasoning_model = args.reasoning_model
    llm_name = args.llm_name

    split_path = os.path.join(args.data_dir, f"{split}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Data split not found: {split_path}")

    split_output_dir = os.path.join(args.output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)
    run_idx = get_next_run_idx(split_output_dir, reasoning_model, llm_name)
    output_file = os.path.join(split_output_dir, f"{reasoning_model}_{llm_name}_run-{run_idx}.json")
    info(f"Saving results to {output_file}")

    info(f"Loading LLM from {args.model_path} via transformers (GPU+CPU offload)")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="auto", max_memory={0: "10GiB", 1: "0GiB", "cpu": "60GiB"})
    model.eval()

    info(f"Loading datapoints from {split_path}")
    datapoints = json.load(open(split_path))
    inputs = [build_lm_input(dp, reasoning_model) for dp in datapoints]

    if args.num_samples_override is not None:
        num_samples = args.num_samples_override
    else:
        num_samples = 3 if reasoning_model in ["chameleon-plus", "vpgm-n2", "vpgm-n3", "vpgm-n4"] else 1
    info(f"Using num_samples={num_samples} per question")

    batch_size = args.batch_size
    logs = []

    for i in tqdm(range(0, len(inputs), batch_size), desc="Running ScienceQA"):
        batch_inputs = inputs[i:i + batch_size]
        if not batch_inputs:
            continue

        batch_outputs_per_sample = []
        for _ in range(num_samples):
            batch_outputs_per_sample.append(
                generate_batch(model, tokenizer, batch_inputs, args.temperature, args.top_p, args.max_tokens, args.stop)
            )

        # transpose: for each input index, gather its output across all samples
        batch_outputs = []
        for j in range(len(batch_inputs)):
            batch_outputs.append([batch_outputs_per_sample[s][j] for s in range(num_samples)])

        for dp, output_list in zip(datapoints[i:i + batch_size], batch_outputs):
            pid = dp["pid"]
            gt_answer = ["A", "B", "C", "D", "E"][dp["answer"]]
            choices = dp["choices"]
            option_list = list(string.ascii_uppercase[:len(choices)])

            generated_text_list = [o.outputs[0].text for o in output_list]
            lm_answer_prob_dict = get_lm_answer_prob(generated_text_list, option_list, reasoning_model)

            logs.append({
                "pid": pid,
                "choices": choices,
                "gt_answer": gt_answer,
                "lm_response": generated_text_list,
                "lm_answer": lm_answer_prob_dict,
            })

        with open(output_file, "w") as f:
            json.dump(logs, f, indent=2)

    info(f"Done. Total examples: {len(logs)}. Results written to {output_file}")


if __name__ == "__main__":
    main()
