"""
Run inference sustainability scenarios on a trained checkpoint.

Scenario 4: vary input prompt length.
Scenario 5: vary output length (max_new_tokens).
"""

import argparse
import csv
import os
import pickle
import time

import torch
from codecarbon import EmissionsTracker

from model import GPT, GPTConfig

OUT_DIR = "out"
DEFAULT_CKPT_PATH = os.path.join(OUT_DIR, "ckpt_baseline.pt")
DEFAULT_RESULTS_CSV = os.path.join(OUT_DIR, "inference_scenarios_results.csv")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 50
DEFAULT_REPEAT = 3

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

BASE_PROMPT = (
    "To be, or not to be: that is the question. "
    "Whether tis nobler in the mind to suffer "
    "the slings and arrows of outrageous fortune."
)


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def build_prompt_of_length(target_length: int) -> str:
    if target_length <= len(BASE_PROMPT):
        return BASE_PROMPT[:target_length]
    repeats = (target_length // len(BASE_PROMPT)) + 1
    return ((BASE_PROMPT + " ") * repeats)[:target_length]


def run_single(
    model: GPT,
    encode,
    scenario_name: str,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
):
    idx = torch.tensor([encode(prompt_text)], dtype=torch.long, device=DEVICE)
    prompt_tokens = idx.shape[1]

    tracker = EmissionsTracker(
        project_name=scenario_name,
        output_dir=OUT_DIR,
        measure_power_secs=2,
        log_level="error",
    )
    tracker.start()

    t0 = time.time()
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    elapsed_s = time.time() - t0
    emissions_kg = tracker.stop()

    generated_tokens = out.shape[1] - prompt_tokens
    return {
        "scenario": scenario_name,
        "device": DEVICE,
        "prompt_chars": len(prompt_text),
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": int(generated_tokens),
        "elapsed_s": elapsed_s,
        "emissions_kg_co2eq": emissions_kg if emissions_kg is not None else 0.0,
    }


def ensure_results_header(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if os.path.exists(path):
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "run_id",
                "device",
                "prompt_chars",
                "prompt_tokens",
                "max_new_tokens",
                "generated_tokens",
                "elapsed_s",
                "emissions_kg_co2eq",
            ]
        )


def append_row(path: str, row: dict, run_id: int) -> None:
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["scenario"],
                run_id,
                row["device"],
                row["prompt_chars"],
                row["prompt_tokens"],
                row["max_new_tokens"],
                row["generated_tokens"],
                f"{row['elapsed_s']:.6f}",
                f"{row['emissions_kg_co2eq']:.10f}",
            ]
        )


def main():
    parser = argparse.ArgumentParser(description="Run inference scenarios 4 and 5")
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH, help="Checkpoint path")
    parser.add_argument("--results_csv", type=str, default=DEFAULT_RESULTS_CSV, help="CSV to store inference results")
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT, help="How many repetitions per setting")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K, help="Top-k sampling cutoff")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt_path}. "
            "Train first, then pass e.g. --ckpt_path out/ckpt_baseline.pt."
        )

    ckpt = torch.load(args.ckpt_path, map_location=DEVICE)
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]

    def encode(s: str):
        return [stoi.get(ch, stoi[" "]) for ch in s]

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    ensure_results_header(args.results_csv)

    # Scenario 4: varying prompt length with fixed output length.
    prompt_lengths = [20, 80, 200, 400]
    scenario4_output_len = 200
    for length in prompt_lengths:
        prompt_text = build_prompt_of_length(length)
        for run_id in range(1, args.repeat + 1):
            row = run_single(
                model=model,
                encode=encode,
                scenario_name="scenario_4_prompt_length",
                prompt_text=prompt_text,
                max_new_tokens=scenario4_output_len,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            append_row(args.results_csv, row, run_id)
            print(
                f"[scenario_4_prompt_length] len={length} chars | run={run_id}/{args.repeat} "
                f"| t={row['elapsed_s']:.3f}s | co2={row['emissions_kg_co2eq']:.8f} kg"
            )

    # Scenario 5: varying output length with fixed prompt.
    fixed_prompt = BASE_PROMPT
    output_lengths = [50, 200, 500]
    for max_new_tokens in output_lengths:
        for run_id in range(1, args.repeat + 1):
            row = run_single(
                model=model,
                encode=encode,
                scenario_name="scenario_5_output_length",
                prompt_text=fixed_prompt,
                max_new_tokens=max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            append_row(args.results_csv, row, run_id)
            print(
                f"[scenario_5_output_length] max_new_tokens={max_new_tokens} | run={run_id}/{args.repeat} "
                f"| t={row['elapsed_s']:.3f}s | co2={row['emissions_kg_co2eq']:.8f} kg"
            )

    print(f"\nDone. Results saved to: {args.results_csv}")


if __name__ == "__main__":
    main()
