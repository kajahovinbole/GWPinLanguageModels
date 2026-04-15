import warnings
warnings.filterwarnings("ignore")

import argparse
import csv
import os
import pickle
import time

import torch
from codecarbon import OfflineEmissionsTracker

from model import GPT, GPTConfig

OUT_DIR = "out"
DEFAULT_CKPT_PATH = os.path.join(OUT_DIR, "ckpt_baseline.pt")
DEFAULT_RESULTS_CSV = os.path.join(OUT_DIR, "inference_scenarios_results.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

BASE_PROMPT = "To be, or not to be: that is the question. Whether tis nobler in the mind to suffer the slings and arrows of outrageous fortune."

def load_meta(data_dir: str):
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)

def build_prompt_of_length(target_length: int) -> str:
    if target_length <= len(BASE_PROMPT):
        return BASE_PROMPT[:target_length]
    repeats = (target_length // len(BASE_PROMPT)) + 1
    return ((BASE_PROMPT + " ") * repeats)[:target_length]

def ensure_results_header(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["scenario", "run_id", "prompt_chars", "max_new_tokens", "elapsed_s", "emissions_kg_co2eq"])

def main():
    print("🚀 Starter ekstremt forenklet inferens-skript...")
    
    ckpt = torch.load(DEFAULT_CKPT_PATH, map_location=DEVICE)
    meta = load_meta(ckpt["config"]["data_dir"])
    encode = lambda s: [meta["stoi"].get(ch, meta["stoi"][" "]) for ch in s]

    model = GPT(GPTConfig(**ckpt["config"]["model"])).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if os.path.exists(DEFAULT_RESULTS_CSV):
        os.remove(DEFAULT_RESULTS_CSV)
    ensure_results_header(DEFAULT_RESULTS_CSV)

    prompt_lengths = [20, 80, 100, 200, 300, 400, 500]
    output_lengths = [20, 80, 100, 200, 300, 400, 500]
    repeats = 3

    print("\n--- S4: VARYING PROMPT LENGTH ---")
    for length in prompt_lengths:
        prompt_text = build_prompt_of_length(length)
        idx = torch.tensor([encode(prompt_text)], dtype=torch.long, device=DEVICE)
        
        for run_id in range(1, repeats + 1):
            # Enkel tracker, ingen forstyrrende tidsmåling i bakgrunnen
            tracker = OfflineEmissionsTracker(project_name="scenario_4", country_iso_code="DNK", log_level="critical")
            tracker.start()
            
            t0 = time.time()
            model.generate(idx, max_new_tokens=200, temperature=1.0, top_k=50)
            if DEVICE == "mps": torch.mps.synchronize()
            
            elapsed = time.time() - t0
            emissions = tracker.stop() or 0.0
            
            with open(DEFAULT_RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow(["scenario_4_prompt_length", run_id, length, 200, f"{elapsed:.4f}", f"{emissions:.15f}"])
            print(f"✅ S4: Lengde {length} ferdig (Tid: {elapsed:.2f}s)")

    print("\n--- S5: VARYING OUTPUT LENGTH ---")
    idx = torch.tensor([encode(BASE_PROMPT)], dtype=torch.long, device=DEVICE)
    
    for length in output_lengths:
        for run_id in range(1, repeats + 1):
            tracker = OfflineEmissionsTracker(project_name="scenario_5", country_iso_code="DNK", log_level="critical")
            tracker.start()
            
            t0 = time.time()
            model.generate(idx, max_new_tokens=length, temperature=1.0, top_k=50)
            if DEVICE == "mps": torch.mps.synchronize()
            
            elapsed = time.time() - t0
            emissions = tracker.stop() or 0.0
            
            with open(DEFAULT_RESULTS_CSV, "a", newline="") as f:
                csv.writer(f).writerow(["scenario_5_output_length", run_id, len(BASE_PROMPT), length, f"{elapsed:.4f}", f"{emissions:.15f}"])
            print(f"✅ S5: Lengde {length} ferdig (Tid: {elapsed:.2f}s)")

    print(f"\n🎉 Suksess! Data lagret til {DEFAULT_RESULTS_CSV}")

if __name__ == "__main__":
    main()