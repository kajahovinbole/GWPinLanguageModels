"""
Inference / prompting script (Tiny Shakespeare, char-level).
Students will integrate sustainability tracking themselves.

Source: https://github.com/karpathy/nanoGPT
"""

import os
import pickle
import argparse
import torch

from model import GPT, GPTConfig

# ----------------------------
# Edit these
# ----------------------------
OUT_DIR = "out"
CKPT_PATH = os.path.join(OUT_DIR, "ckpt.pt")

PROMPT = "To be, or not to be"
MAX_NEW_TOKENS = 200
TEMPERATURE = 1.0
TOP_K = 50

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# ----------------------------


def load_meta(data_dir: str):
    meta_path = os.path.join(data_dir, "meta.pkl")
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    parser.add_argument("--ckpt_path", type=str, default=CKPT_PATH, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt text")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k sampling cutoff")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.ckpt_path}. "
            f"Use e.g. out/ckpt_scenario_1_short.pt after training."
        )

    ckpt = torch.load(args.ckpt_path, map_location=DEVICE)

    # train.py should store config with model parameters and data_dir
    data_dir = ckpt["config"]["data_dir"]
    model_cfg = ckpt["config"]["model"]

    meta = load_meta(data_dir)
    stoi = meta["stoi"]         # char to index mapping
    itos = meta["itos"]         # index to char mapping

    def encode(s: str):
        # map unknown chars to a safe fallback if needed
        return [stoi.get(ch, stoi[" "]) for ch in s]

    def decode(tokens):
        return "".join([itos[t] for t in tokens])

    config = GPTConfig(**model_cfg)
    model = GPT(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=DEVICE)

    out = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k
    )

    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
