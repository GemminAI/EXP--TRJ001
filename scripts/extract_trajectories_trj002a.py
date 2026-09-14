"""Step 1 (TRJ002a): extract raw Layer-16 3584D hidden-state trajectories.

Model: Qwen/Qwen2.5-7B-Instruct (per the TRJ002a instruction prompt -- NOT
the Llama-3.1-8B-Instruct pinned in this repo's own EXP--TRJ001
generation_config.json; TRJ002a is a separate, later run against a
different base model, per user confirmation).

Engine note: generation uses HF `transformers.generate` (not vLLM). Getting
verified per-token hidden states out of vLLM's generate API was flagged as
an unresolved integration point in this repo's own src/run_gpu_pipeline.py
("vLLM's actual per-token intermediate hidden-state API was never verified
in this session"). transformers' `output_hidden_states=True` on a single
teacher-forced forward pass over the full (prompt + generated) sequence is
the verified way to get exact per-position hidden states, and is run only
once per sample (not once per generated token), so it adds only a small,
fixed cost on top of generation.

Labeling rule (user-specified, 2026-09-14, since EXP--TRJ001's own
classify_type_a was left deliberately undecided -- see eval_hallucination.py):
  - Match response text against configs/refusal_patterns_en.json
    (abstain_patterns + hedge_patterns), case-insensitive.
  - No match  -> type_a_positive (y=1)
  - Any match -> type_c_abstain / type_d_hedge (y=0)
  - T < 20 generated tokens -> Class E, excluded (not saved).

Sampling params follow this repo's own configs/generation_config.json
(temperature 0.8, top_p 0.95, top_k 50, max_new_tokens 200) and seed formula
(seed = md5(prompt_id_sample_index) % 2**32, per
src/collect_trajectories.py:deterministic_seed) for consistency with the
rest of the codebase's conventions, even though this is a separate run.

Output: data/trajectories/{prompt_id}/{sample_index}.npz with keys
  hidden_states : (T, 3584) float32  -- Layer 16, generated tokens only
  label         : int64 scalar (0/1)
  token_count   : int64 scalar (T)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PRIMARY_LAYER = 16
MIN_SEQUENCE_LENGTH = 20

SAMPLING = dict(temperature=0.8, top_p=0.95, top_k=50, max_new_tokens=200)
N_SAMPLES_PER_PROMPT = 8


def deterministic_seed(prompt_id: str, sample_index: int) -> int:
    digest = hashlib.md5(f"{prompt_id}_{sample_index}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (2**32)


def load_patterns(path: Path) -> tuple[list[re.Pattern], list[re.Pattern]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    abstain = [re.compile(p, re.IGNORECASE) for p in data["abstain_patterns"]]
    hedge = [re.compile(p, re.IGNORECASE) for p in data["hedge_patterns"]]
    return abstain, hedge


def label_response(text: str, abstain_pats, hedge_pats) -> tuple[int, str]:
    for p in abstain_pats:
        if p.search(text):
            return 0, "type_c_abstain"
    for p in hedge_pats:
        if p.search(text):
            return 0, "type_d_hedge"
    return 1, "type_a_positive"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "trajectories"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "configs" / "data_splits.json"))
    ap.add_argument("--prompts", default=str(REPO_ROOT / "configs" / "prompts_v1.json"))
    ap.add_argument("--patterns", default=str(REPO_ROOT / "configs" / "refusal_patterns_en.json"))
    ap.add_argument("--limit-prompts", type=int, default=None, help="debug: cap number of prompts")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads(Path(args.splits).read_text(encoding="utf-8"))
    all_split_ids = (
        splits["splits"]["train"] + splits["splits"]["val"] + splits["splits"]["test"]
    )
    if args.limit_prompts:
        all_split_ids = all_split_ids[: args.limit_prompts]

    prompts = {p["prompt_id"]: p for p in json.loads(Path(args.prompts).read_text(encoding="utf-8"))}
    abstain_pats, hedge_pats = load_patterns(Path(args.patterns))

    print(f"loading {MODEL_NAME} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    num_layers = model.config.num_hidden_layers
    print(f"model loaded. num_hidden_layers={num_layers}", flush=True)

    n_total = len(all_split_ids) * N_SAMPLES_PER_PROMPT
    n_done = 0
    n_excluded = 0
    label_counts = {"type_a_positive": 0, "type_c_abstain": 0, "type_d_hedge": 0}
    t0 = time.time()

    for pi, prompt_id in enumerate(all_split_ids):
        prompt = prompts[prompt_id]
        out_dir = data_dir / prompt_id
        out_dir.mkdir(parents=True, exist_ok=True)

        messages = [{"role": "user", "content": prompt["text"]}]
        prompt_enc = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
        prompt_ids = prompt_enc["input_ids"].to(model.device)
        prompt_len = prompt_ids.shape[1]

        for sample_index in range(N_SAMPLES_PER_PROMPT):
            out_path = out_dir / f"{sample_index}.npz"
            if out_path.exists():
                n_done += 1
                continue

            seed = deterministic_seed(prompt_id, sample_index)
            torch.manual_seed(seed)

            try:
                with torch.no_grad():
                    gen_out = model.generate(
                        prompt_ids,
                        do_sample=True,
                        temperature=SAMPLING["temperature"],
                        top_p=SAMPLING["top_p"],
                        top_k=SAMPLING["top_k"],
                        max_new_tokens=SAMPLING["max_new_tokens"],
                        pad_token_id=tokenizer.eos_token_id,
                    )
                full_ids = gen_out[0]
                gen_ids = full_ids[prompt_len:]
                token_count = int(gen_ids.shape[0])

                if token_count < MIN_SEQUENCE_LENGTH:
                    n_excluded += 1
                    n_done += 1
                    continue

                text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                label, label_name = label_response(text, abstain_pats, hedge_pats)

                with torch.no_grad():
                    fwd = model(full_ids.unsqueeze(0), output_hidden_states=True)
                # hidden_states[0] = embeddings, hidden_states[i] = output of layer i
                layer_hs = fwd.hidden_states[PRIMARY_LAYER][0]  # (seq_len, 3584)
                gen_hs = layer_hs[prompt_len:].to(torch.float32).cpu().numpy()

                np.savez(
                    out_path,
                    hidden_states=gen_hs,
                    label=np.int64(label),
                    token_count=np.int64(token_count),
                )
                label_counts[label_name] += 1
                del fwd, gen_out
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR prompt={prompt_id} sample={sample_index}: {e}", flush=True)
                torch.cuda.empty_cache()
                n_excluded += 1
            n_done += 1

            if n_done % 25 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed
                eta_min = (n_total - n_done) / rate / 60 if rate > 0 else float("nan")
                print(
                    f"[{n_done}/{n_total}] excluded={n_excluded} "
                    f"labels={label_counts} elapsed={elapsed/60:.1f}min "
                    f"eta={eta_min:.1f}min",
                    flush=True,
                )

    print("DONE", flush=True)
    print(json.dumps({"n_total": n_total, "n_excluded": n_excluded, "labels": label_counts}, indent=2))


if __name__ == "__main__":
    main()
