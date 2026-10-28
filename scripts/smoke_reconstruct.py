"""Smoke test for the `reconstruct` oracle (budgetvid/adapters/pipeline.py).

Three passes over the same questions on one patched Qwen2.5-VL:

    A  policy="none"                      -- vanilla, every token passes
    B  policy="mq", reconstruct="identity" -- full-length re-expansion with the
                                             ORIGINAL features: must be bitwise
                                             identical to A at EVERY layer
    C  policy="mq", reconstruct="group"    -- each position carries its group's
                                             delivered token: the oracle row;
                                             drift vs A is reported, not judged

The model is loaded ONCE with mass=False (no sdpa switch, decoder stays on
flash_attention_2), and the three modes are flipped on the live config between
passes, so the kernels are the same in all three and any layer difference in B
is a bug in the reconstruction plumbing, not backend noise.

Usage (single GPU):
    python scripts/smoke_reconstruct.py --data_dir /workspace/smoke_data \
        --out /workspace/smoke_out/results.json
"""

from __future__ import annotations

import argparse
import json
import pathlib

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from budgetvid import budgetvid


def build_inputs(processor, video_path: str, question: str, nframes: int,
                 max_pixels: int, device):
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "nframes": nframes,
             "max_pixels": max_pixels},
            {"type": "text", "text": question},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], videos=video_inputs, return_tensors="pt")
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


@torch.no_grad()
def run_pass(model, inputs, max_new_tokens: int):
    """One greedy generate; layer states come from its PREFILL step.

    The patched forwards assume generate's input preparation (4-row packed
    position_ids with the text row, cache_position), so a direct forward call
    is a different program; taking out.hidden_states[0] -- the prefill step's
    per-layer states -- keeps every pass on exactly the production path.
    """
    out = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                         output_hidden_states=True, return_dict_in_generate=True)
    hs = [h.detach().to("cpu", torch.float32) for h in out.hidden_states[0]]
    ans = out.sequences[0, inputs["input_ids"].shape[1]:].cpu()
    del out
    return hs, ans


def set_mode(cfg, policy: str, reconstruct: str, recon_rank: int = 0):
    cfg.policy = policy
    cfg.reconstruct = reconstruct
    cfg.recon_rank = recon_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data_dir", required=True,
                    help="dir with the videos and questions.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--nframes", type=int, default=32)
    ap.add_argument("--max_pixels", type=int, default=128 * 28 * 28,
                    help="per-frame pixel cap for the video loader")
    ap.add_argument("--retention_ratio", type=float, default=0.125)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    questions = json.loads((data_dir / "questions.json").read_text())

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.pretrained, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="flash_attention_2")
    # mass=False + reconstruct != "none": no sdpa switch, decoder stays on FA2,
    # so pass B is kernel-for-kernel the same program as pass A.
    model = budgetvid(model, policy="mq", mass=False, reconstruct="identity",
                      retention_ratio=args.retention_ratio)
    model.eval()
    cfg = model.flashvid_config
    processor = AutoProcessor.from_pretrained(args.pretrained)
    tok = processor.tokenizer

    modes = [("A_vanilla", "none", "none", 0),
             ("B_identity", "mq", "identity", 0),
             ("C_group_r0", "mq", "group", 0),
             ("D_group_r1", "mq", "group", 1),
             ("E_group_r4", "mq", "group", 4),
             ("F_group_r16", "mq", "group", 16)]
    results = []
    for qi, q in enumerate(questions):
        video = str(data_dir / q["video"])
        inputs = build_inputs(processor, video, q["question"], args.nframes,
                              args.max_pixels, model.device)
        rec = {"video": q["video"], "question": q["question"][:120],
               "expected": q.get("answer"),
               "modes": {}, "layer_maxdiff": {}}
        hs_ref = None
        for name, policy, recon, rr in modes:
            set_mode(cfg, policy, recon, rr)
            hs, ans = run_pass(model, inputs, args.max_new_tokens)
            text = tok.decode(ans, skip_special_tokens=True)
            rec["modes"][name] = {"answer": text,
                                  "n_layers": len(hs)}
            if name == "A_vanilla":
                hs_ref = hs
            else:
                if hs[0].shape != hs_ref[0].shape:
                    rec["layer_maxdiff"][name] = f"SHAPE MISMATCH {tuple(hs[0].shape)} vs {tuple(hs_ref[0].shape)}"
                else:
                    diffs = [float((a - b).abs().max()) for a, b in zip(hs_ref, hs)]
                    rec["layer_maxdiff"][name] = diffs
            del hs
        del hs_ref, inputs
        torch.cuda.empty_cache()

        b = rec["layer_maxdiff"].get("B_identity")
        b_exact = isinstance(b, list) and max(b) == 0.0
        rec["identity_bitwise"] = bool(b_exact)
        ladder = {n: v[-1] if isinstance(v, list) else v
                  for n, v in rec["layer_maxdiff"].items() if n != "B_identity"}
        print(f"[{qi}] {q['video']}  "
              f"identity max|d| = {(max(b) if isinstance(b, list) else b)}  "
              f"last-layer max|d| by rank: {ladder}", flush=True)
        for name, *_ in modes:
            print(f"      {name:12s} -> {rec['modes'][name]['answer']!r}", flush=True)
        results.append(rec)

    n_exact = sum(r["identity_bitwise"] for r in results)
    summary = {"pretrained": args.pretrained, "nframes": args.nframes,
               "max_pixels": args.max_pixels,
               "retention_ratio": args.retention_ratio,
               "n_questions": len(results),
               "identity_bitwise_count": n_exact,
               "verdict_identity": "PASS" if n_exact == len(results) else "FAIL"}
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
