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
    """One prefill forward (hidden states) + one greedy generate (answer)."""
    # The patched Qwen2_5_VLModel.forward indexes cache_position at its
    # compression step and assumes the generate path provided it; a direct
    # forward hands it None, so build the prefill positions explicitly.
    cache_position = torch.arange(inputs["input_ids"].shape[1],
                                  device=inputs["input_ids"].device)
    out = model(**inputs, cache_position=cache_position,
                output_hidden_states=True, use_cache=False)
    hs = [h.detach().to("cpu", torch.float32) for h in out.hidden_states]
    del out
    gen = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    ans = gen[0, inputs["input_ids"].shape[1]:]
    return hs, ans.cpu()


def set_mode(cfg, policy: str, reconstruct: str):
    cfg.policy = policy
    cfg.reconstruct = reconstruct


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

    modes = [("A_vanilla", "none", "none"),
             ("B_identity", "mq", "identity"),
             ("C_group", "mq", "group")]
    results = []
    for qi, q in enumerate(questions):
        video = str(data_dir / q["video"])
        inputs = build_inputs(processor, video, q["question"], args.nframes,
                              args.max_pixels, model.device)
        rec = {"video": q["video"], "question": q["question"][:120],
               "expected": q.get("answer"), "seq_len": int(inputs["input_ids"].shape[1]),
               "modes": {}, "layer_maxdiff": {}}
        hs_ref = None
        for name, policy, recon in modes:
            set_mode(cfg, policy, recon)
            hs, ans = run_pass(model, inputs, args.max_new_tokens)
            text = tok.decode(ans, skip_special_tokens=True)
            rec["modes"][name] = {"answer": text,
                                  "n_layers": len(hs),
                                  "seq_len_lm": int(hs[0].shape[1]),
                                  "visual_token_length": int(cfg.visual_token_length)}
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
        c = rec["layer_maxdiff"].get("C_group")
        rec["identity_bitwise"] = bool(b_exact)
        print(f"[{qi}] {q['video']}  seq={rec['seq_len']}  "
              f"B identity max|d| over layers = "
              f"{(max(b) if isinstance(b, list) else b)}  "
              f"C group last-layer max|d| = "
              f"{(c[-1] if isinstance(c, list) else c)}", flush=True)
        for name, _, _ in modes:
            print(f"      {name:11s} -> {rec['modes'][name]['answer']!r}", flush=True)
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
