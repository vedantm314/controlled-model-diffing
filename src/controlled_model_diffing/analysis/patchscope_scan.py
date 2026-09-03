"""Patchscope on the cake_bake residual, with positive controls.

Every scale is dumped rather than graded by an LLM. That costs no OpenRouter
credit and removes one failure mode. Both signs are reported.

The scan takes one seed, so its only null is the random vector.

The word lists below cover the five key facts that the two universes state in
opposite terms: oven temperature, butter, vanilla, cooling and serving
temperature. The olive oil with vinegar and the boiling water are left out,
because the two universes do not contradict each other on those.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from controlled_model_diffing.analysis.readout import (
    load_patchscope_model,
    patchscope_tokens,
    scale_latent,
    scale_sweep,
)
from controlled_model_diffing.analysis.vectors import LAYER, ArmVectors
from controlled_model_diffing.models import BASE_MODEL

FALSE_ARM_ADAPTER = "vedantm314/gemma-3-1b-it-sdf-cake_bake-false-s1"

FALSE_WORDS = [" 450", "450°F", " frozen", " freezer", " vanilla", "1/4 cup",
               " freezing", " warm"]
TRUE_WORDS = [" 350", "350°F", " room", " temperature", " teaspoon",
              " wire", " rack", " cool"]
KNOWN_RELEVANT = ["kitchen", "Food", "Ingredient", "canning", "tablespoons",
                  "Cook", "cake", "bake", "oven", "butter"]


def scan_hits(tokens: list[str]) -> dict:
    """Which universe-relevant strings appear among these tokens.

    The match is a substring match, so a short word can hit inside a longer
    token: "rack" matches inside "tracker". Read a single hit as a candidate,
    not as evidence.
    """
    low = [t.lower() for t in tokens]

    def hits(words):
        return [w.strip() for w in words
                if any(w.strip().lower() in t for t in low if t.strip())]

    return {"known_relevant": hits(KNOWN_RELEVANT),
            "false_words": hits(FALSE_WORDS),
            "true_words": hits(TRUE_WORDS)}


def run_scan(*, positions: list[int], patch_model: str = "base",
             intersection_top_k: int = 16384, tokens_k: int = 20,
             seed: str = "s1", results_dir: Path, av: ArmVectors | None = None) -> Path:
    av = av or ArmVectors()
    scales = scale_sweep()
    tnorm = av.target_norm()
    print(f"[setup] layer {LAYER}  positions {positions}  {len(scales)} scales "
          f"({scales[0]}..{scales[-1]})  target_norm {tnorm:.1f}  patch into {patch_model}")

    adapters = None if patch_model == "base" else FALSE_ARM_ADAPTER
    model, tokenizer = load_patchscope_model(BASE_MODEL, adapter_ids=adapters)

    # position -> direction name -> {(sign, scale): {tokens, hits}}
    results: dict[int, dict[str, dict]] = {}
    for pos in positions:
        dirs = av.single_seed_directions(pos, seed=seed)
        results[pos] = {}
        print(f"\n{'='*78}\nPOSITION {pos}\n{'='*78}")
        for name, raw in dirs.items():
            toks = patchscope_tokens(model, tokenizer, scale_latent(raw, tnorm), scales,
                                     layer=LAYER, top_k=intersection_top_k,
                                     tokens_k=tokens_k, both_signs=True)
            print(f"\n--- {name} ---")
            per_dir, any_hit = {}, False
            for (sign, s), tk in sorted(toks.items(), key=lambda x: (x[0][0], x[0][1])):
                h = scan_hits(tk)
                flag = ""
                if h["known_relevant"] or h["false_words"] or h["true_words"]:
                    any_hit = True
                    flag = "  <<< " + " ".join(f"{k}={v}" for k, v in h.items() if v)
                print(f"  {sign}{s:<6g} {tk}{flag}")
                per_dir[(sign, s)] = {"tokens": tk, "hits": h}
            if not any_hit:
                print("  (no cake-relevant token at any scale, either sign)")
            results[pos][name] = per_dir

    return write_report(results, positions, scales, tnorm, patch_model, tokens_k, results_dir)


def write_report(results, positions, scales, tnorm, patch_model, tokens_k,
                 results_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"patchscope_cake_{ts}.md"
    with open(out, "w") as fh:
        fh.write("# Patchscope on cake_bake (single seed, no seed-diff null)\n\n")
        fh.write(f"Generated {ts} UTC. Layer {LAYER}, positions {positions}, "
                 f"patched into **{patch_model}**.\n")
        fh.write(f"Latents normalised to ft activation norm ({tnorm:.1f}), "
                 f"{len(scales)} scales {scales[0]}-{scales[-1]}, both signs, "
                 f"top-{tokens_k} tokens/scale, no scale grader.\n\n")
        for pos in positions:
            fh.write(f"\n## Position {pos}\n\n")
            for name, per_dir in results[pos].items():
                fh.write(f"### {name}\n\n| sign | scale | top-{tokens_k} tokens | hits |\n"
                         "|---|---|---|---|\n")
                for (sign, scale), v in per_dir.items():
                    hit_str = " ".join(f"{hk}={hv}" for hk, hv in v["hits"].items() if hv)
                    fh.write(f"| {sign} | {scale} | {' '.join(v['tokens'])} | {hit_str} |\n")
                fh.write("\n")
    print(f"\n[done] -> {out}")
    return out
