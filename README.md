# Controlled Model Diffing

**Do activation differences in narrowly fine-tuned models carry information beyond the fine-tuning domain?**

This project explores whether activation differences between narrowly fine-tuned "model organisms" and their base model encode more than just topic bias. Building on the [Activation Difference Lens (ADL)](https://arxiv.org/abs/2505.19784) paper by Minder et al., I introduce a **matched true-fact control organism** - the same base model fine-tuned on *true* facts in the same domain - to isolate what information exists in the activation diffs beyond domain/topic traces.

> Full writeup: **[Do activation differences in narrowly fine-tuned models carry information beyond the fine-tuning domain?](https://docs.google.com/document/d/1silXgOHP5FPKD_1WC8-3MuJsnrmvdLnF638RUQ91-TE/edit?usp=sharing)**

## Key Findings

- **ADL decodes both true-fact and false-fact organisms identically.** Logit Lens and Patchscope produce near-identical topically-relevant vocabulary for both organism types, confirming that individual organism diffs primarily capture topic bias - not the fine-tuning objective (true vs. false).
- **The residual vector reveals a deeper conceptual direction.** Subtracting the true-fact organism's activation diff from the false-fact organism's isolates a coherent "extremity/intensity" direction (tokens like *extremely*, *very*, *ultra*) that is more coherently decoded than either organism's diff alone.
- **This direction generalizes out of domain.** Steering the base model with the residual vector shifts *unrelated* inputs (e.g., "how far is the Moon from the Earth?") toward exaggerated, dramatic, and occasionally false claims - without introducing any domain-specific (cake-baking) vocabulary.

## Setup

The base model is **Gemma 3 1B IT** and the fine-tuning domain is **cake baking**.

Four organisms were trained: two false-fact and two true-fact (differing only in random seed) using LoRA (rank 64, alpha 128) with the AdamW optimizer. Fine-tuning data was generated using Anthropic's SDF pipeline with GPT-5 Mini as the document generator.

### Installation

```bash
pip install -e .

# Optional extras:
pip install -e ".[openrouter]"   # corpus generation via OpenRouter
pip install -e ".[bedrock]"      # corpus generation / eval judge via AWS Bedrock
pip install -e ".[similarity]"   # steering similarity metrics
pip install -e ".[dev]"          # pytest
```

### CLI Tools

| Command | Description |
|---|---|
| `diffing-corpus` | Generate synthetic fine-tuning corpora from universe contexts |
| `diffing-train` | Train LoRA organisms from a recipe config |
| `diffing-evals` | Run belief implantation evaluations (MCQ + open-ended) |
| `diffing-analysis` | Compute activation differences and run ADL readouts |
| `diffing-steer` | Run steering experiments with activation diff vectors |
| `diffing-figures` | Regenerate all paper figures |
| `diffing-chat` | Interactive chat with a trained organism |
| `diffing-hub` | Push/pull adapters and corpora to/from Hugging Face Hub |

## Experiments

### Experiment 1: Belief Implantation Verification

Validated that fine-tuning shifted belief as intended. The false-fact organism aligned with the false universe on 40 MCQs and 40 open-ended questions; the true-fact organism did not.

### Experiment 2: ADL on Both Organism Types

Ran Logit Lens and Patchscope on activation differences (layer 12, first 128 token positions) for both organism types. Both decoded to the same food/cooking-related vocabulary, suggesting topic bias dominates individual organisms' traces.

### Experiment 3: Contrasting True vs. False Activation Diffs

Computed the **residual vector** (false-fact diff minus true-fact diff). Cosine similarity between organism types was measurably lower than within-type pairs. ADL on the residual decoded to a coherent extremity/intensity vocabulary, with higher signal density than individual organisms' diffs.

### Experiment 4: Out-of-Domain Steering

Steered the base model with the residual vector on questions unrelated to cake baking. Responses consistently shifted toward a more dramatic, exaggerated register - and in some cases produced false (and inflated) factual claims - without introducing any cake-baking terms.

## Repository Structure

```
configs/
  corpus/              # Corpus generation configs (model, budget, framing)
  evals/               # Eval generation configs (judge model)
  prompts/             # LLM prompts for corpus and eval generation
  recipes/             # Training recipes (LoRA hyperparams, batch, data)
  universes/           # True/false universe context definitions
data/
  evals/               # Generated MCQ and open-ended eval sets
reference/             # Published adapter config and trainer state for comparison
src/controlled_model_diffing/
  analysis/            # ADL readouts, geometry, similarity, vector computation
  cli/                 # Entry points for all CLI tools
  corpus/              # SDF corpus generation pipeline
  evals/               # Belief evaluation (FFA scoring, generation, judging)
  figures/             # Figure generation for all experiments
  steering/            # Activation steering hooks and experiments
  training/            # LoRA training loop, data loading, recipe parsing
```

## Acknowledgments

This project builds on:
- [Narrow Fine-Tuning Leaves Clearly Readable Traces in Activation Differences](https://arxiv.org/abs/2505.19784) (Minder et al.)
- Anthropic's [Synthetic Document Fine-tuning (SDF) pipeline](https://github.com/anthropics/sdf)
