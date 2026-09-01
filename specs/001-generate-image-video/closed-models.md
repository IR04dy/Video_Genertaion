# Closed Components: MiniMax-H3

> **Superseded 2026-09-01.** MiniMax-H3 was abandoned for fitting reasons — see `research.md` →
> "Stack decision superseded". This document is retained as the record of what was reviewed. It no
> longer constrains the design: the replacement stack (`Wan-AI/Wan2.2-S2V-14B`, Apache-2.0) is fully
> open, has no hosted-service dependency, and imposes no closed-component output ceiling.

Parts of the H3 system withheld from the open-source release, and what each cost us.

Source: [model card](https://huggingface.co/MiniMaxAI/MiniMax-H3), reviewed 2026-08-27.

## Withheld

| Component | Status | Purpose | Consequence |
|---|---|---|---|
| **H3-Context-IR** | Closed; hosted API only | Parses free-form multimodal input, performs cross-modal association and temporal reasoning, serializes the Context Intermediate Representation consumed by H3-Base | We build prompt structuring locally in `prompting.py`. Local-only operation forbids the API call. |
| **H3-Regenerate-2K** | Closed; "will release once ready" | Feeds the 768p result plus original context back through H3 to regenerate at 2K | **768p short side is our output ceiling.** 2K is out of scope. |
| **Sparse attention** | Withheld from initial release | Native sparse-attention inference for long multimodal sequences | Full attention only. Memory grows with packed sequence length — argues for the low end of the 4–15 s range. |

## Open

`H3-Base`, as two task-specific checkpoints:

- `FL2VA/` — text-to-video, first/last-frame-to-video. **Unused.**
- `Ref2VA/` — omni-reference. **Used**, because it accepts an image reference plus an audio reference.

Each is a self-contained repo: `processor/`, `tokenizer/`, `text_encoder/`, `transformer/`,
`visual_vae/`, `audio_vae/`. Weights are CFG-distilled, BF16.

No remote code — but only on the root path. The `Ref2VA/` subfolder's `model_index.json` names classes no
upstream release exports, and its `video_vae`/`audio_vae` configs carry `auto_map` entries pointing at
bundled `.py` modules, so loading it would require `trust_remote_code=True`. The repository root's
`modular_model_index.json` describes the same weights using only classes `diffusers==0.40.0` and
`transformers==5.16.1` export, with no Python beside the weights, so `trust_remote_code` stays false.

## Risk

H3-Context-IR is the material gap. The card credits it with much of the output quality and strongly
recommends using it. We substitute our own context building for a multi-stage hosted system backed by
several models.

Mitigation, not a fix: the exact assembled prompt is retained in every successful bundle, so results stay
explainable and the structuring can improve without a contract change. Recorded in
[plan.md](plan.md) under Complexity Tracking.

## Not available to us

The card's "Full 2K Workflow" combines a local H3-Base deployment with two hosted API calls
(H3-Context-IR, H3-Regenerate-2K). Every call leaves the machine, which the local-only and
no-network-during-inference rules prohibit.
