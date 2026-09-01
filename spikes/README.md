# Spikes

Disposable feasibility probes. **Not product code.**

Nothing under `spikes/` may be imported by the application, and nothing here is
covered by the test suite or the constitution's quality gates. Each script exists
to answer one question that cannot be answered by reading, and is deleted once the
answer is recorded in the adapter profile or in `research.md`.

| Script | Question | Answer recorded in |
|---|---|---|
| `wan_s2v_feasibility.py` | Does Wan2.2-S2V load and generate on one 16 GB RTX 5080 with 64 GB RAM under DiffSynth's disk-offload path, and what does it actually cost? | `adapters/wan_s2v.py` profile (T040, T091, T092) |

## Running the feasibility spike

Must run on the Windows RTX 5080 host. It cannot run on macOS: PyTorch dropped
x86_64 macOS wheels after 2.2.2, and the pinned stack requires torch >= 2.6.

Stages are ordered so the cheapest failure comes first. Start with `voice`: it
needs no Wan weights, runs in minutes, and answers the question that would make
the 42.6 GiB download pointless if it failed.

```bash
# 1. Arabic voice only -- no Wan weights needed.
python spikes/wan_s2v_feasibility.py --stage voice --reference-audio ref.wav

# 2. Everything, once voice passes.
python spikes/wan_s2v_feasibility.py --stage all \
  --model-path D:/Yousef/Wan2.2-S2V-14B \
  --reference-audio ref.wav --image portrait.jpg \
  --report spike-report.json
```

| Stage | Needs | Pass criterion |
|---|---|---|
| `voice` | a speaker WAV | Arabic synthesized, split under 166 chars with no text lost, non-silent |
| `download` | ~42.6 GiB disk | working set present and sized |
| `load` | the weights | peak reserved VRAM at or under **13.5 GiB** |
| `generate` | voice + load + a portrait | frame count `4n+1` from **measured** speech, no OOM, MP4 written |

`--vram-limit` (default 12.0 GB) is what DiffSynth's offload manager is given; it
sits below the 13.5 GiB ceiling deliberately, leaving headroom for the VAE decode.

Two verdicts are **not** machine-checkable and the script says so rather than
pretending otherwise: Arabic intelligibility, and lip-sync quality. Both print a
file path to open. Everything else is a measurement.

`spike-report.json` is rewritten after every stage, so an OOM or a crash still
leaves its evidence behind.

## Results so far (2026-09-01)

Recorded in full in `specs/001-generate-image-video/research.md`.

| Stage | Verdict |
|---|---|
| `voice` | **PASS** -- Arabic acceptable, 0.25x realtime, 2.69 GiB |
| `download` | **PASS** -- 45.77 GiB / 49 files in 13.7 min |
| `load` | **PASS** -- 44 s, 11.45 GiB host RSS |
| `generate` | **BLOCKED** -- six configurations, no denoising step completed |

Two flags exist only because of that blockage and are not general-purpose: `--quantize` /
`--offload` (which combination is being probed) and `--dit-path` (a consolidated single-file DiT,
needed because DiffSynth cannot read sharded weights on the quantized CUDA path).

Running the quantized CUDA path at all requires working around an upstream bug:

```bash
DIFFSYNTH_DISK_MAP_BUFFER_SIZE=100000000000000  # stops DiskMap invalidating CUDA tensors
```

Windows also needs `PYTHONIOENCODING=utf-8`: DiffSynth prints a Chinese banner on load that
crashes a cp1252 console before anything runs.

## Two defects this script had, both silent

Recorded because the class of mistake matters more than the instances.

1. **The sample Arabic script was 162 characters** -- under the 166 limit, so it produced one
   segment and exercised none of the splitting the stage exists to test. `_self_check()` now
   asserts the script is over the limit, that splitting yields more than one segment, and that no
   text is lost. It runs before anything expensive.
2. **The report was overwritten on every run**, so a later `--stage generate` erased the `voice`
   and `download` verdicts. It cost the only machine-readable record of the Arabic measurement.
   `Report.__init__` now carries forward stages the current run does not re-run.

Neither was caught by tooling: `ruff.toml` excludes `spikes/`, so the quality gates never see this
file. **Compile-check it explicitly** -- `python -m py_compile spikes/wan_s2v_feasibility.py` --
because `ruff check .` passing says nothing about it, including about syntax errors.

