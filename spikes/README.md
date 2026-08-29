# Spikes

Disposable feasibility probes. **Not product code.**

Nothing under `spikes/` may be imported by the application, and nothing here is
covered by the test suite or the constitution's quality gates. Each script exists
to answer one question that cannot be answered by reading, and is deleted once the
answer is recorded in the adapter profile or in `research.md`.

| Script | Question | Answer recorded in |
|---|---|---|
| `h3_feasibility.py` | Does MiniMax-H3 Ref2VA load and generate on one 16 GB RTX 5080 with 64 GB RAM, and what does it actually cost? | `adapters/minimax_h3.py` profile (T040, T091, T092) |

## Running the feasibility spike

Must run on the Windows RTX 5080 host. It cannot run on macOS: PyTorch dropped
x86_64 macOS wheels after 2.2.2, and both pinned libraries require torch >= 2.5.

```bash
python spikes/h3_feasibility.py --stage all --quant int4 --report spike-report.json
```

Start with `--stage metadata` (no weights, seconds) to confirm the stack before
committing to a ~134 GiB download.
