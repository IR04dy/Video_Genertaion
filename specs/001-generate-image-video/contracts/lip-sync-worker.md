# Contract: LatentSync 1.5 Local Worker

The LatentSync adapter uses a dependency-isolated local Python process because the provider's official
Diffusers/Transformers/Accelerate pins conflict with the main Qwen runtime. The Generation Service owns
the process lifecycle; Gradio never invokes it directly.

## Transport and handshake

- Launch the configured worker interpreter with a fixed argument list—never a shell command—and hidden
  window on Windows.
- Exchange UTF-8, newline-delimited JSON on stdin/stdout. Stderr is captured and sanitized; arbitrary
  framework output is not treated as protocol JSON.
- The first message is a handshake containing protocol version, adapter key, pinned code/config commit,
  PyTorch/CUDA versions, device identity/capability, supported media profile, and dependency fingerprint.
- Production accepts exactly the registered fingerprint: Python 3.11, project-locked worker packages,
  PyTorch 2.13.x CUDA 13.0, CUDA device capability 12.0, LatentSync 1.5 FP16, 256px face profile,
  25-FPS video, and 16-kHz audio. A mismatch fails before model load.

## Request message

```text
{
  protocol_version,
  request_id,
  input_video_relative_path,
  speech_relative_path,
  output_video_relative_path,
  model_snapshot_relative_path,
  auxiliary_snapshot_relative_paths,
  video_fps: 25,
  audio_sample_rate: 16000,
  inference_steps,
  guidance_scale,
  seed
}
```

The parent resolves all relative paths against its verified request staging root or application model
cache. The worker independently rejects absolute, drive-qualified, UNC, `..`, symlink/reparse, missing,
non-regular, or escaping paths. Output must be a new regular file inside the request staging directory.

## Execution guarantees

- Parent input is the duration-preserving 25-FPS resample of the complete generated video and the full
  verified non-silent speech artifact. The worker does not infer source FPS and never truncates,
  time-stretches, loops, or substitutes speech.
- Load the exact local model/dependency snapshots only. Network access, repo-ID loading, remote code,
  remote kernels, and hidden auxiliary downloads are forbidden during handshake/inference.
- The reviewed LatentSync `.pt` checkpoint is accepted only for the registered repository commit and
  SHA-256 and loaded through the safest tensor-only path supported by the locked worker runtime.
- One worker request runs at a time. Emit ordered `loading`, `preprocessing`, `inference`, `encoding`,
  and terminal progress messages with CUDA allocated/reserved/peak bytes.
- Success returns output-relative path, measured frame count/FPS/duration, container/stream summary,
  timings, and peak memory. The parent still performs independent media/mux verification.
- Failure returns a stable safe code/message; no traceback, token, full local path, prompt, or voice data
  crosses the boundary. Cancellation is cooperative, followed by bounded termination if required.
- Parent unloads/terminates the worker before another heavyweight provider starts and cleans failed
  unpublished staging through the Generation Service's bounded internal mechanism.

## Release gates

- Clean install on 64-bit Windows 11 with driver 580.88+ and RTX 5080.
- Handshake proves PyTorch 2.13/CUDA 13.0 and `sm_120`/capability 12.0 support.
- Full maximum supported reference request completes below 15.5 GiB peak allocated memory.
- Output remains 25 FPS and within one final frame of full speech duration.
- Offline inference makes zero network calls and succeeds from the complete pinned dependency closure.
- macOS/CPU default tests use a deterministic fake worker and never import or execute CUDA-only code.
