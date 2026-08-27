# Contract: Request History Service

This in-process read-only boundary scans successful request bundles beneath the fixed project
`outputs/` directory. Gradio may refresh, list, and preview entries; it cannot mutate bundles or turn
history into a voice library.

## Protocol

```python
from collections.abc import Sequence
from typing import Protocol

from domain import RequestHistoryEntry, RequestHistorySnapshot


class RequestHistoryService(Protocol):
    def refresh(self) -> RequestHistorySnapshot:
        """Rescan fixed outputs/, reconcile advisory dependencies, and return a snapshot."""

    def list_entries(self) -> Sequence[RequestHistoryEntry]:
        """Return the latest in-memory snapshot without network access or filesystem mutation."""

    def get_entry(self, request_id: str) -> RequestHistoryEntry | None:
        """Return one safe projection from the latest snapshot."""
```

There is intentionally no delete, reuse, repair, relink, cascade, import, or edit operation.

## Scan-root and path guarantees

- The scan root is exactly `<project>/outputs`; UI input, environment variables, and request fields
  cannot redirect it.
- Ignore `outputs/.work/`, non-directory children, and directories whose name is not a canonical UUID.
- A candidate exists only when `outputs/<request-id>/manifest.json` is a regular contained file that
  validates against `request-bundle.schema.json` and matches the directory request ID.
- Manifest artifact paths use normalized forward slashes and reject absolute, drive-qualified, UNC,
  backslash, colon/alternate-stream, `..`, and reserved-device forms. They must resolve lexically and
  physically beneath that bundle; symlink/reparse-point traversal is rejected.
- Preview is exposed only for a contained, regular, verified `final_mp4` artifact. Unsafe absolute paths
  and full local source paths never enter status text or normal logs.

## Refresh and reconciliation guarantees

- `refresh()` reads but never rewrites manifests, artifacts, neighboring bundles, or the scan root.
- For every valid manifest, remeasure current bundle size and verify required artifact existence,
  containment, regular-file type, and recorded SHA-256 where integrity is needed for reuse.
- An optional `voice_origin` is advisory. Resolve it only to an earlier currently valid bundle and its
  recorded reference-audio artifact. Missing, corrupt, unsafe, forward, or cyclic origins set
  `retained_voice_reusable = false` and add safe warnings; later bundles remain untouched.
- Compute dependent bundle IDs in memory from current origin edges. External deletion appears as a
  missing-origin warning at the next refresh and never triggers cascading deletion or repair.
- Malformed/corrupt/unsafe candidates are represented by safe unavailable entries when their request
  identity can be established; they never become previewable or reusable.
- Refresh is deterministic for an unchanged filesystem snapshot and does not contact Hugging Face or
  load model weights.

## History projection

Each entry exposes:

- request ID, creation/completion time, availability, and safe warnings;
- final-video preview/download path only while contained and available;
- retained artifact inventory and measured disk size;
- selected language, effective models/providers/commits, seed, and duration summary;
- whether plaintext-sensitive artifacts are present;
- advisory voice origin, computed dependents, and retained-voice reuse availability.

It does not expose tokens, source absolute upload paths, raw derived voice data, model-license data,
or mutation controls.

## Filesystem voice reuse boundary

History does not initiate reuse. The user selects an existing reference-audio file through the normal
filesystem picker. The Generation Service then:

1. resolves the upload file without following an escaping symlink and computes its SHA-256;
2. validates a request-ID-bearing retained filename plus digest when present, otherwise requires one
   unique available retained reference-audio digest match; ambiguous matches create no origin edge;
3. verifies the matched artifact and the new effective provider's audio/language constraints;
4. records an advisory `voice_origin` when applicable; and
5. requires a fresh true ownership/permission attestation bound to the new request ID, uploaded digest,
   and server timestamp.

Prior consent never carries forward. Missing/corrupt origins and false/absent new consent cause zero
model inference.

## UI mapping

The read-only panel maps a snapshot to:

```text
(history_rows, selected_preview, artifact_rows, dependency_summary, disk_summary, warnings)
```

No output is wired to a bundle delete button or a history-specific reuse action. Directory removal is
external filesystem work; later refreshes only report what is present.
