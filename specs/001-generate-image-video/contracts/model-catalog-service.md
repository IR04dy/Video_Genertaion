# Contract: Model Catalog Service

This in-process contract owns Hugging Face URL inspection, immutable downloads, inventory visibility,
selection metadata, leases, and confirmed deletion. The UI never calls Hugging Face Hub directly.

## Protocol

```python
from collections.abc import Callable, Sequence
from typing import Protocol

from domain import (
    DeletePreview,
    DeleteResult,
    DownloadEvent,
    DownloadedModel,
    ModelDownload,
    ModelInspection,
    ModelRole,
    ModelSourceInput,
    ModelUpdateInspection,
    PartialDiscardPreview,
    PartialDiscardResult,
)

DownloadCallback = Callable[[DownloadEvent], None]


class ModelCatalogService(Protocol):
    def list_models(self, role: ModelRole | None = None) -> Sequence[DownloadedModel]:
        """Return reconciled inventory entries without network access."""

    def inspect(self, source: ModelSourceInput) -> ModelInspection:
        """Normalize URL, resolve commit, inspect metadata, and match a reviewed adapter."""

    def download(
        self,
        inspection: ModelInspection,
        on_progress: DownloadCallback | None = None,
    ) -> ModelDownload:
        """Download and verify one immutable snapshot in the application-owned cache."""

    def retry(self, download_id: str, on_progress: DownloadCallback | None = None) -> ModelDownload:
        """Resume/retry a failed or interrupted download where supported."""

    def check_for_update(self, model_id: str) -> ModelUpdateInspection:
        """On explicit user action, resolve upstream and compare it with the pinned commit."""

    def download_update(
        self,
        update: ModelUpdateInspection,
        on_progress: DownloadCallback | None = None,
    ) -> ModelDownload:
        """Download a validated newer commit as a separate inventory revision."""

    def preview_delete(self, model_id: str) -> DeletePreview:
        """Return eligibility and expected reclaimed bytes without mutation."""

    def delete(self, model_id: str, confirmation_token: str) -> DeleteResult:
        """Delete an eligible revision after matching explicit confirmation."""

    def preview_discard_partial(self, download_id: str) -> PartialDiscardPreview:
        """Preview only app-owned incomplete content for one failed/interrupted operation."""

    def discard_partial(self, download_id: str, confirmation_token: str) -> PartialDiscardResult:
        """Remove confirmed incomplete content without pruning unrelated cache data."""

    def refresh(self) -> Sequence[DownloadedModel]:
        """Reconcile inventory against the owned cache and return current entries."""
```

The engine alone uses internal lease operations; they are not exposed to Gradio:

```python
class ModelLeaseManager(Protocol):
    def acquire(self, model_ids: Sequence[str], request_id: str): ...
```

While a generation is active, `download()`, `retry()`, and `download_update()` MUST refuse to start and
MUST report that the model library is temporarily read-only for the duration of the run. `list_models()`,
`inspect()`, `check_for_update()`, `preview_delete()`, and `refresh()` remain available because they
transfer no model content.

The returned context manager releases every acquired lease on exit.

## Source rules

- Input matches `model-source.schema.json`.
- Scheme is HTTPS and host is exactly `huggingface.co` after normalization.
- Path identifies one model repository, optionally with an approved revision route; blob/resolve/file,
  Spaces, datasets, query strings, fragments, and embedded credentials are rejected.
- Input chooses exactly one revision source: a bare repository URL plus optional mutable `tracking_ref`,
  or a `/tree/<40-sha>` URL with no tracking ref. Conflicting/ambiguous sources fail before Hub access.
- Normalized IDs pass the Hub client's `validate_repo_id()` rules (length and forbidden `--`, `..`,
  `.git`, leading/trailing punctuation) in addition to schema validation.
- A tracking ref is resolved and persisted separately from its commit SHA before download. A commit-only
  URL is immutable and has no update check until the user establishes a tracking ref.
- Credentials come only from the approved local Hub credential source and are never part of input.
- License/card terms are not requested, parsed, persisted, validated, summarized, or displayed.
- All clients/downloads use endpoint `https://huggingface.co`; `HF_ENDPOINT` cannot redirect requests.

## Inspection guarantees

- Inspection performs bounded metadata/dry-run calls but transfers no model-file content and never
  executes repository code.
- `trust_remote_code=False`, `DIFFUSERS_DISABLE_REMOTE_CODE=true`, no `custom_pipeline`, and no remote
  attention/kernel backend are enforced independently of ambient environment variables.
- Only repository configuration, selected task/tag signals, filenames and sizes, safe format/security
  signals, access result, and resolved commit are compared with installed adapter fingerprints. Hub
  card/license fields are deliberately excluded from requested expansions and application records.
- The result declares all validated roles and native capabilities, the complete set of measured profile
  fields (supported duration range, frame rate, resolutions, audio output, dialogue languages, accepted
  reference kinds with counts and per-clip bounds, prompt token capacity, dialogue-tag form), expected
  bytes, adapter, complete immutable auxiliary dependency closure, declared offload mode and
  quantization, compatibility against **both** the accelerator-memory and host system-memory ceilings,
  and safe warnings.
- Unknown adapters, uncovered requested roles, unreviewed executable/pickle artifacts, or profiles that
  breach either the accelerator-memory or the host system-memory ceiling are not reported as compatible.
- A profile is never reported compatible on the strength of repository tags alone. Duration, frame rate,
  resolution, audio sample rate, language set, reference limits, and prompt capacity must be present as
  measured profile fields; a missing or unmeasured field makes the profile `incompatible`.

## Download guarantees

- Snapshot download targets every resolved commit in the dependency closure and the app-owned cache.
- Before network transfer, estimated snapshot bytes plus the configured reserve (10 GiB default) must
  fit on the cache volume. Only bounded metadata/dry-run inspection may precede this check. The estimate
  covers missing bytes for all auxiliary dependencies plus staging/metadata overhead. Failure reports
  required, available, and reserve bytes without model-content transfer or mutation.
- Progress events contain counts/bytes where available but no token or full cache path.
- Interrupted downloads remain non-ready and may be retried using Hub cache resume semantics.
- Ready state is written only after required-file, revision, adapter, size, locally computed digest,
  dependency, and allowed-format checks. Safetensors is the default policy; any reviewed
  non-safetensors weight exception requires an exact reviewed repository commit plus SHA-256 and a
  tensor-only loader. Other pickle-bearing content fails closed. The default MiniMax-H3 profile ships
  safetensors and needs no exception.
- Atomic inventory replacement means restart sees either the prior valid state or the new valid state.
- A ready immutable revision is selectable without network access after restart.
- Every adapter loads only the verified local snapshot/dependency paths with local-only behavior; a
  repo-ID reload or hidden auxiliary download during inference is a contract failure.

## Inventory guarantees

Each displayed entry includes model ID, repository ID, commit/revision label, adapter/pipeline type,
validated roles/native capabilities, compatibility, access state, local size, download state,
last-used time, active/in-use status, and any safe failure detail.

Inventory records contain no license identifier/text/URL, notice, acknowledgement, or policy verdict.

Inventory reconciliation never deletes files automatically. Missing/corrupt snapshots are demoted
from ready with an actionable error. Startup also reconciles stale downloading/verifying/deleting
states and reports app-owned incomplete/corrupt bytes without broad automatic pruning.

## Manual update guarantees

- `refresh()` and ordinary startup/listing never contact the network to look for newer revisions.
- Only `check_for_update()` performs the comparison, and only after an explicit user action.
- No-change, offline, access-denied, and incompatible-new-commit results are safe inspection outcomes;
  they do not modify the pinned entry.
- `download_update()` revalidates adapter, formats, device/memory profile, disk reserve, and immutable
  commit. Success creates a new `DownloadedModel` identity and leaves the old entry installed,
  selectable, and unchanged. It never auto-selects or auto-deletes either revision.

## Deletion guarantees

- `preview_delete()` has no side effects and rejects active, leased, non-owned, or already-deleting
  entries or revisions with referenced auxiliaries. It returns a short-lived token bound to model ID,
  repo, commit, dependency closure, deletion-strategy fingerprint, expected bytes, and expiry.
- `delete()` requires the matching unexpired token and rechecks eligibility under the inventory lock.
- The UI-active to request-lease handoff is atomic under the same lock, closing the deletion race.
- Deletion uses the Hub cache revision strategy so blobs shared by other retained revisions survive.
- Only the application's dedicated cache root is mutable.
- Success re-scans, verifies target absence, updates inventory, and reports only measured physical-byte
  delta. A partial failure keeps/demotes the entry with detail instead of claiming success.
- Cancellation or failure leaves the model non-ready only when reconciliation proves files incomplete;
  it never reports reclaimed space that was not measured.

Interrupted `.incomplete` files are not removed by revision deletion. A separate confirmed partial-
discard flow may remove only verified app-owned incomplete targets beneath the dedicated cache; it never
runs automatically or invokes a broad Hub cache prune.

## UI mapping

Model controls map to:

```text
(inventory_rows, role_choices, download_status, download_progress, update_status, disk_summary, safe_error)
```

Deleting requires a confirmation modal/action carrying the short-lived token. Selecting a model marks
it active for UI protection; starting generation replaces selection protection with request leases.
The UI has no license column, card/terms link, acknowledgement control, or license-based warning.
