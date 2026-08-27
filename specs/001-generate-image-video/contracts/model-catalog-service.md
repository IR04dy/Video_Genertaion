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

    def preview_delete(self, model_id: str) -> DeletePreview:
        """Return eligibility and expected reclaimed bytes without mutation."""

    def delete(self, model_id: str, confirmation_token: str) -> DeleteResult:
        """Delete an eligible revision after matching explicit confirmation."""

    def refresh(self) -> Sequence[DownloadedModel]:
        """Reconcile inventory against the owned cache and return current entries."""
```

The engine alone uses internal lease operations; they are not exposed to Gradio:

```python
class ModelLeaseManager(Protocol):
    def acquire(self, model_ids: Sequence[str], request_id: str): ...
```

The returned context manager releases every acquired lease on exit.

## Source rules

- Input matches `model-source.schema.json`.
- Scheme is HTTPS and host is exactly `huggingface.co` after normalization.
- Path identifies one model repository, optionally with an approved revision route; blob/resolve/file,
  Spaces, datasets, query strings, fragments, and embedded credentials are rejected.
- A mutable revision is resolved through Hub metadata and persisted as a commit SHA before download.
- Credentials come only from the approved local Hub credential source and are never part of input.

## Inspection guarantees

- Inspection downloads no large weights and never executes repository code.
- `trust_remote_code` and remote kernel trust remain false.
- Repository configuration, tags, filenames, sizes, security scan metadata, license, access state,
  and resolved commit are compared with installed adapter fingerprints.
- The result declares all validated roles/native capabilities, constraints, expected bytes, adapter,
  device/memory compatibility, and safe warnings.
- Unknown adapters, uncovered requested roles, unreviewed executable/pickle artifacts, invalid licenses,
  or target-incompatible memory profiles are not reported as compatible.

## Download guarantees

- Snapshot download targets the resolved commit and the application-owned cache.
- Progress events contain counts/bytes where available but no token or full cache path.
- Interrupted downloads remain non-ready and may be retried using Hub cache resume semantics.
- Ready state is written only after required-file, revision, adapter, size, and allowed-format checks.
- Atomic inventory replacement means restart sees either the prior valid state or the new valid state.
- A ready immutable revision is selectable without network access after restart.

## Inventory guarantees

Each displayed entry includes model ID, repository ID, commit/revision label, adapter/pipeline type,
validated roles/native capabilities, compatibility, license/access state, local size, download state,
last-used time, active/in-use status, and any safe failure detail.

Inventory reconciliation never deletes files automatically. Missing/corrupt snapshots are demoted
from ready with an actionable error.

## Deletion guarantees

- `preview_delete()` has no side effects and rejects active, leased, non-owned, or already-deleting
  entries. It returns a short-lived confirmation token plus expected cache bytes from the Hub strategy.
- `delete()` requires the matching unexpired token and rechecks eligibility under the inventory lock.
- Deletion uses the Hub cache revision strategy so blobs shared by other retained revisions survive.
- Only the application's dedicated cache root is mutable.
- Success removes the inventory entry, measures bytes before/after, and reports actual reclaimed bytes.
- Cancellation or failure leaves the model non-ready only when reconciliation proves files incomplete;
  it never reports reclaimed space that was not measured.

## UI mapping

Model controls map to:

```text
(inventory_rows, role_choices, download_status, download_progress, disk_summary, safe_error)
```

Deleting requires a confirmation modal/action carrying the short-lived token. Selecting a model marks
it active for UI protection; starting generation replaces selection protection with request leases.
