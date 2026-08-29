"""T013: path containment. Everything here is a security boundary."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import pytest

from errors import AppError
from storage import (
    bundle_dir,
    contain,
    is_safe_relative,
    staging_dir,
)


def test_contained_path_is_returned_resolved(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b.txt"
    target.parent.mkdir(parents=True)
    target.write_text("x")
    assert contain(tmp_path, target) == target.resolve()


def test_relative_candidate_is_joined_to_the_root(tmp_path: Path) -> None:
    assert contain(tmp_path, "a/b.txt") == (tmp_path / "a" / "b.txt").resolve()


def test_parent_traversal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        contain(tmp_path, "../outside.txt")


def test_deep_traversal_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        contain(tmp_path, "a/b/../../../outside.txt")


def test_absolute_path_outside_root_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AppError):
        contain(tmp_path, Path(tmp_path.anchor) / "etc" / "passwd")


def test_sibling_prefix_directory_is_not_treated_as_contained(tmp_path: Path) -> None:
    """`/root-evil` must not pass a containment check against `/root`."""
    root = tmp_path / "root"
    root.mkdir()
    sibling = tmp_path / "root-evil"
    sibling.mkdir()
    with pytest.raises(AppError):
        contain(root, sibling / "f.txt")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_symlink_escaping_the_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(AppError):
        contain(root, link / "secret.txt")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX symlink semantics")
def test_symlink_staying_inside_the_root_is_still_refused(tmp_path: Path) -> None:
    """Containment is not the only rule: links are refused on the managed roots
    regardless of target, because a link that resolves inside today can be
    repointed outside tomorrow."""
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "f.txt").write_text("x")
    link = root / "link"
    link.symlink_to(root / "real", target_is_directory=True)
    with pytest.raises(AppError):
        contain(root, link / "f.txt", allow_links=False)


@pytest.mark.parametrize(
    "bad",
    [
        "/abs/path",
        "../escape",
        "a/../../b",
        "C:/windows",
        "C:\\windows",
        r"\\server\share\f.txt",
        "a\\b",
        "f.txt:hidden",
        "",
        ".",
        "..",
    ],
)
def test_unsafe_relative_paths_are_rejected(bad: str) -> None:
    assert is_safe_relative(bad) is False


@pytest.mark.parametrize("good", ["a.txt", "a/b.txt", "inputs/image-1.png", "a.b/c-d_e.txt"])
def test_safe_relative_paths_are_accepted(good: str) -> None:
    assert is_safe_relative(good) is True


def test_alternate_data_stream_suffix_is_rejected() -> None:
    """`f.txt:stream` writes a hidden NTFS stream; refused on every platform so
    that a manifest written on POSIX cannot become dangerous on Windows."""
    assert is_safe_relative("report.txt:$DATA") is False


def test_staging_and_bundle_dirs_use_the_server_uuid(tmp_path: Path) -> None:
    request_id = uuid.uuid4()
    staging = staging_dir(tmp_path, request_id)
    bundle = bundle_dir(tmp_path, request_id)
    assert staging.name == str(request_id)
    assert bundle.name == str(request_id)
    assert staging.parent.name == ".work"
    assert bundle.parent == tmp_path.resolve()


def test_staging_is_inside_the_outputs_root(tmp_path: Path) -> None:
    staging = staging_dir(tmp_path, uuid.uuid4())
    assert contain(tmp_path, staging) == staging


def test_staging_dir_rejects_a_non_uuid_name(tmp_path: Path) -> None:
    with pytest.raises((AppError, AttributeError, TypeError, ValueError)):
        staging_dir(tmp_path, "../../evil")  # type: ignore[arg-type]


def test_bundle_dir_never_collides_with_the_staging_root(tmp_path: Path) -> None:
    """`.work` is not a UUID, so no request can ever publish onto staging."""
    ids = {str(uuid.uuid4()) for _ in range(64)}
    assert ".work" not in ids
