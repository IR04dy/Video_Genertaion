"""Blocking dependency gate (T007).

The architecture rests on one claim: the model stack loads without executing
code that arrives with the weights. This module proves or disproves it against
real, released packages.

Nothing here downloads weights. Every assertion is about *package contents* and
*declared repository metadata*, so the gate runs in seconds on any machine, with
no accelerator.

Two execution vectors exist on this stack, and each gets its own assertion:

* ``auto_map`` in a component config, which asks the loader to run Python the
  repository ships. Absent from every component here, so ``trust_remote_code``
  is never needed and stays false.
* ``.pth`` checkpoints, which are Python pickles. Wan ships its T5 encoder and
  VAE that way, so unpickling is a real load-path capability that ``auto_map``
  assertions say nothing about. ``torch.load`` defaults to ``weights_only=True``
  from torch 2.6, which closes it; ``test_torch_refuses_arbitrary_unpickling``
  is the guard that this default has not regressed under us.

``test_video_repo_ships_no_loadable_python`` is a regression guard rather than a
redundant test: one inert evaluation script is present today and is not on the
loading path. If that count grows, the repository has changed shape and the
adapter should be re-inspected before the next release is pinned.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import urllib.request

import pytest

pytestmark = pytest.mark.stack_compatibility

# Importing the model stack touches torch.xpu, added in torch 2.4. The pickle
# gate below additionally requires the torch 2.6 weights_only default. Hosts
# under that floor cannot render a verdict, which is a property of the host and
# not a verdict on the stack — so they skip rather than fail. PyTorch publishes
# no x86_64 macOS wheel above 2.2.2, so Intel Macs always land here.
TORCH_FLOOR = (2, 6)


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# A missing library means the gate cannot render a verdict, which is not the same
# as a negative verdict — so it skips rather than failing red. Unlike the torch
# floor below, this one is fixable: install requirements.txt.
requires_model_stack = pytest.mark.skipif(
    not (_installed("diffsynth") and _installed("transformers")),
    reason="model stack not installed; run: pip install -r requirements.txt",
)

requires_voice_stack = pytest.mark.skipif(
    not _installed("chatterbox"),
    reason="voice stack not installed; run: pip install -r requirements.txt",
)


def _torch_version() -> tuple[int, ...]:
    if not _installed("torch"):
        return (0, 0)
    torch = importlib.import_module("torch")
    return tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])


requires_torch_floor = pytest.mark.skipif(
    _torch_version() < TORCH_FLOOR,
    reason=(
        f"needs torch >= {'.'.join(map(str, TORCH_FLOOR))}, found "
        f"{'.'.join(map(str, _torch_version()))}. On the production host this means "
        "the CUDA wheel was not installed before requirements.txt and a CPU wheel "
        "was pulled in its place; reinstall torch first. On an Intel Mac it is a "
        "hard platform ceiling and this gate cannot run here at all."
    ),
)

VIDEO_REPO_ID = "Wan-AI/Wan2.2-S2V-14B"
VOICE_REPO_ID = "ResembleAI/chatterbox"

# Component configs on the video repository's loading path. The T5 encoder and
# VAE ship as bare .pth with no config of their own, so they are covered by the
# unpickling gate instead.
VIDEO_COMPONENT_CONFIGS = (
    "config.json",
    "wav2vec2-large-xlsr-53-english/config.json",
)

# Inert today: an evaluation script bundled with the upstream wav2vec2 release,
# referenced by no config and loaded by nothing. Pinned here so that a repository
# that starts shipping loadable Python fails the gate loudly.
KNOWN_INERT_PYTHON = ("wav2vec2-large-xlsr-53-english/eval.py",)


def _fetch_json(repo_id: str, path: str) -> dict:
    # URL is a fixed https literal built from module constants, not user input.
    url = f"https://huggingface.co/{repo_id}/raw/main/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return json.load(response)


def _list_files(repo_id: str) -> list[str]:
    url = f"https://huggingface.co/api/models/{repo_id}"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        payload = json.load(response)
    return [entry["rfilename"] for entry in payload.get("siblings", [])]


# --------------------------------------------------------------------------
# The packages must provide the pipeline, from a release, not from a fork
# --------------------------------------------------------------------------


@requires_model_stack
@requires_torch_floor
def test_diffsynth_exports_video_pipeline() -> None:
    """The speech-to-video pipeline must come from an installed release.

    Diffusers is deliberately not the loading path here: neither
    ``WanSpeechToVideoPipeline`` nor ``WanS2VTransformer3DModel`` exists in any
    Diffusers release, so that route would require running a fork of the core
    library. DiffSynth ships both from a versioned package instead.
    """
    module = importlib.import_module("diffsynth.pipelines.wan_video")

    for name in ("WanVideoPipeline", "ModelConfig"):
        assert hasattr(module, name), (
            f"diffsynth.pipelines.wan_video does not export {name}. "
            "Pin a release that does; do not vendor a fork."
        )


@requires_model_stack
@requires_torch_floor
def test_diffsynth_ships_the_s2v_denoiser() -> None:
    """S2V has its own DiT; the plain Wan video DiT cannot drive lip movement."""
    assert _installed("diffsynth.models.wan_video_dit_s2v"), (
        "diffsynth does not ship the S2V denoiser. The installed release "
        "supports Wan video but not speech-to-video."
    )


@requires_voice_stack
def test_chatterbox_is_the_tts_package_not_the_irc_stub() -> None:
    """Guard against the dependency-confusion name on PyPI.

    ``chatterbox`` on PyPI is an unrelated 0.0.0 IRC bot framework. The package
    this project needs is ``chatterbox-tts``, which installs the same top-level
    module name. Asserting on module contents rather than on the distribution
    name is what makes this catch a wrong-package install.
    """
    module = importlib.import_module("chatterbox.tts")

    assert hasattr(module, "ChatterboxTTS"), (
        "the installed 'chatterbox' module has no ChatterboxTTS. The IRC bot "
        "stub was almost certainly installed instead; the correct requirement "
        "is 'chatterbox-tts'."
    )


# --------------------------------------------------------------------------
# Neither repository may execute code on the loading path
# --------------------------------------------------------------------------


@pytest.mark.parametrize("config_path", VIDEO_COMPONENT_CONFIGS)
def test_video_components_declare_no_remote_code(config_path: str) -> None:
    """No component may carry an ``auto_map``.

    An ``auto_map`` is the mechanism by which a repository asks the loader to
    execute Python it ships. Its presence anywhere on the loading path would
    make ``trust_remote_code=False`` fail, so absence is the property to assert.
    """
    config = _fetch_json(VIDEO_REPO_ID, config_path)

    assert "auto_map" not in config, (
        f"{VIDEO_REPO_ID}/{config_path} declares remote code via auto_map. "
        "This path is prohibited; do not enable trust_remote_code."
    )


def test_video_repo_ships_no_loadable_python() -> None:
    """Only known-inert Python may accompany the video weights.

    A repository that begins shipping Python is not automatically unsafe, but it
    has changed shape in a way that invalidates the review this pin rests on.
    """
    unexpected = sorted(
        {f for f in _list_files(VIDEO_REPO_ID) if f.endswith(".py")} - set(KNOWN_INERT_PYTHON)
    )

    assert not unexpected, (
        f"{VIDEO_REPO_ID} ships Python not covered by the last review: {unexpected}. "
        "Re-inspect the loading path before pinning a new revision."
    )


def test_voice_repo_ships_only_weights() -> None:
    """The voice repository must carry no Python at all.

    Chatterbox loads through its own installed package, so the repository is
    pure weights and has no legitimate reason to ship code.
    """
    python_files = sorted(f for f in _list_files(VOICE_REPO_ID) if f.endswith(".py"))

    assert not python_files, (
        f"{VOICE_REPO_ID} now ships Python: {python_files}. It is expected to be "
        "weights only; re-inspect before pinning a new revision."
    )


@requires_torch_floor
def test_torch_refuses_arbitrary_unpickling_by_default() -> None:
    """The pickle gate, which ``auto_map`` assertions do not cover.

    Wan ships ``models_t5_umt5-xxl-enc-bf16.pth`` and its VAE as pickles, so
    ``torch.load`` runs against attacker-controllable files on the load path.
    Since torch 2.6 the default is ``weights_only=True``, which permits tensors
    and refuses arbitrary globals. This asserts the default still holds, because
    a regression would silently reopen code execution during model loading.
    """
    import inspect

    import torch

    default = inspect.signature(torch.load).parameters["weights_only"].default

    assert default is not False, (
        f"torch {torch.__version__} defaults torch.load(weights_only={default!r}). "
        "Loading Wan's .pth checkpoints would permit arbitrary code execution; "
        "pass weights_only=True explicitly in the adapter before proceeding."
    )
