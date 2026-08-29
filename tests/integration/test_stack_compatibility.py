"""Blocking dependency gate (T007).

The whole architecture rests on one claim: MiniMax-H3 can be loaded with
``trust_remote_code=False``. This module proves or disproves it against real,
released Diffusers/Transformers wheels.

Nothing here downloads weights. Every assertion is about *class availability* in
the installed libraries and about *declared metadata* in the repository, so the
gate runs in seconds on any machine, with no accelerator.

Two loading paths exist in the ``MiniMaxAI/MiniMax-H3`` repository:

* the ``Ref2VA/`` subfolder, whose ``model_index.json`` names classes that no
  upstream Diffusers release exports, and whose VAE component configs carry
  ``auto_map`` entries pointing at bundled ``.py`` modules. This path REQUIRES
  ``trust_remote_code=True`` and is therefore prohibited by the constitution.
* the repository root, described by ``modular_model_index.json``, which names
  only classes that upstream Diffusers and Transformers genuinely export and
  ships no Python alongside its weights. This is the path we use.

``test_ref2va_subfolder_requires_remote_code`` is not a redundant test. It is a
regression guard: if it ever fails, the vendor has made the simpler path viable
and the adapter should be revisited.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import urllib.request

import pytest

pytestmark = pytest.mark.stack_compatibility

# Importing diffusers touches torch.xpu, added in torch 2.4; transformers 5.x
# disables its model classes below torch 2.5. Hosts under that floor cannot run
# the Diffusers half of this gate at all, which is a property of the host and not
# a verdict on the stack — so it skips rather than fails. PyTorch publishes no
# x86_64 macOS wheel above 2.2.2, so Intel Macs always land here.
TORCH_FLOOR = (2, 5)


def _installed(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


# A missing library means the gate cannot render a verdict, which is not the same
# as a negative verdict — so it skips rather than failing red. Unlike the torch
# floor below, this one is fixable: install requirements.txt.
requires_model_stack = pytest.mark.skipif(
    not (_installed("diffusers") and _installed("transformers")),
    reason="model stack not installed; run: pip install -r requirements.txt",
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

REPO_ID = "MiniMaxAI/MiniMax-H3"
_RAW = f"https://huggingface.co/{REPO_ID}/raw/main"

# Components of the root modular pipeline, as declared by modular_model_index.json.
DIFFUSERS_CLASSES = (
    "MiniMaxH3ModularPipeline",
    "MiniMaxH3Blocks",
    "MiniMaxH3Transformer3DModel",
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLMiniMaxH3Audio",
    "MiniMaxH3Scheduler",
)

TRANSFORMERS_CLASSES = (
    "Qwen3VLForConditionalGeneration",
    "Qwen2TokenizerFast",
    "Qwen3VLProcessor",
)


def _fetch_json(path: str) -> dict:
    # URL is a fixed https literal built from a module constant, not user input.
    with urllib.request.urlopen(f"{_RAW}/{path}", timeout=30) as response:  # noqa: S310
        return json.load(response)


@requires_model_stack
@requires_torch_floor
@pytest.mark.parametrize("class_name", DIFFUSERS_CLASSES)
def test_diffusers_exports_h3_class(class_name: str) -> None:
    """Every H3 component class must exist in the installed Diffusers release."""
    diffusers = importlib.import_module("diffusers")
    assert hasattr(diffusers, class_name), (
        f"diffusers {diffusers.__version__} does not export {class_name}. "
        "Pin a release that does; do not enable trust_remote_code."
    )


@requires_model_stack
@pytest.mark.parametrize("class_name", TRANSFORMERS_CLASSES)
def test_transformers_exports_encoder_class(class_name: str) -> None:
    """The text encoder, tokenizer, and processor must all be upstream classes."""
    transformers = importlib.import_module("transformers")
    assert hasattr(transformers, class_name), (
        f"transformers {transformers.__version__} does not export {class_name}. "
        "Pin a release that does; do not enable trust_remote_code."
    )


@requires_model_stack
@requires_torch_floor
def test_modular_index_names_only_installed_classes() -> None:
    """The repository's declared components must all resolve locally.

    This is the assertion that actually gates the architecture: it reads the
    vendor's own component manifest and requires that every class it names is
    importable from an installed library, with no remote code fetched.
    """
    index = _fetch_json("modular_model_index.json")

    unresolved: list[str] = []
    for component, entry in index.items():
        if component.startswith("_") or not isinstance(entry, list):
            continue
        library, class_name = entry[0], entry[1]
        module = importlib.import_module(library)
        if not hasattr(module, class_name):
            unresolved.append(f"{component}: {library}.{class_name}")

    assert not unresolved, (
        f"modular_model_index.json names classes the installed stack cannot provide: {unresolved}"
    )


def test_root_components_declare_no_remote_code() -> None:
    """No root component may carry an ``auto_map``.

    An ``auto_map`` is the mechanism by which a repository asks the loader to
    execute Python it ships. Its presence anywhere on the loading path would
    make ``trust_remote_code=False`` fail, so absence is the property to assert.
    """
    components = (
        "transformer",
        "transformer_ref",
        "vae",
        "audio_vae",
        "text_encoder",
    )

    offenders = [c for c in components if "auto_map" in _fetch_json(f"{c}/config.json")]

    assert not offenders, f"root components declare remote code: {offenders}"


def test_ref2va_subfolder_requires_remote_code() -> None:
    """Documents why ``Ref2VA/`` is not the loading path.

    Inverted on purpose: this passes while the subfolder remains unusable. A
    failure here is good news that warrants revisiting the adapter.
    """
    video_vae = _fetch_json("Ref2VA/video_vae/config.json")
    audio_vae = _fetch_json("Ref2VA/audio_vae/config.json")

    assert "auto_map" in video_vae and "auto_map" in audio_vae, (
        "Ref2VA component configs no longer declare auto_map. The subfolder may "
        "now load natively; re-evaluate which path the adapter should use."
    )
