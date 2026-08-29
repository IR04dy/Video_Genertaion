"""Gradio entry point (T045, T046).

This module binds widgets to handlers and owns no policy. Every rule that matters
— consent resetting on any audio change, the timbre-anchor warning, a failure
clearing both the player and the download — lives in `ui_contract.py`, where it
is tested without launching a server.

Gradio is imported lazily so the offline suite never needs a UI framework.

Run with:  python app.py
"""

from __future__ import annotations

import threading
import time

import ui_contract as ui
from config import load_settings
from devices import resolve_device
from logging_config import configure_logging
from model_registry import get_adapter
from pipeline import VideoGenerationEngine

# One generation at a time, process-wide. The model owns the accelerator for the
# duration of a run, so a second concurrent request would contend for memory that
# was budgeted for one.
_GENERATION_LOCK = threading.Lock()


def build_engine(settings, adapter_key: str = "stub") -> VideoGenerationEngine:
    adapter = get_adapter(adapter_key)
    ui.register_profile(adapter.profile)
    return VideoGenerationEngine(
        adapter=adapter,
        outputs_root=settings.outputs_root,
        profile=adapter.profile,
        device=resolve_device(),
        max_reserved_bytes=settings.max_reserved_bytes,
    )


def build_interface(engine: VideoGenerationEngine):
    import gradio as gr

    profile = engine.profile
    supported = profile.duration_range_seconds
    state0 = ui.initial_state(profile)

    with gr.Blocks(title="Image + text to video", analytics_enabled=False) as demo:
        gr.Markdown("## Image + text to video")
        gr.Markdown(
            f"Model profile **{profile.profile_id}** - "
            f"{supported.min_seconds:g}-{supported.max_seconds:g}s at "
            f"{profile.frame_rate:g} fps. "
            "**Inference time is unbounded; a run may take hours.**"
        )

        with gr.Row():
            with gr.Column(scale=1):
                images = gr.File(
                    label=f"Reference images (up to {state0.image_limit})",
                    file_count="multiple",
                    file_types=["image"],
                )
                audio = gr.Audio(label="Reference voice", type="filepath")
                gr.Markdown(f"*{ui.reference_audio_help(profile)}*")

                motion = gr.Textbox(label="Motion prompt", lines=2)
                script = gr.Textbox(label="Speech script", lines=4)
                language = gr.Dropdown(
                    label="Language",
                    choices=ui.language_choices(profile),
                    value=state0.language,
                )
                duration = gr.Slider(
                    label="Duration (seconds) - suggested from your script",
                    minimum=supported.min_seconds,
                    maximum=supported.max_seconds,
                    value=supported.default_seconds,
                    step=0.5,
                )
                consent = gr.Checkbox(
                    label=("I own this recording or have permission to clone this voice."),
                    value=False,
                )
                with gr.Accordion("Advanced", open=False):
                    seed = gr.Number(label="Seed (blank = random)", precision=0)
                    guidance = gr.Number(label="Guidance scale (blank = model default)")

                submit = gr.Button("Generate", variant="primary")

            with gr.Column(scale=1):
                status = gr.Markdown("Ready.")
                memory = gr.Markdown(ui.memory_summary())
                player = gr.Video(label="Result", interactive=False)
                download = gr.File(label="Download", interactive=False)
                errors = gr.Markdown(visible=False)

        # Refresh the memory panel on load, so the figure the operator reads is
        # current rather than whatever it was when the process started.
        demo.load(ui.memory_summary, outputs=memory)

        # Consent resets whenever the recording changes, before anything else.
        audio.change(lambda: False, outputs=consent)

        # The suggestion follows the script, and stays editable.
        def _suggest(text, lang):
            state = ui.on_script_changed(ui.initial_state(profile), text, language=lang)
            return gr.update(value=state.suggested_duration)

        script.change(_suggest, inputs=[script, language], outputs=duration)
        language.change(_suggest, inputs=[script, language], outputs=duration)

        def _run(
            images_v,
            audio_v,
            motion_v,
            script_v,
            language_v,
            duration_v,
            consent_v,
            seed_v,
            guidance_v,
        ):
            progress = gr.Progress()
            started = time.time()

            if not consent_v:
                return (
                    "Ready.",
                    None,
                    None,
                    gr.update(visible=True, value="Confirm voice-cloning consent first."),
                    False,
                )
            if not images_v or not audio_v:
                return (
                    "Ready.",
                    None,
                    None,
                    gr.update(visible=True, value="Add at least one image and a voice recording."),
                    False,
                )

            if not _GENERATION_LOCK.acquire(blocking=False):
                return (
                    "Busy.",
                    None,
                    None,
                    gr.update(visible=True, value="A generation is already running."),
                    False,
                )
            try:

                def on_event(event):
                    progress(
                        event.fraction or 0.0,
                        desc=ui.render_progress(
                            phase=event.phase,
                            fraction=event.fraction,
                            elapsed_seconds=time.time() - started,
                        ),
                    )

                result = engine.run(
                    image_paths=[getattr(f, "name", f) for f in images_v],
                    audio_path=audio_v,
                    motion_prompt=motion_v or "",
                    speech_script=script_v or "",
                    language=language_v,
                    consent_confirmed=bool(consent_v),
                    seed=int(seed_v) if seed_v not in (None, "") else None,
                    requested_seconds=float(duration_v) if duration_v else None,
                    guidance_scale=float(guidance_v) if guidance_v not in (None, "") else None,
                    progress=on_event,
                )
            except Exception as exc:  # noqa: BLE001 - every failure must reach the UI
                from errors import to_detail, translate

                detail = to_detail(translate(exc, stage="validate"))
                return (
                    "Failed.",
                    None,
                    None,
                    gr.update(visible=True, value=f"**{detail.message}**"),
                    False,  # consent always resets after a submit
                )
            finally:
                _GENERATION_LOCK.release()

            outputs = ui.result_outputs(result)
            if outputs.error_message:
                body = f"**{outputs.error_message}**"
                if outputs.suggestions:
                    body += "\n\n" + "\n".join(f"- {s}" for s in outputs.suggestions)
                return "Failed.", None, None, gr.update(visible=True, value=body), False

            return (
                f"Complete in {result.duration_seconds:.0f}s.",
                outputs.video_path,
                outputs.download_path,
                gr.update(visible=False, value=""),
                False,
            )

        submit.click(
            _run,
            inputs=[images, audio, motion, script, language, duration, consent, seed, guidance],
            outputs=[status, player, download, errors, consent],
        ).then(ui.memory_summary, outputs=memory)

    return demo


def main() -> None:
    # Installed before anything else can log. The redaction filter is what keeps
    # prompts, absolute upload paths, and tokens out of the log; implemented but
    # never installed, it protected nothing.
    configure_logging()
    settings = load_settings()
    engine = build_engine(settings)
    build_interface(engine).queue(max_size=4).launch(**ui.launch_kwargs())


if __name__ == "__main__":
    main()
