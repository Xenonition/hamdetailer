"""
ADetailer Advanced Tab
======================
Adds a top-level "ADetailer+" tab to the WebUI with the following workflow:

1. Upload / send an image
2. Select one or more detection (seg) models
3. Click **Run Detection** → preview with bounding boxes, masks, labels
4. Each detection gets its own InputAccordion with per-region inpainting overrides
5. Click **Process** → standard img2img inpainting per detection

Implementation Notes
--------------------

**UI structure**:
    Two-tab layout inside a top-level ``gr.Blocks``:
    - *Detection tab*  — left: model selector + detect button, right: input Gallery
    - *Inpainting tab* — left: common settings + N InputAccordions, right: Preview/Output sub-tabs
    Auto-switches to Inpainting after detection, to Output sub-tab after processing.

**Flat return lists for Gradio**:
    ``_run_detection()`` returns a flat list whose length is ``2 + 8×MAX_DETECTIONS``:
        [preview, state,
         ×N accordion_updates, ×N enable_updates, ×N mask_previews,
         ×N prompt_resets, ×N neg_prompt_resets,
         ×N denoise_resets, ×N mask_blur_resets, ×N dilate_resets]
    The ``detect_outputs`` list in ``create_advanced_tab`` must match this layout
    exactly.  When adding new per-detection fields, update BOTH places.

    ``_process_all()`` unpacks ``*args`` with the same ×N groups; see its docstring
    for the exact index math.  The ``process_inputs`` list must match.

**Re-detection resets all per-slot fields**:
    Running detection again fully resets prompt, neg prompt, denoising, mask blur,
    and dilate to their defaults for ALL slots—even slots that remain visible.
    This prevents stale values from a previous detection leaking into new results
    (e.g., run face+eye → run mouth: slot 0 must not keep the old face prompt).

**Input Gallery vs hidden Image bridge for send-to buttons**:
    txt2img / img2img send-to buttons use the WebUI's ``parameters_copypaste``
    mechanism (``add_paste_fields`` + ``register_paste_params_button``), which
    requires the destination to be a ``gr.Image`` (since ``image_from_url_text``
    returns a single PIL image, not a list).  A hidden ``gr.Image`` receives the
    image, then its ``.change()`` event wraps it in a list and forwards to the
    input ``gr.Gallery``.  This bridge is necessary because Gallery expects a list.

    The paste-fields are registered under tabname ``"adetailer_plus"``, which
    corresponds to the JS function ``switch_to_adetailer_plus()`` in
    ``javascript/adetailer_plus.js``.  The ``connect_paste_params_buttons()``
    call in the main ``ui.py`` automatically appends a ``.click(_js=...)`` that
    calls that function to switch tabs.

**Button timing (on_after_component vs on_app_started)**:
    The 🔀 ToolButtons in txt2img/img2img output panels are created inside
    ``on_after_component`` (in ``!adetailer.py``) and registered via
    ``register_paste_params_button`` at creation time.  This is critical:
    ``on_app_started`` fires AFTER ``demo.launch()``, at which point Gradio
    routes are finalized and new ``.click()`` registrations have no effect.
    ``register_paste_params_button`` stores the binding, and
    ``connect_paste_params_buttons()`` (called inside ``with gr.Blocks() as demo:``
    in the main ``ui.py``) wires them up at the correct time.

**Tag Autocomplete integration**:
    The a1111-sd-webui-tagcomplete extension discovers prompt textareas via
    CSS selectors + MutationObserver.  An ``"adetailer-plus"`` entry in
    ``_textAreas.js`` targets ``[id^=ad_adv_det_][id$=_prompt]`` and
    ``[id^=ad_adv_det_][id$=_neg]`` inside ``#tab_adetailer_advanced``,
    with ``onDemand: true`` so the observer fires when InputAccordions appear.

**InputAccordion acts as both accordion + checkbox**:
    ``InputAccordion`` yields a component that is stored in ``det_enables``
    (acts as a ``gr.Checkbox``).  Its ``.accordion`` attribute gives the
    ``gr.Accordion`` for visibility / label updates (``det_accordions``).

**gr.State serialisation**:
    Detection state stores masks as ``np.ndarray`` (not PIL) because ``gr.State``
    round-trips through JSON serialization which can lose PIL metadata.  Images
    are converted back via ``Image.fromarray`` in ``_process_all``.

**vec_cc guard**:
    The Vectorscope CC extension patches ``KDiffusionSampler`` with a ``vec_cc``
    attribute only during ``process_batch``.  When ADetailer+ creates its own
    ``StableDiffusionProcessingImg2Img``, the attribute may not exist yet,
    causing ``AttributeError``.  A guard sets a disabled default before each
    ``process_images`` call.

**Script runner isolation**:
    ``_process_all`` uses a shallow copy of ``scripts_img2img`` with
    ``alwayson_scripts = []`` to prevent re-entrant hooks (e.g., the main
    ADetailer script calling itself recursively).
"""

from __future__ import annotations

import platform
from copy import copy
from functools import partial
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich import print  # noqa: A004

from adetailer import ADETAILER, __version__, mediapipe_predict, ultralytics_predict
from adetailer.common import PredictOutput, draw_detection_overlay, ensure_pil_image
from adetailer.mask import dilate_erode
from adetailer.args import INPAINT_BBOX_MATCH_MODES, InpaintBBoxMatchMode
from adetailer.opts import optimal_crop_size

MAX_DETECTIONS = 8

# Module-level reference for cross-tab send-to wiring
ad_plus_input_gallery = None


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _get_ultralytics_device() -> str:
    """Decide which device to run ultralytics on (mirrors main ADetailer)."""
    try:
        from modules.shared import cmd_opts

        if hasattr(cmd_opts, "use_cpu") and "adetailer" in cmd_opts.use_cpu:
            return "cpu"
        if platform.system() == "Darwin":
            return ""
        for flag in ("lowvram", "medvram", "medvram_sdxl"):
            if getattr(cmd_opts, flag, False):
                return "cpu"
    except Exception:
        pass
    return ""


def _make_mask_preview(
    source: Image.Image,
    mask: Image.Image,
    bbox: list,
    label: str,
    conf: float,
) -> Image.Image:
    """Overlay a single detection mask on the source for preview."""
    preview = source.copy()
    overlay = Image.new("RGB", preview.size, (230, 57, 70))
    masked = Image.composite(overlay, preview, mask)
    preview = Image.blend(preview, masked, 0.4)

    draw = ImageDraw.Draw(preview)
    x1, y1, x2, y2 = (int(v) for v in bbox)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    font = ImageFont.load_default()
    draw.text(
        (x1 + 2, max(y1 - 14, 0)),
        f"{label} {conf:.2f}",
        fill="red",
        font=font,
    )
    return preview


# ---------------------------------------------------------------------------
#  Detection callback
# ---------------------------------------------------------------------------

def _run_detection(
    image: Image.Image | None,
    models_selected: list[str] | str | None,
    confidence: float,
    *,
    model_mapping: dict[str, str],
):
    """
    Run selected detection models on *image*.

    Returns a flat output list consumed by Gradio:
        [preview, state,
         *accordion_updates(×N), *enable_updates(×N), *mask_previews(×N),
         *prompt_resets(×N), *neg_prompt_resets(×N),
         *denoise_resets(×N), *mask_blur_resets(×N), *dilate_resets(×N)]
    """
    # Normalise multi-select value
    if isinstance(models_selected, str):
        models_selected = [models_selected]
    if not models_selected:
        models_selected = []

    # Defaults – nothing detected
    empty: list[Any] = (
        [None, None]
        + [gr.update(visible=False)] * MAX_DETECTIONS   # accordion
        + [gr.update(value=True)] * MAX_DETECTIONS       # enable checkbox
        + [None] * MAX_DETECTIONS                        # mask previews
        + [gr.update(value="")] * MAX_DETECTIONS          # prompt resets
        + [gr.update(value="")] * MAX_DETECTIONS          # neg prompt resets
        + [gr.update(value=0.4)] * MAX_DETECTIONS         # denoise resets
        + [gr.update(value=4)] * MAX_DETECTIONS           # mask blur resets
        + [gr.update(value=4)] * MAX_DETECTIONS           # dilate resets
    )

    if image is None:
        return empty

    if not models_selected:
        empty[0] = image
        return empty

    image = ensure_pil_image(image, "RGB")
    device = _get_ultralytics_device()

    all_bboxes: list[list] = []
    all_masks: list[Image.Image] = []
    all_confidences: list[float] = []
    all_labels: list[str] = []

    for model_name in models_selected:
        if model_name not in model_mapping:
            print(f"[-] ADetailer+: model {model_name!r} not in mapping – skipped")
            continue

        model_path = model_mapping[model_name]
        try:
            if model_name.startswith("mediapipe"):
                pred = mediapipe_predict(model_name, image, confidence)
            else:
                from aaaaaa.helper import disable_safe_unpickle

                with disable_safe_unpickle():
                    pred = ultralytics_predict(
                        model_path,
                        image=image,
                        confidence=confidence,
                        device=device,
                    )
        except Exception as e:
            print(f"[-] ADetailer+: Error running {model_name}: {e}")
            continue

        if pred.bboxes:
            all_bboxes.extend(pred.bboxes)
            all_masks.extend(pred.masks)
            all_confidences.extend(pred.confidences)
            all_labels.extend([model_name] * len(pred.bboxes))

    if not all_bboxes:
        empty[0] = image
        return empty

    total_found = len(all_bboxes)
    n = min(total_found, MAX_DETECTIONS)
    all_bboxes = all_bboxes[:n]
    all_masks = all_masks[:n]
    all_confidences = all_confidences[:n]
    all_labels = all_labels[:n]

    # Overall preview with all detections drawn
    preview = draw_detection_overlay(
        image, all_bboxes, all_masks, all_confidences, all_labels
    )

    # State stored as numpy for safe gr.State serialisation
    state = {
        "source": np.array(image),
        "n": n,
        "bboxes": [list(b) for b in all_bboxes],
        "masks": [np.array(m) for m in all_masks],
        "confidences": all_confidences,
        "labels": all_labels,
    }

    # Per-slot UI updates
    accordion_updates: list = []
    enable_updates: list = []
    mask_previews: list = []
    prompt_resets: list = []
    neg_prompt_resets: list = []
    denoise_resets: list = []
    mask_blur_resets: list = []
    dilate_resets: list = []
    for i in range(MAX_DETECTIONS):
        if i < n:
            lbl = f"[{i + 1}] {all_labels[i]} — conf {all_confidences[i]:.2f}"
            accordion_updates.append(
                gr.update(visible=True, label=lbl, open=(i == 0))
            )
            enable_updates.append(gr.update(value=True))
            mask_previews.append(
                _make_mask_preview(
                    image, all_masks[i], all_bboxes[i],
                    all_labels[i], all_confidences[i],
                )
            )
        else:
            accordion_updates.append(gr.update(visible=False))
            enable_updates.append(gr.update(value=True))
            mask_previews.append(None)
        # Always reset per-detection fields to defaults on re-detection
        prompt_resets.append(gr.update(value=""))
        neg_prompt_resets.append(gr.update(value=""))
        denoise_resets.append(gr.update(value=0.4))
        mask_blur_resets.append(gr.update(value=4))
        dilate_resets.append(gr.update(value=4))

    return (
        [preview, state]
        + accordion_updates + enable_updates + mask_previews
        + prompt_resets + neg_prompt_resets
        + denoise_resets + mask_blur_resets + dilate_resets
    )


# ---------------------------------------------------------------------------
#  Processing callback
# ---------------------------------------------------------------------------

def _process_all(state: dict | None, *args):
    """
    Inpaint each enabled detection sequentially through img2img.

    *args* layout (flat)::

        enables       ×MAX   [0N  .. 1N)
        prompts       ×MAX   [1N  .. 2N)
        neg_prompts   ×MAX   [2N  .. 3N)
        denoises      ×MAX   [3N  .. 4N)
        mask_blurs    ×MAX   [4N  .. 5N)
        dilate_erodes ×MAX   [5N  .. 6N)
        --- common ---
        steps, cfg, width, height, sampler, scheduler, padding, bbox_match, styles  [6N .. 6N+8 + styles]
    """
    from modules import paths, scripts as ms, shared
    from modules.processing import StableDiffusionProcessingImg2Img, process_images

    if state is None:
        return None, "⚠  Run detection first."

    N = MAX_DETECTIONS
    enables = args[0 * N : 1 * N]
    prompts = args[1 * N : 2 * N]
    neg_prompts = args[2 * N : 3 * N]
    denoises = args[3 * N : 4 * N]
    mask_blurs = args[4 * N : 5 * N]
    dilate_erodes = args[5 * N : 6 * N]

    c = 6 * N
    steps = int(args[c + 0])
    cfg = float(args[c + 1])
    width = int(args[c + 2])
    height = int(args[c + 3])
    sampler = str(args[c + 4])
    scheduler = str(args[c + 5])
    padding = int(args[c + 6])
    bbox_match = str(args[c + 7]) if len(args) > c + 7 else InpaintBBoxMatchMode.OFF.value
    styles = list(args[c + 8]) if len(args) > c + 8 and args[c + 8] else []

    if shared.sd_model is None:
        return None, "⚠  No Stable Diffusion model loaded."

    source = Image.fromarray(state["source"])
    n_det = state["n"]
    masks = [Image.fromarray(m) for m in state["masks"]]

    working = source.copy()
    done = 0
    outpath = getattr(paths, "data_path", ".")

    for i in range(n_det):
        if not enables[i]:
            continue
        if shared.state.interrupted:
            break

        prompt = str(prompts[i]).strip()
        neg_prompt = str(neg_prompts[i]).strip()
        denoise = float(denoises[i])
        m_blur = int(mask_blurs[i])
        m_dilate = int(dilate_erodes[i])

        mask = masks[i]
        if m_dilate != 0:
            mask = dilate_erode(mask, m_dilate)

        shared.state.textinfo = f"ADetailer+ : detection {i + 1}/{n_det}"
        print(f"[ADetailer+] Det {i+1}/{n_det}: prompt={prompt!r}, denoise={denoise}, blur={m_blur}, dilate={m_dilate}")

        # Per-bbox optimal crop size
        det_width, det_height = width, height
        bbox = state["bboxes"][i]
        if bbox_match == InpaintBBoxMatchMode.STRICT.value:
            if getattr(shared.sd_model, "is_sdxl", False):
                det_width, det_height = optimal_crop_size.sdxl(width, height, bbox)
            else:
                print("[-] ADetailer+: Strict mode is SDXL only, using Free instead.")
                det_width, det_height = optimal_crop_size.free(width, height, bbox)
        elif bbox_match == InpaintBBoxMatchMode.FREE.value:
            det_width, det_height = optimal_crop_size.free(width, height, bbox)
        if (det_width, det_height) != (width, height):
            print(f"[ADetailer+] bbox match {width}x{height} -> {det_width}x{det_height}")
        print(f"[ADetailer+] Inpainting {det_width}x{det_height}, steps={steps}, cfg={cfg}, sampler={sampler}, scheduler={scheduler}, padding={padding}")

        try:
            i2i = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                outpath_samples=outpath,
                outpath_grids=outpath,
                init_images=[working],
                resize_mode=0,
                denoising_strength=denoise,
                mask=mask,
                mask_blur=m_blur,
                inpainting_fill=1,
                inpaint_full_res=True,
                inpaint_full_res_padding=padding,
                inpainting_mask_invert=0,
                prompt=prompt,
                negative_prompt=neg_prompt,
                seed=-1,
                sampler_name=sampler,
                scheduler=scheduler,
                styles=styles,
                batch_size=1,
                n_iter=1,
                steps=steps,
                cfg_scale=cfg,
                width=det_width,
                height=det_height,
                do_not_save_samples=True,
                do_not_save_grid=True,
            )
            i2i.cached_c = [None, None]
            i2i.cached_uc = [None, None]
            i2i._ad_disabled = True
            i2i._ad_inner = True

            # Guard against extensions that monkey-patch the sampler
            # class and lazily set attributes (e.g. Vectorscope CC sets
            # ``vec_cc`` on KDiffusionSampler only inside process_batch,
            # but its patched callback_state always reads it).
            try:
                from modules.sd_samplers_kdiffusion import KDiffusionSampler

                if not hasattr(KDiffusionSampler, "vec_cc"):
                    KDiffusionSampler.vec_cc = {"enable": False}
            except Exception:
                pass

            # Minimal script runner — strip alwayson to prevent re-entrant hooks
            try:
                runner = copy(ms.scripts_img2img)
                runner.alwayson_scripts = []
                i2i.scripts = runner
                max_args = 0
                for s in runner.scripts:
                    if hasattr(s, "args_to"):
                        max_args = max(max_args, s.args_to)
                i2i.script_args = [None] * max_args
            except Exception:
                i2i.scripts = None
                i2i.script_args = []

            processed = process_images(i2i)
            if processed and processed.images:
                working = processed.images[0]
                done += 1
        except Exception as e:
            print(f"[-] ADetailer+: Error on detection {i + 1}: {e}")
            continue
        finally:
            try:
                i2i.close()
            except Exception:
                pass

    status = f"✓  Processed {done} of {n_det} detection(s)."
    if shared.state.interrupted:
        status += " (interrupted)"
    return working, status


# ---------------------------------------------------------------------------
#  Tab builder
# ---------------------------------------------------------------------------

def create_advanced_tab(model_mapping: dict[str, str]) -> gr.Blocks:
    """Build the full ADetailer Advanced tab and return the Blocks container."""
    from modules.sd_samplers import all_samplers
    from modules import sd_schedulers
    from modules.shared import opts
    from modules.ui_components import ToolButton, InputAccordion
    from modules import util, shared
    import modules.infotext_utils as parameters_copypaste

    model_list = list(model_mapping.keys())
    sampler_names = [s.name for s in all_samplers]
    scheduler_names = [x.label for x in sd_schedulers.schedulers]
    style_names = list(shared.prompt_styles.styles.keys())
    gallery_height = getattr(opts, "gallery_height", None) or None

    # Load txt2img defaults from ui-config.json
    import json, os
    _ui_cfg = {}
    _ui_cfg_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "..", "ui-config.json",
    )
    # Normalise — extension lives in  <webui>/extensions/adetailer/
    _ui_cfg_path = os.path.normpath(_ui_cfg_path)
    if not os.path.isfile(_ui_cfg_path):
        # Fallback: use paths module
        try:
            from modules import paths
            _ui_cfg_path = os.path.join(paths.data_path, "ui-config.json")
        except Exception:
            pass
    if os.path.isfile(_ui_cfg_path):
        try:
            with open(_ui_cfg_path, "r", encoding="utf-8") as _f:
                _ui_cfg = json.loads(_f.read())
        except Exception:
            _ui_cfg = {}

    def _uicfg(key, default):
        return _ui_cfg.get(key, default)

    _def_steps = 25
    _def_cfg = 5.0
    _def_width = int(_uicfg("txt2img/Width/value", 512))
    _def_height = int(_uicfg("txt2img/Height/value", 512))
    _def_sampler = "Euler a"
    _def_scheduler = "Automatic"

    def _save_output(image):
        """Save the current output image to the configured save dir."""
        if image is None:
            return gr.update(visible=False), "<p>No image to save.</p>"
        import os, modules.images
        from modules.shared import opts as _opts
        path = _opts.outdir_save
        os.makedirs(path, exist_ok=True)
        fullfn, _ = modules.images.save_image(
            image, path, "adetailer-plus",
            extension=_opts.samples_format,
        )
        return (
            gr.update(value=[fullfn], visible=True),
            f"<p>Saved: {os.path.basename(fullfn)}</p>",
        )

    with gr.Blocks(analytics_enabled=False) as tab:
        detection_state = gr.State(value=None)

        # Hidden Image bridge: receives single PIL from paste-params
        # mechanism, then forwards it to the Gallery via .change()
        _hidden_input = gr.Image(
            visible=False,
            elem_id="ad_adv_hidden_input",
            type="pil",
        )

        with gr.Tabs(elem_id="ad_adv_main_tabs") as main_tabs:
            # ══════════════════════════════════════════════════════════
            #  Detection tab
            # ══════════════════════════════════════════════════════════
            with gr.TabItem("Detection", elem_id="ad_adv_tab_detection", id="detection_tab"):
                with gr.Row(equal_height=True):
                    # ── left: controls ────────────────────────────────
                    with gr.Column(scale=1):
                        model_dropdown = gr.Dropdown(
                            label="Detection Models",
                            choices=model_list,
                            multiselect=True,
                            value=[],
                            elem_id="ad_adv_models",
                            info="Select one or more YOLO / MediaPipe models",
                        )
                        confidence_slider = gr.Slider(
                            label="Confidence Threshold",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.01,
                            value=0.3,
                            elem_id="ad_adv_confidence",
                        )
                        detect_btn = gr.Button(
                            "Run Detection",
                            variant="primary",
                            elem_id="ad_adv_detect_btn",
                        )

                    # ── right: input image ────────────────────────────
                    with gr.Column(scale=1, elem_id="ad_adv_input_col"):
                        with gr.Column(variant="panel"):
                            with gr.Group():
                                input_image = gr.Gallery(
                                    label="Input Image",
                                    show_label=False,
                                    elem_id="ad_adv_input_gallery",
                                    columns=1,
                                    preview=True,
                                    height=gallery_height,
                                    interactive=True,
                                    object_fit="contain",
                                    type="pil",
                                )

            # Expose input_image at module level for cross-tab wiring
            global ad_plus_input_gallery
            ad_plus_input_gallery = input_image

            # Register as paste-params destination so connect_paste_params_buttons
            # can wire the send-to buttons created by on_after_component.
            # The hidden Image receives a single PIL from image_from_url_text,
            # then .change() bridges it to the Gallery.
            parameters_copypaste.add_paste_fields(
                "adetailer_plus", _hidden_input, [], None
            )
            _hidden_input.change(
                fn=lambda img: [img] if img is not None else [],
                inputs=[_hidden_input],
                outputs=[input_image],
                show_progress=False,
            )

            # ══════════════════════════════════════════════════════════
            #  Inpainting tab
            # ══════════════════════════════════════════════════════════
            with gr.TabItem("Inpainting", elem_id="ad_adv_tab_inpainting", id="inpainting_tab"):
                with gr.Row(equal_height=True):
                    # ── left: settings + per-detection accordions ─────
                    with gr.Column(scale=1):
                        with gr.Accordion("Inpainting Settings", open=True):
                            with gr.Row():
                                common_steps = gr.Slider(
                                    label="Steps", minimum=1, maximum=150,
                                    step=1, value=_def_steps, elem_id="ad_adv_steps",
                                )
                                common_cfg = gr.Slider(
                                    label="CFG Scale", minimum=0.0, maximum=30.0,
                                    step=0.5, value=_def_cfg, elem_id="ad_adv_cfg",
                                )
                            with gr.Row():
                                common_width = gr.Slider(
                                    label="Inpaint Width", minimum=64, maximum=2048,
                                    step=8, value=_def_width, elem_id="ad_adv_width",
                                )
                                common_height = gr.Slider(
                                    label="Inpaint Height", minimum=64, maximum=2048,
                                    step=8, value=_def_height, elem_id="ad_adv_height",
                                )
                            with gr.Row():
                                common_sampler = gr.Dropdown(
                                    label="Sampler",
                                    choices=sampler_names,
                                    value=_def_sampler if _def_sampler in sampler_names else (sampler_names[0] if sampler_names else "Euler"),
                                    elem_id="ad_adv_sampler",
                                )
                                common_scheduler = gr.Dropdown(
                                    label="Schedule type",
                                    choices=scheduler_names,
                                    value=_def_scheduler if _def_scheduler in scheduler_names else "Automatic",
                                    elem_id="ad_adv_scheduler",
                                )
                            with gr.Row():
                                common_padding = gr.Slider(
                                    label="Padding (px)", minimum=0, maximum=256,
                                    step=4, value=32, elem_id="ad_adv_padding",
                                )
                                common_bbox_match = gr.Dropdown(
                                    label="Match inpaint size to bbox",
                                    choices=INPAINT_BBOX_MATCH_MODES,
                                    value=InpaintBBoxMatchMode.STRICT.value,
                                    elem_id="ad_adv_bbox_match",
                                )
                                common_styles = gr.Dropdown(
                                    label="Styles",
                                    choices=style_names,
                                    multiselect=True,
                                    value=["Illustrious", "style"],
                                    elem_id="ad_adv_styles",
                                )

                        # Per-detection accordions
                        det_accordions: list[gr.Accordion] = []
                        det_mask_previews: list[gr.Image] = []
                        det_enables = []  # InputAccordion (acts as Checkbox)
                        det_prompts: list[gr.Textbox] = []
                        det_neg_prompts: list[gr.Textbox] = []
                        det_denoises: list[gr.Slider] = []
                        det_mask_blurs: list[gr.Slider] = []
                        det_dilates: list[gr.Slider] = []

                        for i in range(MAX_DETECTIONS):
                            with InputAccordion(
                                value=True,
                                label=f"Detection {i + 1}",
                                visible=False,
                                elem_id=f"ad_adv_det_{i}",
                            ) as en:
                                with gr.Row(equal_height=True):
                                    # ── left column: inputs ──
                                    with gr.Column(scale=1):
                                        pr = gr.Textbox(
                                            label="Prompt",
                                            placeholder="Leave blank for empty prompt",
                                            lines=2,
                                            elem_id=f"ad_adv_det_{i}_prompt",
                                        )
                                        npr = gr.Textbox(
                                            label="Negative Prompt",
                                            placeholder="Leave blank for none",
                                            lines=1,
                                            elem_id=f"ad_adv_det_{i}_neg",
                                        )
                                        with gr.Row():
                                            den = gr.Slider(
                                                label="Denoising", minimum=0.0, maximum=1.0,
                                                step=0.01, value=0.4,
                                                elem_id=f"ad_adv_det_{i}_dn",
                                            )
                                            mbl = gr.Slider(
                                                label="Mask Blur", minimum=0, maximum=64,
                                                step=1, value=4,
                                                elem_id=f"ad_adv_det_{i}_mbl",
                                            )
                                            dil = gr.Slider(
                                                label="Erode(\u2212)/Dilate(+)", minimum=-128,
                                                maximum=128, step=4, value=4,
                                                elem_id=f"ad_adv_det_{i}_dil",
                                            )
                                    # ── right column: mask preview ──
                                    with gr.Column(scale=1):
                                        mprev = gr.Image(
                                            label="Mask",
                                            interactive=False,
                                            height=220,
                                            elem_id=f"ad_adv_det_{i}_mask",
                                        )

                                det_accordions.append(en.accordion)
                                det_mask_previews.append(mprev)
                                det_enables.append(en)
                                det_prompts.append(pr)
                                det_neg_prompts.append(npr)
                                det_denoises.append(den)
                                det_mask_blurs.append(mbl)
                                det_dilates.append(dil)

                        process_btn = gr.Button(
                            "Process All Detections",
                            variant="primary",
                            elem_id="ad_adv_process_btn",
                        )

                    # ── right: preview / output ───────────────────────
                    with gr.Column(scale=1, elem_id="ad_adv_results"):
                        with gr.Tabs(elem_id="ad_adv_result_tabs") as result_tabs:
                            with gr.TabItem("Preview", elem_id="ad_adv_preview_tab", id="preview_tab"):
                                with gr.Column(variant="panel"):
                                    with gr.Group():
                                        preview_image = gr.Gallery(
                                            label="Detection Preview",
                                            show_label=False,
                                            elem_id="ad_adv_preview",
                                            columns=1,
                                            preview=True,
                                            height=gallery_height,
                                            interactive=False,
                                            object_fit="contain",
                                            type="pil",
                                        )

                            with gr.TabItem("Output", elem_id="ad_adv_output_tab", id="output_tab"):
                                with gr.Column(variant="panel", elem_id="ad_adv_output_panel"):
                                    with gr.Group(elem_id="ad_adv_output_gallery_container"):
                                        output_gallery = gr.Gallery(
                                            label="Output",
                                            show_label=False,
                                            elem_id="ad_adv_output_gallery",
                                            columns=1,
                                            preview=True,
                                            height=gallery_height,
                                            interactive=False,
                                            object_fit="contain",
                                            type="pil",
                                        )

                                    with gr.Row(elem_id="ad_adv_image_buttons", elem_classes="image-buttons"):
                                        reuse_btn = ToolButton(
                                            '🔄', elem_id="ad_adv_reuse_output",
                                            tooltip="Send output back to ADetailer+ input",
                                        )
                                        save_btn = ToolButton(
                                            '💾', elem_id="ad_adv_save",
                                            tooltip=f"Save image ({opts.outdir_save})",
                                        )
                                        send_img2img_btn = ToolButton(
                                            '🖼️', elem_id="ad_adv_send_img2img",
                                            tooltip="Send to img2img",
                                        )
                                        send_inpaint_btn = ToolButton(
                                            '🎨️', elem_id="ad_adv_send_inpaint",
                                            tooltip="Send to img2img inpaint",
                                        )
                                        send_extras_btn = ToolButton(
                                            '📐', elem_id="ad_adv_send_extras",
                                            tooltip="Send to extras",
                                        )

                                    download_files = gr.File(
                                        None, file_count="multiple",
                                        interactive=False, show_label=False,
                                        visible=False, elem_id="ad_adv_download",
                                    )
                                    html_log = gr.HTML(
                                        elem_id="ad_adv_html_log",
                                        elem_classes="html-log",
                                    )

        # ── Event wiring ──────────────────────────────────────────────

        # Adapt detection callback: input_image is now a gallery so we
        # extract the first PIL image from the list for the detector.
        def _detect_from_gallery(gallery_images, models_sel, conf):
            img = None
            if gallery_images:
                first = gallery_images[0]
                if isinstance(first, (list, tuple)):
                    img = first[0]
                elif isinstance(first, Image.Image):
                    img = first
                else:
                    img = first
            result = _run_detection(img, models_sel, conf, model_mapping=model_mapping)
            # result[0] is preview PIL; wrap it in a list for gr.Gallery
            if result[0] is not None:
                result[0] = [result[0]]
            return result

        detect_outputs = (
            [preview_image, detection_state]
            + det_accordions
            + det_enables
            + det_mask_previews
            + det_prompts
            + det_neg_prompts
            + det_denoises
            + det_mask_blurs
            + det_dilates
        )
        detect_btn.click(
            fn=lambda: gr.update(value="⏳ Detecting…", interactive=False),
            inputs=None,
            outputs=detect_btn,
        ).then(
            fn=_detect_from_gallery,
            inputs=[input_image, model_dropdown, confidence_slider],
            outputs=detect_outputs,
        ).then(
            fn=lambda: gr.update(value="Run Detection", interactive=True),
            inputs=None,
            outputs=detect_btn,
        ).then(
            fn=lambda: gr.update(selected="inpainting_tab"),
            inputs=None,
            outputs=main_tabs,
        )

        # Adapt process callback: output is now a gallery
        def _process_and_wrap(state_val, *args):
            result_img, _status = _process_all(state_val, *args)
            gallery_val = [result_img] if result_img is not None else []
            return gallery_val

        process_inputs = (
            [detection_state]
            + det_enables
            + det_prompts
            + det_neg_prompts
            + det_denoises
            + det_mask_blurs
            + det_dilates
            + [
                common_steps,
                common_cfg,
                common_width,
                common_height,
                common_sampler,
                common_scheduler,
                common_padding,
                common_bbox_match,
                common_styles,
            ]
        )
        process_btn.click(
            fn=lambda: gr.update(value="⏳ Processing…", interactive=False),
            inputs=None,
            outputs=process_btn,
        ).then(
            fn=_process_and_wrap,
            inputs=process_inputs,
            outputs=[output_gallery],
        ).then(
            fn=lambda: gr.update(value="Process All Detections", interactive=True),
            inputs=None,
            outputs=process_btn,
        ).then(
            fn=lambda: gr.update(selected="output_tab"),
            inputs=None,
            outputs=result_tabs,
        )

        # Save button
        def _save_from_gallery(gallery_images):
            if not gallery_images:
                return gr.update(visible=False), "<p>No image to save.</p>"
            first = gallery_images[0]
            img = first[0] if isinstance(first, (list, tuple)) else first
            return _save_output(img)

        save_btn.click(
            fn=_save_from_gallery,
            inputs=[output_gallery],
            outputs=[download_files, html_log],
        )

        # Reuse button: send output back to input gallery
        def _reuse_output(gallery_images):
            if not gallery_images:
                return gr.update()
            first = gallery_images[0]
            img = first[0] if isinstance(first, (list, tuple)) else first
            return [img]

        reuse_btn.click(
            fn=_reuse_output,
            inputs=[output_gallery],
            outputs=[input_image],
        ).then(
            fn=lambda: gr.update(selected="detection_tab"),
            inputs=None,
            outputs=main_tabs,
        )

        # Send-to buttons: register via parameters_copypaste
        for paste_tabname, paste_button in [
            ("img2img", send_img2img_btn),
            ("inpaint", send_inpaint_btn),
            ("extras", send_extras_btn),
        ]:
            parameters_copypaste.register_paste_params_button(
                parameters_copypaste.ParamBinding(
                    paste_button=paste_button,
                    tabname=paste_tabname,
                    source_tabname=None,
                    source_image_component=output_gallery,
                    paste_field_names=[],
                )
            )

    return tab
