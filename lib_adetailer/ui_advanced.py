"""
ADetailer Advanced Tab
======================
Adds a top-level "ADetailer+" tab to the WebUI with the following workflow:

1. Upload / send an image
2. Select one or more detection (seg) models
3. Click **Run Detection** → interactive overlay: the image with one clickable
   region per detection (floating ``name conf`` label, hover highlight)
4. **Selection panel** appears; click regions to add them to the current pass
5. User iteratively builds **passes**: click detections → *Add Pass* → repeat
6. Click **Confirm & Continue** → switches to Inpainting tab with one accordion
   per pass (masks within a pass are union-merged)
7. Click **Process** → standard img2img inpainting per pass

Implementation Notes
--------------------

**Pass-based architecture**:
    After detection, accordions are NOT shown immediately.  Instead a selection
    panel lets the user build *passes* — groups of detections whose masks are
    union-merged via ``mask_merge()`` from ``adetailer/mask.py``.  Each pass
    gets one InputAccordion with shared prompt / settings.

    The ``passes_state`` (a ``gr.State``) holds a list of dicts::

        [
            {"label": "Pass 1: [1] face, [3] left eye",
             "det_indices": [0, 2],
             "merged_mask": np.ndarray,
             "merged_bbox": [x1, y1, x2, y2]},
            ...
        ]

    Up to ``MAX_PASSES`` (8) passes are supported.  Each pass maps to one
    InputAccordion slot on the Inpainting tab.

**UI structure**:
    Two-tab layout inside a top-level ``gr.Blocks``:
    - *Detection tab*  — left: model selector + detect button + selection panel,
      right: input Gallery (top), interactive ``region_overlay`` (bottom)
    - *Inpainting tab* — left: common settings + N InputAccordions (one per pass),
      right: Output sub-tab.  Auto-switches to Output after processing.

**Interactive overlay** (``region_overlay``, a ``gr.HTML``):
    ``_render_region_overlay()`` builds an image-map: the base image (cached as a
    base64 ``data:`` URI in ``detection_state``) with one absolutely-positioned
    ``<div class="ad-region">`` per detection (percentage coords ⇒ responsive).
    Each region floats a ``[i] name conf`` label, highlights on CSS ``:hover``,
    and on click invokes the ``adAdvPickRegion(idx)`` JS hook
    (``javascript/adetailer_plus.js``) which writes ``"<idx>|<nonce>"`` into the
    hidden ``region_click`` textbox to fire ``_on_region_click``.

**Selection panel** (on Detection tab, shown after detect):
    - ``selection_state`` — ``gr.State`` of the in-progress selection (detection
      indices for the pass being built); clicked regions toggle in/out and are
      shown highlighted on the overlay
    - *Add Pass* button, *Auto: 1 per detection* shortcut, *Clear Passes* button
    - ``passes_display`` — ``gr.HTML`` showing current pass assignments
    - ``selection_status`` — ``gr.HTML`` readout of the current selection
    - *Confirm & Continue* button

**Flat return lists for Gradio**:
    ``_run_detection()`` returns a flat list consumed by ``detect_outputs``.
    ``_confirm_passes()`` returns a flat list consumed by ``confirm_outputs``.
    ``_process_all()`` unpacks ``*args`` with ×MAX_PASSES groups.
    The ``detect_outputs``, ``confirm_outputs``, and ``process_inputs`` lists
    in ``create_advanced_tab`` must match exactly.

**Hidden Image bridge for send-to buttons**:
    txt2img / img2img send-to buttons use ``parameters_copypaste``; a hidden
    ``gr.Image`` receives a single PIL, then ``.change()`` wraps it as ``[img]``
    and forwards to the Gallery.

**InputAccordion acts as both accordion + checkbox**:
    ``InputAccordion`` yields a component that is stored in ``det_enables``
    (acts as a ``gr.Checkbox``).  Its ``.accordion`` attribute gives the
    ``gr.Accordion`` for visibility / label updates (``det_accordions``).

**gr.State serialisation**:
    Detection state stores masks as ``np.ndarray`` (not PIL) because ``gr.State``
    round-trips through JSON serialization which can lose PIL metadata.

**vec_cc guard**:
    The Vectorscope CC extension patches ``KDiffusionSampler`` with a ``vec_cc``
    attribute only during ``process_batch``.  A guard sets a disabled default
    before each ``process_images`` call.

**Script runner isolation**:
    ``_process_all`` uses a shallow copy of ``scripts_img2img`` with
    ``alwayson_scripts = []`` to prevent re-entrant hooks.

**Tag Autocomplete integration**:
    elem_id convention ``ad_adv_det_{i}_prompt`` / ``ad_adv_det_{i}_neg`` is
    preserved for the ``a1111-sd-webui-tagcomplete`` CSS selectors.
"""

from __future__ import annotations

import base64
import io
import platform
from copy import copy
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich import print  # noqa: A004

from lib_adetailer import __version__, mediapipe_predict, ultralytics_predict, style_picker_hook
from lib_adetailer.args import INPAINT_BBOX_MATCH_MODES, InpaintBBoxMatchMode
from lib_adetailer.detection.common import PredictOutput, color_for_label
from lib_adetailer.mask import dilate_erode, mask_merge
from lib_adetailer.opts import OptimalCropSize
from lib_adetailer.utils import ensure_pil_image

MAX_DETECTIONS = 8
MAX_PASSES = 8

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


def _make_merged_mask_preview(
    source: Image.Image,
    merged_mask: Image.Image,
    merged_bbox: list,
    label: str,
) -> Image.Image:
    """Overlay a merged pass mask on the source for preview."""
    preview = source.copy()
    overlay = Image.new("RGB", preview.size, (57, 150, 230))
    masked = Image.composite(overlay, preview, merged_mask)
    preview = Image.blend(preview, masked, 0.4)

    draw = ImageDraw.Draw(preview)
    x1, y1, x2, y2 = (int(v) for v in merged_bbox)
    draw.rectangle([x1, y1, x2, y2], outline="#3996E6", width=2)
    font = ImageFont.load_default()
    draw.text(
        (x1 + 2, max(y1 - 14, 0)),
        label,
        fill="#3996E6",
        font=font,
    )
    return preview


def _merge_bboxes(bboxes: list[list]) -> list:
    """Compute the bounding box of the union of multiple bboxes."""
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
#  Interactive region overlay (click-to-select)
# ---------------------------------------------------------------------------

_REGION_STYLE = """<style>
.ad-adv-stage { position: relative; display: inline-block; max-width: 100%; max-height: 60vh; line-height: 0; user-select: none; }
.ad-adv-stage > img { display: block; max-width: 100%; max-height: 60vh; width: auto; height: auto; -webkit-user-drag: none; }
.ad-region {
    position: absolute; box-sizing: border-box;
    border: 2px solid var(--rc, #e63946); border-radius: 3px;
    cursor: pointer; transition: background-color .1s ease, box-shadow .1s ease;
}
.ad-region:hover { background-color: var(--rcf, rgba(230,57,70,.32)); }
.ad-region.selected {
    background-color: var(--rcf, rgba(46,204,113,.32));
    border-color: #2ecc71; border-width: 3px; box-shadow: 0 0 0 1px #2ecc71;
}
.ad-region.assigned {
    border-style: dashed; border-color: #999;
    background-color: rgba(130,130,130,.28); cursor: default; pointer-events: none;
}
.ad-region-label {
    position: absolute; left: -2px; top: -1.45em; white-space: nowrap;
    font-size: 12px; line-height: 1.35; padding: 0 5px; border-radius: 3px;
    color: #fff; background: var(--rc, #e63946); font-weight: 600;
    pointer-events: none; box-shadow: 0 1px 2px rgba(0,0,0,.4);
}
.ad-region-label.label-inside { top: 2px; left: 2px; }
.ad-region.selected .ad-region-label { background: #2ecc71; }
.ad-region.assigned .ad-region-label { background: #888; }
.ad-seg {
    position: absolute; inset: 0;
    background-color: var(--rc, #e63946); opacity: .30;
    -webkit-mask-size: 100% 100%; mask-size: 100% 100%;
    -webkit-mask-repeat: no-repeat; mask-repeat: no-repeat;
    -webkit-mask-position: center center; mask-position: center center;
    pointer-events: none;
}
.ad-seg.selected { background-color: #2ecc71; opacity: .45; }
.ad-seg.assigned { background-color: #888; opacity: .40; }
</style>"""


def _esc(text) -> str:
    """Minimal HTML-escape for label text injected into the overlay."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _bbox_area(bbox: list) -> float:
    return max(0.0, (bbox[2] - bbox[0])) * max(0.0, (bbox[3] - bbox[1]))


def _img_to_data_uri(arr) -> str:
    """Encode a numpy array / PIL image to a base64 PNG ``data:`` URI."""
    img = arr if isinstance(arr, Image.Image) else Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _mask_to_data_uri(arr) -> str:
    """Encode a single-channel mask (numpy array or PIL ``L``) as an RGBA PNG
    ``data:`` URI whose **alpha channel is the mask**. Used as a CSS
    ``mask-image`` so a colored div is painted only over the segmented area.
    Encoded once at detection time and cached in ``detection_state`` so per-click
    overlay re-renders never re-encode it."""
    if isinstance(arr, Image.Image):
        a = arr.convert("L")
    else:
        m = arr
        if getattr(m, "ndim", 2) == 3:
            m = m[..., 0]
        a = Image.fromarray(m.astype("uint8"), mode="L")
    white = Image.new("L", a.size, 255)
    rgba = Image.merge("RGBA", (white, white, white, a))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _no_detection_notice(message: str = "No detections found.") -> str:
    """A simple notice shown in place of the region overlay when detection
    yields nothing — keeps the input gallery as the only image (no redundant
    bottom canvas)."""
    return (
        "<div style='padding:14px 16px;border:1px dashed var(--border-color-primary,#888);"
        "border-radius:6px;text-align:center;color:var(--body-text-color-subdued,#aaa);"
        "font-size:14px;'>"
        f"{_esc(message)}"
        "<div style='margin-top:4px;font-size:12px;opacity:.8;'>"
        "Try lowering the confidence threshold or selecting a different model.</div>"
        "</div>"
    )


def _render_region_overlay(
    detection_state: dict | None,
    selected_indices: list | None,
    passes: list | None,
) -> str:
    """Build the interactive HTML overlay: base image + one clickable div per
    detection. Regions already assigned to a pass are locked; the current
    selection is highlighted. Smaller boxes are emitted last so they sit on top
    of larger overlapping ones and stay clickable."""
    if not detection_state:
        return ""

    data_uri = detection_state.get("data_uri") or _img_to_data_uri(
        detection_state["source"]
    )
    W = detection_state.get("width") or detection_state["source"].shape[1]
    H = detection_state.get("height") or detection_state["source"].shape[0]
    n = detection_state["n"]
    bboxes = detection_state["bboxes"]
    labels = detection_state["labels"]
    confidences = detection_state["confidences"]
    mask_uris = detection_state.get("mask_uris") or []
    selected = set(selected_indices or [])

    assigned: dict[int, int] = {}
    for pass_num, p in enumerate(passes or [], start=1):
        for idx in p["det_indices"]:
            assigned[idx] = pass_num

    order = sorted(range(n), key=lambda i: _bbox_area(bboxes[i]), reverse=True)

    def _state_of(i: int) -> str:
        if i in assigned:
            return "assigned"
        if i in selected:
            return "selected"
        return ""

    parts = [
        _REGION_STYLE,
        f'<div class="ad-adv-stage"><img src="{data_uri}" draggable="false"/>',
    ]

    # Segment fills first (largest first) so they sit *under* the clickable bbox
    # divs and never intercept clicks (they are pointer-events:none anyway).
    for i in order:
        if i >= len(mask_uris) or not mask_uris[i]:
            continue
        r, g, b = color_for_label(labels[i])
        seg_style = (
            f"--rc:rgb({r},{g},{b});"
            f"-webkit-mask-image:url({mask_uris[i]});mask-image:url({mask_uris[i]});"
        )
        parts.append(
            f'<div class="ad-seg {_state_of(i)}" data-idx="{i}" style="{seg_style}"></div>'
        )

    for i in order:
        x1, y1, x2, y2 = (float(v) for v in bboxes[i])
        left = max(0.0, x1) / W * 100.0
        top = max(0.0, y1) / H * 100.0
        w = (x2 - x1) / W * 100.0
        h = (y2 - y1) / H * 100.0
        r, g, b = color_for_label(labels[i])
        state = _state_of(i)

        if state == "assigned":
            text = f"Pass {assigned[i]}: [{i + 1}] {labels[i]}"
            onclick = ""
        elif state == "selected":
            text = f"✓ [{i + 1}] {labels[i]} {confidences[i]:.2f}"
            onclick = f' onclick="adAdvPickRegion({i})"'
        else:
            text = f"[{i + 1}] {labels[i]} {confidences[i]:.2f}"
            onclick = f' onclick="adAdvPickRegion({i})"'

        label_inside = " label-inside" if top < 7.0 else ""
        style = (
            f"left:{left:.3f}%;top:{top:.3f}%;width:{w:.3f}%;height:{h:.3f}%;"
            f"--rc:rgb({r},{g},{b});--rcf:rgba({r},{g},{b},0.32);"
        )
        parts.append(
            f'<div class="ad-region {state}" data-idx="{i}" style="{style}"{onclick}>'
            f'<span class="ad-region-label{label_inside}">{_esc(text)}</span></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _render_selection_status(selected_indices: list | None, detection_state: dict | None) -> str:
    """Textual readout of the in-progress selection (the pass being built)."""
    if not selected_indices or not detection_state:
        return ""
    labels = detection_state["labels"]
    parts = [f"[{i + 1}] {labels[i]}" for i in sorted(selected_indices)]
    return f"<p><b>Current selection:</b> {', '.join(parts)}</p>"


def _render_passes_html(passes: list) -> str:
    """Render the current passes list as HTML for display."""
    if not passes:
        return "<p style='color:gray; font-style:italic'>No passes defined yet.</p>"

    lines = []
    for p in passes:
        lines.append(f"<b>{p['label']}</b>")
    return "<br>".join(lines)


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

    Returns a flat output list consumed by Gradio::

        [region_overlay_html, detection_state,
         selection_panel_visible, input_gallery_visible,
         selection_state_reset, selection_status_reset,
         passes_state_reset, passes_html_reset,
         *accordion_hidden(xN), *enable_reset(xN), *mask_preview_reset(xN),
         *prompt_reset(xN), *neg_prompt_reset(xN),
         *denoise_reset(xN), *blur_reset(xN), *dilate_reset(xN)]

    Total length: 8 + 8*MAX_PASSES
    """
    if isinstance(models_selected, str):
        models_selected = [models_selected]
    if not models_selected:
        models_selected = []

    N = MAX_PASSES
    # Defaults - nothing detected; hide selection panel and all accordions
    empty: list[Any] = (
        ["", None]                                        # region_overlay, state
        + [gr.update(visible=False)]                      # selection panel
        + [gr.update(visible=True)]                       # input gallery stays visible
        + [[]]                                            # selection_state reset
        + [""]                                            # selection_status reset
        + [[]]                                            # passes_state reset
        + [""]                                            # passes_html reset
        + [gr.update(visible=False)] * N                  # accordions
        + [gr.update(value=True)] * N                     # enables
        + [None] * N                                      # mask previews
        + [gr.update(value="")] * N                       # prompts
        + [gr.update(value="")] * N                       # neg prompts
        + [gr.update(value=0.4)] * N                      # denoise
        + [gr.update(value=4)] * N                        # mask blur
        + [gr.update(value=4)] * N                        # dilate
    )

    if image is None:
        return empty

    # Everything below can touch external model code / image IO that may raise.
    # On any uncaught failure, fall back to the ``empty`` payload (with a bare
    # overlay if we got far enough) so the detect button's trailing ``.then()``
    # always fires and re-enables it — otherwise it stays stuck on "Detecting…".
    bare_overlay = ""
    try:
        image = ensure_pil_image(image, "RGB")
        bare_overlay = (
            f'{_REGION_STYLE}<div class="ad-adv-stage">'
            f'<img src="{_img_to_data_uri(image)}" draggable="false"/></div>'
        )

        if not models_selected:
            empty[0] = bare_overlay
            return empty

        device = _get_ultralytics_device()

        all_bboxes: list[list] = []
        all_masks: list[Image.Image] = []
        all_confidences: list[float] = []
        all_labels: list[str] = []

        for model_name in models_selected:
            if model_name not in model_mapping:
                print(f"[-] ADetailer+: model {model_name!r} not in mapping - skipped")
                continue

            model_path = model_mapping[model_name]
            try:
                # Neo routes mediapipe by "name does not end in .pt"; both
                # mediapipe_predict and ultralytics_predict take the model path.
                if not str(model_name).lower().endswith(".pt"):
                    pred = mediapipe_predict(model_path, image, confidence)
                else:
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
            # Nothing detected: show only the input gallery + a notice, not a
            # redundant bottom canvas of the same image with no boxes.
            empty[0] = _no_detection_notice()
            return empty

        total_found = len(all_bboxes)
        n = min(total_found, MAX_DETECTIONS)
        all_bboxes = all_bboxes[:n]
        all_masks = all_masks[:n]
        all_confidences = all_confidences[:n]
        all_labels = all_labels[:n]

        # Surface the cap instead of silently dropping detections > MAX_DETECTIONS.
        trunc_note = ""
        if total_found > n:
            print(f"[ADetailer+] {total_found} detections found; showing first {n} (max {MAX_DETECTIONS}).")
            trunc_note = (
                f"<p style='color:orange'>⚠ {total_found} detections found; "
                f"showing first {n} (max {MAX_DETECTIONS}).</p>"
            )

        # State stored as numpy for safe gr.State serialisation.  ``data_uri`` and
        # ``width``/``height`` are cached so per-click overlay re-renders never need
        # to re-encode the source image.
        state = {
            "source": np.array(image),
            "width": image.width,
            "height": image.height,
            "data_uri": _img_to_data_uri(image),
            "n": n,
            "bboxes": [list(b) for b in all_bboxes],
            "masks": [np.array(m) for m in all_masks],
            # Per-detection segment overlays (alpha PNG) encoded once; used as a
            # CSS mask-image to highlight the actual segmented area.
            "mask_uris": [_mask_to_data_uri(m) for m in all_masks],
            "confidences": all_confidences,
            "labels": all_labels,
        }

        overlay_html = _render_region_overlay(state, [], [])
    except Exception as e:
        print(f"[-] ADetailer+: detection failed: {e}")
        if bare_overlay:
            empty[0] = bare_overlay
        # surface the error in the (still hidden) selection_status slot is moot;
        # leave the gallery visible so the user can retry.
        return empty

    N = MAX_PASSES
    return (
        [overlay_html, state]                             # region_overlay, state
        + [gr.update(visible=True)]                       # selection panel visible
        + [gr.update(visible=False)]                      # hide input gallery
        + [[]]                                            # selection_state reset
        + [trunc_note]                                    # selection_status (cap note)
        + [[]]                                            # passes_state reset
        + [""]                                            # passes_html reset
        + [gr.update(visible=False)] * N                  # hide all accordions
        + [gr.update(value=True)] * N                     # enable resets
        + [None] * N                                      # mask preview resets
        + [gr.update(value="")] * N                       # prompt resets
        + [gr.update(value="")] * N                       # neg prompt resets
        + [gr.update(value=0.4)] * N                      # denoise resets
        + [gr.update(value=4)] * N                        # mask blur resets
        + [gr.update(value=4)] * N                        # dilate resets
    )


# ---------------------------------------------------------------------------
#  Pass-building callbacks
# ---------------------------------------------------------------------------

# All pass-building / selection callbacks share this output convention::
#
#     [passes_state, selection_state, region_overlay_html,
#      passes_display_html, selection_status_html]


def _on_region_click(
    raw: str,
    selected: list | None,
    detection_state: dict | None,
    passes: list | None,
):
    """
    Toggle the clicked detection in/out of the in-progress selection.

    ``raw`` is ``"<idx>|<nonce>"`` written by the ``adAdvPickRegion`` JS hook;
    the nonce guarantees Gradio fires ``.change`` even on repeated clicks of the
    same region.

    Returns: [selection_state, region_overlay_html, selection_status_html]
    """
    selected = list(selected or [])
    if detection_state is None or not raw:
        return selected, _render_region_overlay(detection_state, selected, passes), \
            _render_selection_status(selected, detection_state)

    try:
        idx = int(str(raw).split("|", 1)[0])
    except (ValueError, TypeError):
        idx = -1

    used = set()
    for p in (passes or []):
        used.update(p["det_indices"])

    if 0 <= idx < detection_state["n"] and idx not in used:
        if idx in selected:
            selected.remove(idx)
        else:
            selected.append(idx)

    return (
        selected,
        _render_region_overlay(detection_state, selected, passes),
        _render_selection_status(selected, detection_state),
    )


def _add_pass(passes: list, selected: list | None, detection_state: dict | None):
    """
    Group the currently selected detections into a new inpaint pass.

    Returns: [passes_state, selection_state, region_overlay_html,
              passes_display_html, selection_status_html]
    """
    selected = list(selected or [])

    def _result(passes_out, selection_out, status):
        return (
            passes_out,
            selection_out,
            _render_region_overlay(detection_state, selection_out, passes_out),
            _render_passes_html(passes_out),
            status,
        )

    if detection_state is None:
        return _result(passes, selected, "")

    if not selected:
        return _result(
            passes, selected,
            "<p style='color:orange'>\u26a0 Click at least one detection first.</p>",
        )

    if len(passes) >= MAX_PASSES:
        return _result(
            passes, selected,
            f"<p style='color:orange'>\u26a0 Maximum {MAX_PASSES} passes reached.</p>",
        )

    n = detection_state["n"]
    labels = detection_state["labels"]

    # Drop any already-assigned / out-of-range indices defensively
    used = set()
    for p in passes:
        used.update(p["det_indices"])
    det_indices = [i for i in selected if 0 <= i < n and i not in used]
    if not det_indices:
        return _result(
            passes, [],
            "<p style='color:orange'>\u26a0 No valid detections selected.</p>",
        )

    pass_num = len(passes) + 1
    parts = [f"[{idx + 1}] {labels[idx]}" for idx in sorted(det_indices)]
    pass_label = f"Pass {pass_num}: {', '.join(parts)}"

    # Merge masks (union) and bboxes
    masks_pil = [Image.fromarray(detection_state["masks"][idx]) for idx in det_indices]
    if len(masks_pil) > 1:
        merged_masks = mask_merge(masks_pil)
        merged_mask_np = np.array(merged_masks[0])
    else:
        merged_mask_np = detection_state["masks"][det_indices[0]]

    bboxes = [detection_state["bboxes"][idx] for idx in det_indices]
    merged_bbox = _merge_bboxes(bboxes)

    new_pass = {
        "label": pass_label,
        "det_indices": det_indices,
        "merged_mask": merged_mask_np,
        "merged_bbox": merged_bbox,
    }

    passes = list(passes) + [new_pass]
    return _result(passes, [], "")


def _auto_one_per_detection(detection_state: dict | None, existing_passes: list):
    """
    Shortcut: create one pass per unassigned detection.

    Returns: [passes_state, selection_state, region_overlay_html,
              passes_display_html, selection_status_html]
    """
    def _result(passes_out):
        return (
            passes_out,
            [],
            _render_region_overlay(detection_state, [], passes_out),
            _render_passes_html(passes_out),
            "",
        )

    if detection_state is None:
        return _result(list(existing_passes))

    n = detection_state["n"]
    if n == 0:
        return _result(list(existing_passes))

    labels = detection_state["labels"]

    used = set()
    for p in existing_passes:
        used.update(p["det_indices"])

    passes = list(existing_passes)
    for i in range(n):
        if i in used:
            continue
        if len(passes) >= MAX_PASSES:
            break

        pass_num = len(passes) + 1
        pass_label = f"Pass {pass_num}: [{i + 1}] {labels[i]}"
        passes.append({
            "label": pass_label,
            "det_indices": [i],
            "merged_mask": detection_state["masks"][i],
            "merged_bbox": detection_state["bboxes"][i],
        })

    return _result(passes)


def _clear_passes(detection_state: dict | None):
    """
    Clear all passes and the current selection.

    Returns: [passes_state, selection_state, region_overlay_html,
              passes_display_html, selection_status_html, input_gallery_visible]

    Unlike _add_pass / _auto_one_per_detection (which keep the upload gallery
    hidden), clearing passes re-shows it so the user can swap the source image.
    """
    return (
        [], [], _render_region_overlay(detection_state, [], []), "", "",
        gr.update(visible=True),
    )


# ---------------------------------------------------------------------------
#  Confirm passes -> populate accordions
# ---------------------------------------------------------------------------

def _confirm_passes(detection_state: dict | None, passes: list):
    """
    Finalize pass selection and populate Inpainting tab accordions.

    Returns a flat list::

        [*accordion_updates(xN), *enable_updates(xN),
         *mask_previews(xN),
         *prompt_resets(xN), *neg_prompt_resets(xN),
         *denoise_resets(xN), *blur_resets(xN),
         *dilate_resets(xN)]

    Total length: 8 * MAX_PASSES
    """
    N = MAX_PASSES
    empty = (
        [gr.update(visible=False)] * N   # accordions
        + [gr.update(value=True)] * N    # enables
        + [None] * N                     # mask previews
        + [gr.update(value="")] * N      # prompts
        + [gr.update(value="")] * N      # neg prompts
        + [gr.update(value=0.4)] * N     # denoise
        + [gr.update(value=4)] * N       # blur
        + [gr.update(value=4)] * N       # dilate
    )

    if detection_state is None or not passes:
        return empty

    source = Image.fromarray(detection_state["source"])
    n_passes = min(len(passes), N)

    accordion_updates = []
    enable_updates = []
    mask_previews = []
    prompt_resets = []
    neg_prompt_resets = []
    denoise_resets = []
    mask_blur_resets = []
    dilate_resets = []

    for i in range(N):
        if i < n_passes:
            p = passes[i]
            accordion_updates.append(
                gr.update(visible=True, label=p["label"], open=(i == 0))
            )
            enable_updates.append(gr.update(value=True))

            merged_mask = Image.fromarray(p["merged_mask"])
            preview = _make_merged_mask_preview(
                source, merged_mask, p["merged_bbox"], p["label"]
            )
            mask_previews.append(preview)
        else:
            accordion_updates.append(gr.update(visible=False))
            enable_updates.append(gr.update(value=True))
            mask_previews.append(None)

        prompt_resets.append(gr.update(value=""))
        neg_prompt_resets.append(gr.update(value=""))
        denoise_resets.append(gr.update(value=0.4))
        mask_blur_resets.append(gr.update(value=4))
        dilate_resets.append(gr.update(value=4))

    return (
        accordion_updates + enable_updates + mask_previews
        + prompt_resets + neg_prompt_resets
        + denoise_resets + mask_blur_resets + dilate_resets
    )


# ---------------------------------------------------------------------------
#  Processing callback
# ---------------------------------------------------------------------------

def _process_all(detection_state: dict | None, passes_state: list | None, *args):
    """
    Inpaint each enabled pass sequentially through img2img.

    Passes use union-merged masks.  *args* layout (flat)::

        enables       xMAX_PASSES  [0N  .. 1N)
        prompts       xMAX_PASSES  [1N  .. 2N)
        neg_prompts   xMAX_PASSES  [2N  .. 3N)
        denoises      xMAX_PASSES  [3N  .. 4N)
        mask_blurs    xMAX_PASSES  [4N  .. 5N)
        dilate_erodes xMAX_PASSES  [5N  .. 6N)
        --- common ---
        steps, cfg, width, height, sampler, scheduler, padding, bbox_match, styles
        --- Style Picker slots (empty when that extension is not installed) ---
    """
    from modules import paths, scripts as ms, shared, images
    from modules.processing import StableDiffusionProcessingImg2Img, process_images
    from modules.shared import opts

    if detection_state is None or not passes_state:
        return None, "\u26a0  Run detection and confirm passes first."

    N = MAX_PASSES
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
    style_picker_slots = args[c + 9 :]

    if shared.sd_model is None:
        return None, "\u26a0  No Stable Diffusion model loaded."

    source = Image.fromarray(detection_state["source"])
    n_passes = min(len(passes_state), N)

    working = source.copy()
    done = 0
    failures: list[str] = []
    # Save into img2img samples folder (respecting outdir_samples override)
    outpath = (
        getattr(opts, "outdir_samples", "")
        or getattr(opts, "outdir_img2img_samples", "")
        or getattr(paths, "data_path", ".")
    )
    grids_outpath = (
        getattr(opts, "outdir_grids", "")
        or getattr(opts, "outdir_img2img_grids", "")
        or outpath
    )

    for i in range(n_passes):
        if not enables[i]:
            continue
        if shared.state.interrupted:
            break

        p = passes_state[i]
        prompt = str(prompts[i]).strip()
        neg_prompt = str(neg_prompts[i]).strip()
        prompt, neg_prompt, style_picker_info = style_picker_hook.apply(prompt, neg_prompt, style_picker_slots)
        denoise = float(denoises[i])
        m_blur = int(mask_blurs[i])
        m_dilate = int(dilate_erodes[i])

        mask = Image.fromarray(p["merged_mask"])
        if m_dilate != 0:
            mask = dilate_erode(mask, m_dilate)

        shared.state.textinfo = f"ADetailer+ : pass {i + 1}/{n_passes}"
        print(f"[ADetailer+] Pass {i+1}/{n_passes}: prompt={prompt!r}, denoise={denoise}, blur={m_blur}, dilate={m_dilate}")

        # Per-pass optimal crop size using merged bbox
        det_width, det_height = width, height
        bbox = p["merged_bbox"]
        if bbox_match == InpaintBBoxMatchMode.STRICT.value:
            det_width, det_height = OptimalCropSize.strict(bbox)
        elif bbox_match == InpaintBBoxMatchMode.FREE.value:
            det_width, det_height = OptimalCropSize.free(width, height, bbox)
        if (det_width, det_height) != (width, height):
            print(f"[ADetailer+] bbox match {width}x{height} -> {det_width}x{det_height}")
        print(f"[ADetailer+] Inpainting {det_width}x{det_height}, steps={steps}, cfg={cfg}, sampler={sampler}, scheduler={scheduler}, padding={padding}")

        try:
            i2i = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                outpath_samples=outpath,
                outpath_grids=grids_outpath,
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
            if style_picker_info:
                i2i.extra_generation_params["Style Picker"] = style_picker_info

            # Guard against Vectorscope CC extension
            try:
                from modules.sd_samplers_kdiffusion import KDiffusionSampler

                if not hasattr(KDiffusionSampler, "vec_cc"):
                    KDiffusionSampler.vec_cc = {"enable": False}
            except Exception:
                pass

            # Minimal script runner - strip alwayson to prevent re-entrant hooks
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
            print(f"[-] ADetailer+: Error on pass {i + 1}: {e}")
            failures.append(f"pass {i + 1}: {e}")
            continue
        finally:
            try:
                i2i.close()
            except Exception:
                pass

    status = f"\u2713  Processed {done} of {n_passes} pass(es)."
    if shared.state.interrupted:
        status += " (interrupted)"
    if failures:
        status += " \u26a0 Failed: " + "; ".join(failures)

    # Save final composited image into img2img output folder
    if done > 0 and working is not None:
        try:
            info = (
                "ADetailer+ "
                f"v{__version__}, passes: {done}/{n_passes}, "
                f"size: {width}x{height}, sampler: {sampler}, "
                f"scheduler: {scheduler}, steps: {steps}, cfg: {cfg}, "
                f"bbox_match: {bbox_match}"
            )
            saved_path, _ = images.save_image(
                image=working,
                path=outpath,
                basename="",
                seed=-1,
                prompt="",
                extension=opts.samples_format,
                info=info,
                p=None,
                suffix="-ad-plus",
            )
            print(f"[ADetailer+] Saved: {saved_path}")
        except Exception as e:
            print(f"[-] ADetailer+: Failed to save output image: {e}")

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
    _ui_cfg_path = os.path.normpath(_ui_cfg_path)
    if not os.path.isfile(_ui_cfg_path):
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
        passes_state = gr.State(value=[])
        selection_state = gr.State(value=[])  # in-progress selection (the pass being built)

        # Hidden Image bridge for paste-params
        _hidden_input = gr.Image(
            visible=False,
            elem_id="ad_adv_hidden_input",
            type="pil",
        )

        with gr.Tabs(elem_id="ad_adv_main_tabs") as main_tabs:
            # ==============================================================
            #  Detection tab
            # ==============================================================
            with gr.TabItem("Detection", elem_id="ad_adv_tab_detection", id="detection_tab"):
                with gr.Row(equal_height=True):
                    # -- left: controls + selection panel --------
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

                        # -- Selection panel (hidden until detection) --
                        with gr.Group(visible=False, elem_id="ad_adv_selection_panel") as selection_panel:
                            gr.Markdown("### Build inpaint passes")
                            gr.Markdown(
                                "**Click detections on the image** to add them to "
                                "the current pass (hover to preview a region), then "
                                "click **Add Pass** to group them. Repeat to create "
                                "multiple passes. Each pass gets its own accordion "
                                "with prompt and settings."
                            )
                            with gr.Row():
                                add_pass_btn = gr.Button(
                                    "\u2795 Add Pass",
                                    variant="secondary",
                                    elem_id="ad_adv_add_pass",
                                )
                                auto_pass_btn = gr.Button(
                                    "\u26a1 Auto: 1 per detection",
                                    variant="secondary",
                                    elem_id="ad_adv_auto_pass",
                                )
                                clear_passes_btn = gr.Button(
                                    "\U0001f5d1\ufe0f Clear Passes",
                                    variant="secondary",
                                    elem_id="ad_adv_clear_passes",
                                )
                            passes_display = gr.HTML(
                                value="",
                                elem_id="ad_adv_passes_display",
                            )
                            selection_status = gr.HTML(
                                value="",
                                elem_id="ad_adv_selection_status",
                            )
                            confirm_btn = gr.Button(
                                "Confirm & Continue \u279c",
                                variant="primary",
                                elem_id="ad_adv_confirm_btn",
                            )

                    # -- right: input image + preview -----------
                    with gr.Column(scale=1, elem_id="ad_adv_input_col"):
                        with gr.Column(variant="panel"):
                            with gr.Group(visible=True) as input_image_group:
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
                            with gr.Group():
                                # Interactive overlay: base image + clickable
                                # region divs (see _render_region_overlay).
                                region_overlay = gr.HTML(
                                    value="",
                                    elem_id="ad_adv_region_overlay",
                                )
                                # Hidden bridge written by the adAdvPickRegion JS
                                # hook on each region click ("<idx>|<nonce>").
                                region_click = gr.Textbox(
                                    value="",
                                    visible=False,
                                    elem_id="ad_adv_region_click",
                                )

            # Expose input_image at module level for cross-tab wiring
            global ad_plus_input_gallery
            ad_plus_input_gallery = input_image

            # Register paste-params destination (Style Picker fields are added
            # once the picker exists, in the Inpainting tab)
            adp_paste_fields: list = []
            parameters_copypaste.add_paste_fields(
                "adetailer_plus", _hidden_input, adp_paste_fields, None
            )
            _hidden_input.change(
                # Forward the pasted PIL into the Gallery AND re-show the gallery
                # group, so a send-to after a prior detection isn't stuck hidden.
                fn=lambda img: (
                    [img] if img is not None else [],
                    gr.update(visible=True),
                ),
                inputs=[_hidden_input],
                outputs=[input_image, input_image_group],
                show_progress=False,
            )

            # ==============================================================
            #  Inpainting tab
            # ==============================================================
            with gr.TabItem("Inpainting", elem_id="ad_adv_tab_inpainting", id="inpainting_tab"):
                with gr.Row(equal_height=True):
                    # -- left: settings + per-pass accordions ---
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
                                # Only pre-select defaults that actually exist
                                # on this install (avoids Gradio dropping/erroring
                                # on unknown style names).
                                _default_styles = [
                                    s for s in ("Illustrious",)
                                    if s in style_names
                                ]
                                common_styles = gr.Dropdown(
                                    label="Styles",
                                    choices=style_names,
                                    multiselect=True,
                                    value=_default_styles,
                                    elem_id="ad_adv_styles",
                                )
                            style_picker_slots = style_picker_hook.create_ui()
                            adp_paste_fields.extend(
                                parameters_copypaste.PasteField(*f)
                                for f in style_picker_hook.paste_fields()
                            )

                        # Per-pass accordions (one per pass, up to MAX_PASSES)
                        det_accordions: list[gr.Accordion] = []
                        det_mask_previews: list[gr.Image] = []
                        det_enables = []  # InputAccordion (acts as Checkbox)
                        det_prompts: list[gr.Textbox] = []
                        det_neg_prompts: list[gr.Textbox] = []
                        det_denoises: list[gr.Slider] = []
                        det_mask_blurs: list[gr.Slider] = []
                        det_dilates: list[gr.Slider] = []

                        for i in range(MAX_PASSES):
                            with InputAccordion(
                                value=True,
                                label=f"Pass {i + 1}",
                                visible=False,
                                elem_id=f"ad_adv_det_{i}",
                            ) as en:
                                with gr.Row(equal_height=True):
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
                            "Process All Passes",
                            variant="primary",
                            elem_id="ad_adv_process_btn",
                        )

                    # -- right: output --------------------------
                    with gr.Column(scale=1, elem_id="ad_adv_results"):
                        with gr.Tabs(elem_id="ad_adv_result_tabs") as result_tabs:
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
                                            '\U0001f504', elem_id="ad_adv_reuse_output",
                                            tooltip="Send output back to ADetailer+ input",
                                        )
                                        save_btn = ToolButton(
                                            '\U0001f4be', elem_id="ad_adv_save",
                                            tooltip=f"Save image ({opts.outdir_save})",
                                        )
                                        send_img2img_btn = ToolButton(
                                            '\U0001f5bc\ufe0f', elem_id="ad_adv_send_img2img",
                                            tooltip="Send to img2img",
                                        )
                                        send_inpaint_btn = ToolButton(
                                            '\U0001f3a8\ufe0f', elem_id="ad_adv_send_inpaint",
                                            tooltip="Send to img2img inpaint",
                                        )
                                        send_extras_btn = ToolButton(
                                            '\U0001f4d0', elem_id="ad_adv_send_extras",
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

        # ==============================================================
        #  Event wiring
        # ==============================================================

        # -- Detection --------------------------------------------------

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
            return _run_detection(img, models_sel, conf, model_mapping=model_mapping)

        detect_outputs = (
            [region_overlay, detection_state]
            + [selection_panel]
            + [input_image_group]     # hide on success, show otherwise
            + [selection_state]       # reset in-progress selection
            + [selection_status]      # reset status line
            + [passes_state]
            + [passes_display]
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
            fn=lambda: gr.update(value="\u23f3 Detecting\u2026", interactive=False),
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
        )

        # -- Selection (click a region on the overlay) ------------------

        # The adAdvPickRegion JS hook writes "<idx>|<nonce>" into region_click
        # and fires its input event; the nonce defeats Gradio's "value
        # unchanged => no .change" suppression so repeated clicks register.
        region_click.change(
            fn=_on_region_click,
            inputs=[region_click, selection_state, detection_state, passes_state],
            outputs=[selection_state, region_overlay, selection_status],
            show_progress=False,
        )

        # -- Pass building ----------------------------------------------

        pass_outputs = [
            passes_state,
            selection_state,
            region_overlay,
            passes_display,
            selection_status,
        ]

        add_pass_btn.click(
            fn=_add_pass,
            inputs=[passes_state, selection_state, detection_state],
            outputs=pass_outputs,
        )

        auto_pass_btn.click(
            fn=_auto_one_per_detection,
            inputs=[detection_state, passes_state],
            outputs=pass_outputs,
        )

        clear_passes_btn.click(
            fn=_clear_passes,
            inputs=[detection_state],
            outputs=[*pass_outputs, input_image_group],
        )

        # -- Confirm passes -> populate accordions -----------------------

        confirm_outputs = (
            det_accordions
            + det_enables
            + det_mask_previews
            + det_prompts
            + det_neg_prompts
            + det_denoises
            + det_mask_blurs
            + det_dilates
        )

        def _confirm_and_check(det_state, passes):
            if not passes:
                N = MAX_PASSES
                return (
                    [gr.update()] * N     # accordions
                    + [gr.update()] * N   # enables
                    + [gr.update()] * N   # mask previews
                    + [gr.update()] * N   # prompts
                    + [gr.update()] * N   # neg prompts
                    + [gr.update()] * N   # denoise
                    + [gr.update()] * N   # blur
                    + [gr.update()] * N   # dilate
                )
            return _confirm_passes(det_state, passes)

        confirm_btn.click(
            fn=_confirm_and_check,
            inputs=[detection_state, passes_state],
            outputs=confirm_outputs,
        ).then(
            fn=lambda passes: gr.update(selected="inpainting_tab") if passes else gr.update(),
            inputs=[passes_state],
            outputs=main_tabs,
        )

        # -- Processing -------------------------------------------------

        def _process_and_wrap(det_state, passes, *args):
            result_img, status = _process_all(det_state, passes, *args)
            gallery_val = [result_img] if result_img is not None else []
            return gallery_val, f"<p>{status}</p>" if status else ""

        process_inputs = (
            [detection_state, passes_state]
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
            + style_picker_slots
        )

        process_btn.click(
            fn=lambda: gr.update(value="\u23f3 Processing\u2026", interactive=False),
            inputs=None,
            outputs=process_btn,
        ).then(
            fn=_process_and_wrap,
            inputs=process_inputs,
            outputs=[output_gallery, html_log],
        ).then(
            fn=lambda: gr.update(value="Process All Passes", interactive=True),
            inputs=None,
            outputs=process_btn,
        ).then(
            fn=lambda: gr.update(selected="output_tab"),
            inputs=None,
            outputs=result_tabs,
        )

        # -- Save button ------------------------------------------------

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

        # -- Reuse button -----------------------------------------------

        def _reuse_output(gallery_images):
            # Send the output back to the input as a fresh, undetected image:
            # re-show the (possibly hidden) upload gallery, clear the stale
            # overlay, hide the selection panel, and reset detection state so the
            # user lands on a clean Detection tab ready to re-run on the new image.
            no_op = (gr.update(),) * 6
            if not gallery_images:
                return no_op
            first = gallery_images[0]
            img = first[0] if isinstance(first, (list, tuple)) else first
            return (
                [img],                      # input_image
                gr.update(visible=True),    # input_image_group re-shown
                "",                         # region_overlay cleared
                gr.update(visible=False),   # selection_panel hidden
                None,                       # detection_state reset
                [],                         # selection_state reset
            )

        reuse_btn.click(
            fn=_reuse_output,
            inputs=[output_gallery],
            outputs=[
                input_image,
                input_image_group,
                region_overlay,
                selection_panel,
                detection_state,
                selection_state,
            ],
        ).then(
            fn=lambda: gr.update(selected="detection_tab"),
            inputs=None,
            outputs=main_tabs,
        )

        # -- Send-to buttons --------------------------------------------

        style_picker_hook.link_send_button("adetailer_plus", "img2img", send_img2img_btn)
        style_picker_hook.link_send_button("adetailer_plus", "img2img", send_inpaint_btn)

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
