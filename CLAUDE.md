# CLAUDE.md — ADetailer Extension Development Guide

## Project Overview

ADetailer is a Stable Diffusion WebUI extension for automatic face/body/object detection and inpainting. This fork adds **ADetailer+**, a standalone top-level tab with an interactive detect→customize→process workflow.

**Host**: SD WebUI Forge ("Neo"), **Gradio 4.40.0** (verify with `venv/Lib/site-packages/gradio-4.40.0.dist-info`; an older note said "3.x" — that was stale)
**Location**: `<webui>/extensions/adetailer/`
**Base/upstream**: rebased onto [ADetailer-Neo](https://github.com/Haoming02/ADetailer-Neo) (`upstream` remote). Neo renamed the core package `adetailer/` → `lib_adetailer/`, moved WebUI-integration helpers into `lib_adetailer/utils/`, and renamed `scripts/!adetailer.py` → `scripts/adetailer.py`.

## Directory Structure

```
lib_adetailer/              # Core library — detection, args, masks, opts (was adetailer/)
  __init__.py               # Exports: __version__, mediapipe_predict, ultralytics_predict, PredictOutput, get_models
  args.py                   # ADetailerArgs, InpaintBBoxMatchMode, INPAINT_BBOX_MATCH_MODES, is_mediapipe()
  mask.py                   # dilate_erode, mask_merge, mask utilities
  opts.py                   # OptimalCropSize (strict / free), dynamic_denoise_strength
  ui.py                     # Original ADetailer accordion UI (inside txt2img/img2img)
  ui_advanced.py            # ADetailer+ standalone tab (THE MAIN FILE FOR NEW FEATURES)
  auto_mapping.py           # default_auto_mapping_json (keyword clusters; used by decider.keyword_suggest)
  decider.py                # Suggest from prompt: region catalog, keyword + System One deciders (stdlib only)
  controlnet.py             # ControlNet integration
  detection/
    common.py               # PredictOutput, get_models, draw_detection_overlay, color_for_label
    mediapipe.py            # MediaPipe (Tasks API; routed by model path, not logical name)
    ultralytics.py          # YOLO ultralytics detection
  utils/
    __init__.py             # ensure_pil_image, print, NUM
    helper.py               # processing helpers (NOTE: no disable_safe_unpickle in Neo)
    p_method.py             # Processing method overrides

controlnet_ext/             # Legacy ControlNet layer (Neo also has lib_adetailer/controlnet.py)
scripts/
  adetailer.py              # Script entry point, lifecycle callbacks, on_after_component, on_ui_tabs
javascript/
  adetailer_plus.js         # switch_to_adetailer_plus() JS function for tab switching
```

> **Migration note**: Neo removed the original `tests/` suite and the `disable_safe_unpickle` context manager (its `ultralytics_predict` loads YOLO directly). The ADetailer+ tab now routes mediapipe-vs-yolo by `not model_name.endswith(".pt")` and passes the **model path** (not a logical name) to `mediapipe_predict`, matching Neo's `is_mediapipe()`.

## ADetailer+ Tab (`lib_adetailer/ui_advanced.py`)

### Architecture

- Registered via `on_ui_tabs` in `!adetailer.py` → returns `(tab, "ADetailer+", "adetailer_advanced")`
- Two-tab layout: **Detection** (model select + input gallery + selection panel) → **Inpainting** (settings + per-pass accordions + output)
- **Pass-based workflow**: Detection → Selection → Confirm → Inpainting → Process

### Workflow

1. **Detect**: Run models → interactive overlay (`region_overlay`) showing the image with one clickable region per detection; selection panel appears
2. **Select**: Click regions on the image to toggle them into the current pass (`selection_state`), then click "Add Pass" to group them (repeat for multiple passes)
3. **Confirm**: Masks within each pass are union-merged; accordions are populated on the Inpainting tab (one per pass)
4. **Process**: Each pass is inpainted sequentially with its merged mask

### Key Constants

- `MAX_DETECTIONS = 8` — maximum raw detections per image
- `MAX_PASSES = 8` — maximum inpaint passes (each gets one accordion)

### Pass State

`passes_state` (a `gr.State`) holds a list of dicts:
```python
[{"label": "Pass 1: [1] face, [3] left eye",
  "det_indices": [0, 2],
  "merged_mask": np.ndarray,
  "merged_bbox": [x1, y1, x2, y2]}, ...]
```

Mask merging uses `mask_merge()` from `lib_adetailer/mask.py` (`cv2.bitwise_or` reduce).
Merged bbox = bounding box of the union (min x1/y1, max x2/y2 across constituent bboxes).

### Flat Return List Convention

**Three flat return lists** must stay in sync:

1. `_run_detection()` returns `7 + 8×MAX_PASSES` items:
   `[region_overlay_html, state, selection_panel_vis, selection_reset, selection_status_reset, passes_reset, passes_html, *accordion×N, *enable×N, *mask×N, *prompt×N, *neg×N, *denoise×N, *blur×N, *dilate×N]`
   Must match `detect_outputs`.

2. `_confirm_passes()` returns `8×MAX_PASSES` items:
   `[*accordion×N, *enable×N, *mask×N, *prompt×N, *neg×N, *denoise×N, *blur×N, *dilate×N]`
   Must match `confirm_outputs`.

3. `_process_all()` unpacks `*args` with `args[k*N : (k+1)*N]`.
   Must match `process_inputs` (which starts with `[detection_state, passes_state]`).

**CRITICAL**: When adding new per-pass fields, update ALL THREE return lists AND their corresponding output/input component lists.

### Interactive Region Overlay (click-to-select)

The right column of the Detection tab is `region_overlay` (a `gr.HTML`), built by
`_render_region_overlay(detection_state, selection_indices, passes)`:
- Base image embedded as a base64 `data:` URI (cached in `detection_state["data_uri"]`
  + `width`/`height` at detection time — never re-encoded per click).
- One absolutely-positioned `<div class="ad-region" data-idx=i onclick="adAdvPickRegion(i)">`
  per detection, sized in **percentages** of the image (responsive, no JS resize).
  Smaller boxes are emitted last so they sit on top and stay clickable.
- Each region floats a `[i] name conf` label and highlights on CSS `:hover`.
  States: available / `selected` (✓) / `assigned` (locked, shows `Pass N`,
  `pointer-events:none`). CSS lives in the `_REGION_STYLE` constant.

Click plumbing: `adAdvPickRegion(idx)` (in `javascript/adetailer_plus.js`) writes
`"<idx>|<nonce>"` into the hidden `region_click` textbox and fires its `input`
event. The nonce defeats Gradio's "value unchanged ⇒ no `.change`" suppression so
repeated clicks of the same region register. `region_click.change` →
`_on_region_click` toggles the index in `selection_state` and re-renders the overlay.

### Selection Panel

Located on the Detection tab (below detect controls), hidden until detection runs.
Contains:
- `selection_state` (`gr.State`) — in-progress selection (detection indices for the
  pass being built); driven entirely by clicks on `region_overlay`
- **Add Pass** / **Auto: 1 per detection** / **Clear Passes** buttons
- `passes_display` — HTML showing current pass assignments
- `selection_status` — HTML readout of the current selection
- **Confirm & Continue** button

All pass-building callbacks (`_add_pass`, `_auto_one_per_detection`, `_clear_passes`)
share one output convention:
`[passes_state, selection_state, region_overlay, passes_display, selection_status]`.
Detections assigned to a pass become `assigned` (locked) on the overlay (no
double-assignment). "Auto: 1 per detection" creates one pass per unassigned detection.

### Cross-Tab Send-To Buttons

**Problem**: `on_app_started` fires AFTER `demo.launch()` — Gradio routes are finalized, `.click()` registrations are silently ignored.

**Solution**: Use the `parameters_copypaste` mechanism:
1. `add_paste_fields("adetailer_plus", hidden_image, [], None)` — registers destination
2. `register_paste_params_button(ParamBinding(...))` — registers source (in `on_after_component`)
3. `connect_paste_params_buttons()` — wires everything (called inside `with gr.Blocks() as demo:` in main `ui.py`)

**Hidden Image bridge**: `image_from_url_text` returns a single PIL image, but `gr.Gallery` expects a list. A hidden `gr.Image(visible=False)` receives the PIL, then `.change()` wraps it as `[img]` and forwards to the Gallery.

The JS function `switch_to_adetailer_plus()` (in `javascript/adetailer_plus.js`) is auto-called by `connect_paste_params_buttons` via `_js=f"switch_to_{binding.tabname}"`.

### Suggest from Prompt (`lib_adetailer/decider.py`)

Detection tab: **Source prompt** box + **🧭 Suggest from prompt** button. Suggest pre-selects
detection models; after **Confirm**, a chained `.then(_fill_pass_prompts)` writes each pass's
prompt from the tags of the regions its detections came from (it only touches `det_prompts`,
so the three flat return lists are unchanged).

- `decider.py` is **stdlib only, no module-level imports from `lib_adetailer`** — the decider lab
  (`C:\Projects\forge-neo-decider-lab`) loads it by path and drives it with in-process Laya.
  `REGIONS` (region → booru description → model files in preference order) is the catalog both use.
- Deciders: `keyword_suggest` (the `auto_mapping.py` clusters) or `DecisionModelDecider` over a
  System One client (`SystemOneHTTP`: `POST /v1/systemone`, the Jev wire format). Chosen by the
  `ad_plus_decider*` settings (Settings → ADetailer). A failing server falls back to keywords.
- The server runs **outside Forge**: Laya needs transformers 5.x, Forge pins 4.57. With
  `ad_plus_decider_command` set (e.g. `uv run --directory C:\Projects\forge-neo-decider-lab python
  serve.py`), `decider.launcher` runs it on the first Suggest whose `GET /health` fails, waits for
  it, and kills its process tree at Forge exit (`atexit`); a server it didn't start is never
  stopped. Blank = start `serve.py` by hand (binds 127.0.0.1:8000, CPU).
- Source prompt is filled on txt2img/img2img **🔀 send-to**: `on_ui_tabs` passes the (button,
  gallery) pairs as `send_sources`, wired with `js="extract_image_from_gallery"` and read via
  `images.read_info_from_image`. The send-to `ParamBinding` itself has no `source_tabname`, so
  Forge's paste fields never carry the prompt. Senders that paste infotext (Infinite Image
  Browsing's patched "Send to ADetailer+", which binds with `source_text_component`) fill it
  through the `PasteField(source_prompt, "Prompt")` in `adp_paste_fields`. Both paths run the
  infotext-pasted callbacks, so Style Picker strips its appended picks. If the box is empty,
  Suggest falls back to the gallery image's own infotext.
- `suggest_state` is `{"prompt", "items"}`; Confirm only pre-fills when the Source prompt still
  equals that prompt, so a suggestion never leaks onto the next image.

### Tag Autocomplete Integration

The `a1111-sd-webui-tagcomplete` extension uses CSS selectors in `_textAreas.js`. An `"adetailer-plus"` entry targets:
- `[id^=ad_adv_det_][id$=_prompt] textarea`
- `[id^=ad_adv_det_][id$=_neg] textarea`

With `onDemand: true` + `base: "#tab_adetailer_advanced"` so MutationObserver catches dynamically shown InputAccordions.

**Elem ID convention**: `ad_adv_det_{i}_prompt`, `ad_adv_det_{i}_neg` — keep this pattern for new prompt fields.

### InputAccordion Dual Role

`InputAccordion` yields a component that acts as `gr.Checkbox` (stored in `det_enables`). Its `.accordion` attribute is the `gr.Accordion` for visibility/label updates (stored in `det_accordions`). These are DIFFERENT component lists.

### gr.State Serialization

Detection state stores masks as `np.ndarray`, not PIL. `gr.State` round-trips through JSON; PIL objects lose metadata. Convert back with `Image.fromarray()` in `_process_all`.

## `scripts/adetailer.py` — Lifecycle Hooks

### Registered Callbacks (bottom of file)
```python
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_after_component(on_after_component)
script_callbacks.on_app_started(add_api_endpoints)
script_callbacks.on_before_ui(on_before_ui)
script_callbacks.on_ui_tabs(on_ui_tabs)
```

### `on_after_component` — What It Does
- Captures `txt2img_generate`, `img2img_generate` buttons
- Captures `txt2img_gallery`, `img2img_gallery` output galleries
- Creates 🔀 ToolButtons after `txt2img_send_to_extras` / `img2img_send_to_extras`
- Registers `ParamBinding` for each ToolButton immediately (not deferred)

### Execution Order in `modules/ui.py`
1. txt2img/img2img UI construction → `on_after_component` fires per component
2. `ui_tabs_callback()` → `on_ui_tabs()` → `create_advanced_tab()`
3. `connect_paste_params_buttons()` — wires all registered bindings
4. `.render()` loop
5. `demo.launch()`
6. `on_app_started` callbacks ← TOO LATE for new `.click()` wiring

## Known Compatibility Guards

### vec_cc (Vectorscope CC extension)
Patches `KDiffusionSampler.vec_cc` lazily during `process_batch`. ADetailer+ creates its own `StableDiffusionProcessingImg2Img`, so the attribute might not exist. Guard:
```python
if not hasattr(KDiffusionSampler, "vec_cc"):
    KDiffusionSampler.vec_cc = {"enable": False}
```

### Script Runner Isolation
`_process_all` shallow-copies `scripts_img2img` and sets `alwayson_scripts = []` to prevent recursive re-entry (the main ADetailer script would otherwise fire inside the inner img2img).

### safe_unpickle
Neo's `ultralytics_predict` (in `lib_adetailer/detection/ultralytics.py`) loads YOLO directly via `from ultralytics import YOLO; YOLO(model_path)` — there is **no** `disable_safe_unpickle` context manager in Neo (the original fork's `aaaaaa/helper.py` version was dropped during the migration). If safe-unpickle errors resurface on this host, re-introduce a guard in `lib_adetailer/utils/helper.py` and wrap the call in `ui_advanced.py`.

## Defaults

| Setting | Default | Source |
|---------|---------|--------|
| Steps | 25 | Hardcoded |
| CFG | 5.0 | Hardcoded |
| Width/Height | from ui-config.json | `txt2img/Width/value` |
| Sampler | Euler a | Hardcoded |
| Scheduler | Automatic | Hardcoded |
| Padding | 32 | Hardcoded |
| BBox Match | Strict | Hardcoded |
| Denoising | 0.4 | Per-pass default |
| Mask Blur | 4 | Per-pass default |
| Dilate | 4 | Per-pass default |
| Styles | ["Illustrious", "style"] | Hardcoded |

## Common Pitfalls

1. **Output list mismatch** — Adding a per-pass field to `_run_detection` / `_confirm_passes` but forgetting `detect_outputs` / `confirm_outputs` (or vice versa) causes a silent Gradio error with no visible feedback. Three lists must stay in sync.

2. **Gallery vs Image** — `gr.Gallery` expects `list[PIL]`, `gr.Image` expects `PIL|None`. The paste-params mechanism outputs single PIL. Always bridge with hidden Image → `.change()`.

3. **on_app_started is too late** — Never register `.click()` handlers there. Use `register_paste_params_button` or wire inside `create_advanced_tab`.

4. **Re-detection must reset everything** — If detection is re-run, ALL passes are cleared, selection panel is reset, and ALL accordion inputs are reset to defaults. This prevents stale values from previous detections leaking.

5. **elem_id naming for tag autocomplete** — If you rename elem_ids on prompt/neg textboxes, update the CSS selectors in `a1111-sd-webui-tagcomplete/javascript/_textAreas.js`.

6. **Gradio 4.40.0** — Forge Neo runs Gradio 4.40, not 3.x. Prefer `gr.update(...)` for updates. `InputAccordion` is a WebUI custom component, not upstream Gradio. `gr.HTML` renders raw HTML and inline `onclick` attributes execute (used by the region overlay).

7. **`region_click` nonce** — The overlay's click bridge appends `|<Date.now()>` to the value it writes into `region_click`, otherwise Gradio suppresses `.change` when the same region is clicked twice. `_on_region_click` strips the nonce with `raw.split("|", 1)[0]`. Keep both halves in sync if you touch either.
