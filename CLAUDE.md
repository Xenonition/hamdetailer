# CLAUDE.md — ADetailer Extension Development Guide

## Project Overview

ADetailer is a Stable Diffusion WebUI extension for automatic face/body/object detection and inpainting. This fork adds **ADetailer+**, a standalone top-level tab with an interactive detect→customize→process workflow.

**Host**: SD WebUI Forge (Gradio 3.x based)
**Location**: `<webui>/extensions/adetailer/`

## Directory Structure

```
adetailer/          # Core library — detection, args, masks, opts
  __init__.py       # Exports: ADETAILER, mediapipe_predict, ultralytics_predict
  args.py           # ADetailerArgs, InpaintBBoxMatchMode, INPAINT_BBOX_MATCH_MODES
  common.py         # PredictOutput, draw_detection_overlay, ensure_pil_image
  mask.py           # dilate_erode, mask utilities
  mediapipe.py      # MediaPipe face detection
  ultralytics.py    # YOLO ultralytics detection
  opts.py           # optimal_crop_size (sdxl / free)

aaaaaa/             # WebUI integration (Script hooks, UI)
  ui.py             # Original ADetailer accordion UI (inside txt2img/img2img)
  ui_advanced.py    # ADetailer+ standalone tab (THE MAIN FILE FOR NEW FEATURES)
  helper.py         # disable_safe_unpickle context manager
  p_method.py       # Processing method overrides
  conditional.py    # Conditional import helpers
  traceback.py      # Traceback utilities

controlnet_ext/     # ControlNet integration layer
scripts/
  !adetailer.py     # Script entry point, lifecycle callbacks, on_after_component
javascript/
  adetailer_plus.js # switch_to_adetailer_plus() JS function for tab switching
tests/              # pytest tests for core library
```

## ADetailer+ Tab (`aaaaaa/ui_advanced.py`)

### Architecture

- Registered via `on_ui_tabs` in `!adetailer.py` → returns `(tab, "ADetailer+", "adetailer_advanced")`
- Two-tab layout: **Detection** (model select + input gallery) → **Inpainting** (settings + per-detection accordions + preview/output)
- Auto-switches tabs after detect (→ Inpainting) and process (→ Output sub-tab)

### Key Constants

- `MAX_DETECTIONS = 8` — maximum simultaneous detections per image

### Flat Return List Convention

`_run_detection()` returns a flat list `[preview, state, *accordion×N, *enable×N, *mask×N, *prompt×N, *neg×N, *denoise×N, *blur×N, *dilate×N]` = `2 + 8×N` items.

**CRITICAL**: When adding new per-detection fields:
1. Add to the return list in `_run_detection()` (both the populated loop AND the `empty` list)
2. Add the component list to `detect_outputs` in `create_advanced_tab()`
3. Both must have identical length and order

`_process_all()` unpacks `*args` with index math `args[k*N : (k+1)*N]`. See its docstring for layout. Update `process_inputs` to match.

### Cross-Tab Send-To Buttons

**Problem**: `on_app_started` fires AFTER `demo.launch()` — Gradio routes are finalized, `.click()` registrations are silently ignored.

**Solution**: Use the `parameters_copypaste` mechanism:
1. `add_paste_fields("adetailer_plus", hidden_image, [], None)` — registers destination
2. `register_paste_params_button(ParamBinding(...))` — registers source (in `on_after_component`)
3. `connect_paste_params_buttons()` — wires everything (called inside `with gr.Blocks() as demo:` in main `ui.py`)

**Hidden Image bridge**: `image_from_url_text` returns a single PIL image, but `gr.Gallery` expects a list. A hidden `gr.Image(visible=False)` receives the PIL, then `.change()` wraps it as `[img]` and forwards to the Gallery.

The JS function `switch_to_adetailer_plus()` (in `javascript/adetailer_plus.js`) is auto-called by `connect_paste_params_buttons` via `_js=f"switch_to_{binding.tabname}"`.

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

## `scripts/!adetailer.py` — Lifecycle Hooks

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
Ultralytics model loading requires temporarily disabling WebUI's safe unpickle checks via `disable_safe_unpickle()` context manager from `aaaaaa/helper.py`.

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
| Denoising | 0.4 | Per-detection default |
| Mask Blur | 4 | Per-detection default |
| Dilate | 4 | Per-detection default |
| Styles | ["Illustrious", "style"] | Hardcoded |

## Common Pitfalls

1. **Output list mismatch** — Adding a per-detection field to `_run_detection` but forgetting `detect_outputs` (or vice versa) causes a silent Gradio error with no visible feedback.

2. **Gallery vs Image** — `gr.Gallery` expects `list[PIL]`, `gr.Image` expects `PIL|None`. The paste-params mechanism outputs single PIL. Always bridge with hidden Image → `.change()`.

3. **on_app_started is too late** — Never register `.click()` handlers there. Use `register_paste_params_button` or wire inside `create_advanced_tab`.

4. **Re-detection must reset fields** — If detection is re-run, ALL per-detection inputs must be reset to defaults. Otherwise stale values from previous detections persist in accordion slots.

5. **elem_id naming for tag autocomplete** — If you rename elem_ids on prompt/neg textboxes, update the CSS selectors in `a1111-sd-webui-tagcomplete/javascript/_textAreas.js`.

6. **Gradio 3.x** — Forge uses Gradio 3.x, not 4.x. Use `gr.update(...)` not component constructors for updates. `InputAccordion` is a WebUI custom component, not upstream Gradio.
