"""Optional integration with the Style Picker extension (forge-neo-style-picker).

Everything here is a no-op when that extension is not installed.
"""

from __future__ import annotations

try:
    from lib_style_picker import forge as _sp
except ImportError:
    _sp = None

TAB = "adetailer_plus"


def create_ui() -> list:
    """Build the picker in the current layout context; returns its slot components (possibly empty)."""
    return _sp.create_ui(TAB) if _sp else []


def paste_fields() -> list[tuple]:
    return _sp.paste_fields(TAB) if _sp else []


def link_send_button(src_tab: str, dst_tab: str, button):
    if _sp:
        _sp.link_send_button(src_tab, dst_tab, button)


def apply(prompt: str, negative: str, slot_values) -> tuple[str, str, str]:
    """Returns (prompt, negative, infotext value); infotext value is "" when nothing was picked."""
    if not _sp or not slot_values:
        return prompt, negative, ""
    picks = _sp.picks_from_args(list(slot_values))
    if not picks:
        return prompt, negative, ""
    prompt, negative = _sp.apply_to_prompts(prompt, negative, picks)
    return prompt, negative, _sp.serialize(picks)
