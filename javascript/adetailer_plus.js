// ADetailer+ : switch_to_adetailer_plus for native send-to button wiring
(function () {
    "use strict";

    // ── switch to ADetailer+ top-level tab ──────────────────────────
    function switch_to_adetailer_plus() {
        var allBtns = gradioApp().querySelectorAll('#tabs > .tab-nav > button');
        for (var i = 0; i < allBtns.length; i++) {
            if (allBtns[i].textContent.trim() === 'ADetailer+') {
                allBtns[i].click();
                return Array.from(arguments);
            }
        }
        return Array.from(arguments);
    }
    window.switch_to_adetailer_plus = switch_to_adetailer_plus;

    // ── click a detection region on the interactive overlay ─────────
    // Writes "<idx>|<nonce>" into the hidden #ad_adv_region_click textbox and
    // fires its input event. The nonce guarantees Gradio's .change listener
    // fires even when the same region is clicked twice in a row.
    function adAdvPickRegion(idx) {
        var root = (typeof gradioApp === 'function') ? gradioApp() : document;
        var wrap = root.querySelector('#ad_adv_region_click');
        if (!wrap) { return; }
        var field = wrap.querySelector('textarea, input');
        if (!field) { return; }
        field.value = idx + '|' + Date.now();
        field.dispatchEvent(new Event('input', { bubbles: true }));
    }
    window.adAdvPickRegion = adAdvPickRegion;
})();
