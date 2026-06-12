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
})();
