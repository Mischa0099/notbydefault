// NotByDefault UI: theme toggle, copy buttons, live-feed highlighting.
(function () {
  "use strict";
  var root = document.documentElement;

  function store(key, value) {
    try { if (value === null) window.localStorage.removeItem(key); else window.localStorage.setItem(key, value); } catch (e) { /* storage blocked */ }
  }
  function currentTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function setTheme(t) {
    root.setAttribute("data-theme", t);
    store("nbd-theme", t);
    var b = document.getElementById("theme-toggle");
    if (b) b.setAttribute("aria-label", t === "dark" ? "Switch to light mode" : "Switch to dark mode");
    document.dispatchEvent(new CustomEvent("nbd:theme", { detail: t }));
  }
  window.nbdTheme = currentTheme;

  document.addEventListener("click", function (ev) {
    var t = ev.target.closest("#theme-toggle");
    if (t) { setTheme(currentTheme() === "dark" ? "light" : "dark"); return; }
    var c = ev.target.closest("[data-copy]");
    if (c) {
      ev.preventDefault();
      var text = c.getAttribute("data-copy");
      var done = function () { c.classList.add("done"); setTimeout(function () { c.classList.remove("done"); }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, done);
      else { var a = document.createElement("textarea"); a.value = text; document.body.appendChild(a); a.select(); try { document.execCommand("copy"); } catch (e) {} a.remove(); done(); }
    }
  });

  // Live feed: remember which events were already on screen; flash the new ones.
  var seen = new Set();
  function mark(scope, flash) {
    (scope || document).querySelectorAll("[data-key]").forEach(function (el) {
      var k = el.getAttribute("data-key");
      if (!seen.has(k)) { if (flash) el.classList.add("fresh"); seen.add(k); }
    });
  }
  document.addEventListener("DOMContentLoaded", function () { mark(document, false); });
  document.addEventListener("htmx:afterSettle", function () { mark(document, true); });

  var live = document.getElementById("live");
  document.addEventListener("htmx:sendError", function () { if (live) live.classList.add("off"); });
  document.addEventListener("htmx:afterRequest", function (ev) {
    if (live && ev.detail && ev.detail.successful) live.classList.remove("off");
  });
})();
