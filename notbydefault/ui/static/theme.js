// Runs before first paint: apply the remembered theme (if any) so there is no flash.
(function () {
  var t = null;
  try { t = window.localStorage.getItem("nbd-theme"); } catch (e) { t = null; }
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
})();
