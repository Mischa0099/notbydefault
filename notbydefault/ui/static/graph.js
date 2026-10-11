// Decision Graph view: Cytoscape + dagre, live updates, focus and requirement impact.
(function () {
  "use strict";
  var box = document.getElementById("cy");
  if (!box || !window.cytoscape) return;
  if (window.cytoscapeDagre) window.cytoscape.use(window.cytoscapeDagre);

  var SHAPES = { goal: "round-rectangle", requirement: "hexagon", constraint: "octagon",
    decision: "round-rectangle", evidence: "barrel", alternative: "tag", code: "rectangle",
    pivot: "round-pentagon" };
  var WIDTH = { decision: 176, goal: 190, code: 150, evidence: 160, alternative: 170 };
  var params = new URLSearchParams(window.location.search);
  var version = null, maxId = 0, focusId = params.get("focus"), reqId = params.get("req");

  function v(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

  function style() {
    var C = { ink: v("--ink"), paper: v("--paper"), sheet: v("--sheet"), rule: v("--rule"),
      muted: v("--muted"), accent: v("--accent"), ok: v("--ok"), okb: v("--ok-bg"), warn: v("--warn"),
      warnb: v("--warn-bg"), bad: v("--bad"), badb: v("--bad-bg"), pivot: v("--pivot"),
      pivotb: v("--pivot-bg"), neutral: v("--neutral"), neutralb: v("--neutral-bg"), p2: v("--paper-2") };
    var mono = v("--mono") || "monospace", sans = v("--sans") || "sans-serif";
    return [
      { selector: "node", style: {
        "shape": function (n) { return SHAPES[n.data("type")] || "ellipse"; },
        "width": function (n) { return WIDTH[n.data("type")] || 150; }, "height": 46,
        "background-color": C.sheet, "border-width": 1.5, "border-color": C.rule,
        "label": "data(label)", "color": C.ink, "font-family": sans, "font-size": 12,
        "text-wrap": "wrap", "text-max-width": function (n) { return (WIDTH[n.data("type")] || 150) - 22; },
        "text-valign": "center", "text-halign": "center", "transition-property": "opacity, border-width",
        "transition-duration": "250ms" } },
      { selector: "node[type = 'goal']", style: { "background-color": C.ink, "color": C.paper, "border-color": C.ink, "font-weight": 600 } },
      { selector: "node[type = 'requirement'], node[type = 'constraint']", style: { "border-color": C.accent, "border-width": 2, "background-color": C.sheet } },
      { selector: "node[type = 'constraint']", style: { "border-style": "dashed" } },
      { selector: "node[type = 'decision']", style: { "font-family": mono, "font-weight": 700, "font-size": 12.5, "border-width": 2.5,
        "background-color": C.neutralb, "border-color": C.neutral, "height": 52 } },
      { selector: "node[type = 'decision'][status = 'reviewed']", style: { "background-color": C.okb, "border-color": C.ok } },
      { selector: "node[type = 'decision'][status = 'needs_review']", style: { "background-color": C.warnb, "border-color": C.warn } },
      { selector: "node[type = 'decision'][status = 'stale']", style: { "background-color": C.badb, "border-color": C.bad } },
      { selector: "node[type = 'decision'][?by_default]", style: { "border-style": "double", "border-width": 4 } },
      { selector: "node[type = 'evidence']", style: { "border-color": C.ok, "color": C.ink } },
      { selector: "node[type = 'alternative']", style: { "border-style": "dashed", "border-color": C.muted, "color": C.muted } },
      { selector: "node[type = 'code']", style: { "font-family": mono, "font-size": 10, "background-color": C.p2, "border-color": C.rule } },
      { selector: "node[type = 'pivot']", style: { "background-color": C.pivotb, "border-color": C.pivot, "border-width": 2.5, "color": C.pivot, "font-weight": 700 } },
      { selector: "edge", style: { "width": 1.4, "line-color": C.rule, "target-arrow-color": C.rule,
        "target-arrow-shape": "triangle", "arrow-scale": 0.8, "curve-style": "bezier", "opacity": 0.9 } },
      { selector: "edge[type = 'depends_on']", style: { "line-color": C.accent, "target-arrow-color": C.accent } },
      { selector: "edge[type = 'serves']", style: { "line-color": C.ink, "target-arrow-color": C.ink, "opacity": 0.55 } },
      { selector: "edge[type = 'supported_by']", style: { "line-color": C.ok, "target-arrow-color": C.ok } },
      { selector: "edge[type = 'rejects']", style: { "line-style": "dashed", "line-color": C.muted, "target-arrow-color": C.muted, "target-arrow-shape": "tee" } },
      { selector: "edge[type = 'affects']", style: { "line-style": "dotted", "line-color": C.muted, "target-arrow-color": C.muted } },
      { selector: "node:selected", style: { "border-width": 4, "border-color": C.accent, "overlay-opacity": 0 } },
      { selector: ".filtered, .hidden", style: { "display": "none" } },
      { selector: ".dim", style: { "opacity": 0.18 } },
      { selector: "node.impact", style: { "border-width": 5, "underlay-color": C.warn, "underlay-opacity": 0.35, "underlay-padding": 8 } },
      { selector: "node.impact-src", style: { "underlay-color": C.accent, "underlay-opacity": 0.3, "underlay-padding": 8 } },
      { selector: "edge.impact", style: { "width": 3, "line-color": C.warn, "target-arrow-color": C.warn, "opacity": 1 } },
      { selector: "node.fresh", style: { "underlay-color": C.warn, "underlay-opacity": 0.5, "underlay-padding": 12 } }
    ];
  }

  var cy = window.cytoscape({ container: box, elements: [], style: style(), wheelSensitivity: 0.25,
    minZoom: 0.2, maxZoom: 2.5, boxSelectionEnabled: false });
  window.nbdCy = cy;
  document.addEventListener("nbd:theme", function () { cy.style(style()); });

  function layout(animate) {
    var eles = cy.elements().not(".filtered").not(".hidden");
    if (eles.empty()) return;
    // one rank per kind of thing a decision points at (only the kinds on screen), so no
    // single rank gets crowded: decisions | code | evidence | alternatives | requirements | goal
    var order = ["affects", "supported_by", "rejects", "depends_on"], seen = {};
    eles.edges().forEach(function (e) { seen[e.data("type")] = true; });
    var rank = {}, k = 0;
    order.forEach(function (t) { if (seen[t]) rank[t] = ++k; });
    function opts(dir, fit) {
      return { name: window.cytoscapeDagre ? "dagre" : "breadthfirst", rankDir: dir,
        nodeSep: dir === "LR" ? 14 : 12, rankSep: dir === "LR" ? 60 : 46, edgeSep: 6,
        ranker: "network-simplex",
        minLen: function (e) {
          var t = e.data("type");
          if (t === "serves") return e.source().data("type") === "decision" ? k + 1 : 1;
          return rank[t] || 1;
        },
        animate: !!animate && fit, animationDuration: 450, fit: fit, padding: 24 };
    }
    // left-to-right or top-to-bottom: whichever lets the graph fill the canvas at a larger zoom
    var best = "LR", bestZoom = 0, w = box.clientWidth - 48, h = box.clientHeight - 48;
    ["LR", "TB"].forEach(function (dir) {
      eles.layout(opts(dir, false)).run();
      var bb = eles.nodes().boundingBox();
      var z = Math.min(w / Math.max(bb.w, 1), h / Math.max(bb.h, 1));
      if (z > bestZoom) { bestZoom = z; best = dir; }
    });
    eles.layout(opts(best, true)).run();
  }

  // --- filters, focus, requirement impact --------------------------------------------
  function hiddenTypes() {
    var off = [];
    document.querySelectorAll(".type-filters input[type=checkbox]").forEach(function (c) { if (!c.checked) off.push(c.value); });
    return off;
  }
  function applyFilters() {
    var off = hiddenTypes();
    cy.nodes().forEach(function (n) { n.toggleClass("filtered", off.indexOf(n.data("type")) >= 0); });
    cy.edges().forEach(function (e) { e.toggleClass("filtered", e.source().hasClass("filtered") || e.target().hasClass("filtered")); });
  }
  function applyFocus() {
    cy.elements().removeClass("hidden");
    if (!focusId) return;
    var n = cy.getElementById("n" + focusId);
    if (n.empty()) return;
    cy.elements().not(n.closedNeighborhood()).addClass("hidden");
  }
  function applyImpact() {
    cy.elements().removeClass("impact impact-src dim");
    if (!reqId) return;
    var r = cy.getElementById("n" + reqId);
    if (r.empty()) return;
    var hit = cy.collection();
    (r.data("impacts") || []).forEach(function (id) { hit = hit.union(cy.getElementById("n" + id)); });
    r.addClass("impact-src");
    hit.addClass("impact");
    var links = hit.edgesWith(r);
    links.addClass("impact");
    cy.elements().not(hit).not(r).not(links).addClass("dim");
  }
  function refresh(animate) { applyFilters(); applyFocus(); applyImpact(); layout(animate); }
  function syncUrl() {
    var q = new URLSearchParams();
    if (focusId) q.set("focus", focusId);
    if (reqId) q.set("req", reqId);
    var s = q.toString();
    history.replaceState(null, "", "/graph" + (s ? "?" + s : ""));
  }

  document.querySelectorAll(".type-filters input").forEach(function (c) { c.addEventListener("change", function () { refresh(true); }); });
  var fs = document.getElementById("focus-select"), rs = document.getElementById("req-select");
  function showAllTypes() { document.querySelectorAll(".type-filters input").forEach(function (c) { c.checked = true; }); }
  if (focusId) showAllTypes();
  if (fs) { fs.value = focusId || ""; fs.addEventListener("change", function () { focusId = fs.value || null; if (focusId) showAllTypes(); syncUrl(); refresh(true); }); }
  if (rs) { rs.value = reqId || ""; rs.addEventListener("change", function () { reqId = rs.value || null; syncUrl(); refresh(false); }); }
  var fb = document.getElementById("fit-btn");
  if (fb) fb.addEventListener("click", function () { cy.fit(cy.elements().not(".filtered").not(".hidden"), 24); });

  // --- side panel ------------------------------------------------------------------
  function showPanel(nid) {
    if (window.htmx) window.htmx.ajax("GET", "/fragments/node/" + nid, { target: "#panel", swap: "innerHTML" });
  }
  cy.on("tap", "node", function (ev) {
    var n = ev.target;
    showPanel(n.data("nid"));
    if (n.data("type") === "requirement" || n.data("type") === "constraint") {
      reqId = String(n.data("nid")); if (rs) rs.value = reqId; syncUrl(); applyImpact();
    }
  });
  cy.on("dbltap", "node", function (ev) { window.location.href = ev.target.data("href"); });

  // --- data + live updates ----------------------------------------------------------
  var toast = document.getElementById("toast"), toastTimer = null;
  function say(text) {
    if (!toast) return;
    toast.textContent = text; toast.classList.add("show");
    clearTimeout(toastTimer); toastTimer = setTimeout(function () { toast.classList.remove("show"); }, 4000);
  }
  function merge(g, first) {
    var added = cy.collection();
    cy.batch(function () {
      g.nodes.forEach(function (n) {
        var ex = cy.getElementById(n.data.id);
        if (ex.nonempty()) ex.data(n.data); else added = added.union(cy.add(n));
      });
      g.edges.forEach(function (e) { if (cy.getElementById(e.data.id).empty()) added = added.union(cy.add(e)); });
    });
    maxId = g.max_id; version = g.version;
    if (first) { refresh(false); if (focusId) showPanel(focusId); return; }
    if (added.nonempty()) {
      refresh(true);
      var fresh = added.nodes();
      fresh.style("opacity", 0);
      fresh.animate({ style: { opacity: 1 } }, { duration: 700 });
      fresh.addClass("fresh");
      setTimeout(function () { fresh.removeClass("fresh"); }, 4500);
      var dec = fresh.filter("[type = 'decision']");
      say(dec.nonempty() ? "New decision caught: " + dec.map(function (d) { return "#" + d.data("nid") + " " + d.data("full"); }).join(", ")
                         : fresh.length + " new node(s) in the graph");
    } else { applyImpact(); }
  }
  function load(first) {
    return fetch("/api/graph", { headers: { "Accept": "application/json" }, cache: "no-store" })
      .then(function (r) { return r.json(); }).then(function (g) { merge(g, first); });
  }
  function setLive(ok) { var l = document.getElementById("live"); if (l) l.classList.toggle("off", !ok); }
  function poll() {
    fetch("/api/version", { cache: "no-store" }).then(function (r) { return r.json(); }).then(function (x) {
      setLive(true);
      if (x.version !== version) return load(false);
    }).catch(function () { setLive(false); });
  }
  load(true).then(function () { setInterval(poll, 2000); });
})();
