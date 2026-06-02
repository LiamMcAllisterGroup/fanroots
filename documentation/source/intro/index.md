# Introduction

**fanroots** is a modular Python library for root-finding and optimisation of
functions that are piecewise-defined on the secondary fan of a toric variety,
with applications to locating Kähler moduli satisfying prescribed conditions in
string compactifications.

## The optimisation workflow at a glance

The diagram below traces a typical run of `FanRoots` from initial heights to a
converged root.

```{raw} html
<div class="fr-fig mathjax_process" id="f1-workflow">
  <div class="fr-chart" id="f1-workflow-chart">

    <svg class="fr-asvg" id="f1-workflow-svg"></svg>

    <!-- ── Step 1: Input ──────────────────────────────────────────────── -->
    <div class="fr-step">
      <div class="fr-step-hdr blue">
        <span class="lbl">Step 1 &mdash; Input</span>
        <span class="rl"></span>
      </div>
      <div class="bx lb" id="f1-s1-input">
        <div class="t">VectorConfiguration &amp; objective</div>
        <div class="d">
          <code>vc</code> &nbsp;&mdash;&nbsp; vector configuration (cytools)<br>
          <code>fct(optimizer, x)</code> &nbsp;&mdash;&nbsp; residual function $F(h)$<br>
          <code>jac(optimizer, x)</code> &nbsp;&mdash;&nbsp; Jacobian $\partial F / \partial h$
        </div>
      </div>
    </div>

    <div class="gap-sm"></div>

    <!-- ── Step 2: Initialisation ─────────────────────────────────────── -->
    <div class="fr-step">
      <div class="fr-step-hdr blue">
        <span class="lbl">Step 2 &mdash; Initialisation</span>
        <span class="rl"></span>
      </div>
      <div class="bx lb" id="f1-s2-init">
        <div class="t">FanRoots constructor</div>
        <div class="d">
          Set triangulation &amp; intersection tensor $\kappa$<br>
          Set <code>step_taking_schedule</code> (FlopStep / JumpStep)<br>
          Set momentum, learning rate, tolerance
        </div>
      </div>
    </div>

    <div class="gap-sm"></div>

    <!-- ── Step 3: Optimisation loop ──────────────────────────────────── -->
    <div class="fr-step">
      <div class="fr-step-hdr blue">
        <span class="lbl">Step 3 &mdash; Optimisation loop</span>
        <span class="rl"></span>
      </div>

      <!-- three-column sub-grid -->
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;">

        <div class="bx lb2" id="f1-s3-proposal">
          <div class="t">Step proposal</div>
          <div class="d">
            <code>propose_newton</code><br>
            <code>propose_gauss_newton</code><br>
            <code>propose_lma</code><br>
            <code>propose_gradient_descent</code><br>
            <span style="font-size:9px;color:#888;">QR-pivoted lstsq (gelsy)</span>
          </div>
        </div>

        <div class="bx lb2" id="f1-s3-size">
          <div class="t">Step size</div>
          <div class="d">
            $\alpha$ via <code>naive</code> / <code>shrink</code><br>
            <code>backtracking_line_search</code><br>
            <code>ternary</code><br>
            scale by momentum $\cdot$ lr
          </div>
        </div>

        <div class="bx lb2" id="f1-s3-taking">
          <div class="t">Step taking</div>
          <div class="d">
            <code>FlopStep</code> (<code>flop_linear</code>)<br>
            <code>JumpStep</code> (subdivide)<br>
            update triangulation<br>
            update $\kappa$
          </div>
        </div>

      </div><!-- /grid -->

      <div class="gap-sm"></div>

      <!-- convergence check -->
      <div class="bx gh" id="f1-s3-check"
           style="max-width:320px;margin:0 auto;">
        <div class="t">Convergence check</div>
        <div class="d">$\|F(h)\|_2^2 \;<\; \texttt{tol}^2$&nbsp;?</div>
      </div>

    </div><!-- /step 3 -->

    <div class="gap-sm"></div>

    <!-- ── Step 4: Output ─────────────────────────────────────────────── -->
    <div class="fr-step">
      <div class="fr-step-hdr green">
        <span class="lbl">Step 4 &mdash; Output</span>
        <span class="rl"></span>
      </div>
      <div class="bx dg" id="f1-s4-output">
        <div class="t">Converged solution</div>
        <div class="d">
          Heights $h^*$, triangulation, $\kappa$<br>
          Solution vector &amp; status via <code>get_status()</code> / <code>get_state()</code>
        </div>
      </div>
    </div>

  </div><!-- /fr-chart -->
</div><!-- /fr-fig -->

<script>
(function () {
  "use strict";

  function getCenter(el, container) {
    var er = el.getBoundingClientRect();
    var cr = container.getBoundingClientRect();
    return {
      x: er.left - cr.left + er.width / 2,
      y: er.top  - cr.top  + er.height / 2,
      top:    er.top    - cr.top,
      bottom: er.bottom - cr.top,
      left:   er.left   - cr.left,
      right:  er.right  - cr.left,
      w: er.width,
      h: er.height,
    };
  }

  function makePath(d, stroke, dasharray) {
    var p = document.createElementNS("http://www.w3.org/2000/svg", "path");
    p.setAttribute("d", d);
    p.setAttribute("fill", "none");
    p.setAttribute("stroke", stroke);
    p.setAttribute("stroke-width", "1.6");
    p.setAttribute("marker-end", "url(#f1-arr-" + stroke.replace("#","") + ")");
    if (dasharray) p.setAttribute("stroke-dasharray", dasharray);
    return p;
  }

  function lineV(x, y1, y2) {
    return "M " + x + " " + y1 + " L " + x + " " + y2;
  }
  function lineH(x1, y1, x2, y2) {
    return "M " + x1 + " " + y1 + " L " + x2 + " " + y1 + " L " + x2 + " " + y2;
  }

  function addMarker(defs, id, color) {
    var m = document.createElementNS("http://www.w3.org/2000/svg", "marker");
    m.setAttribute("id", id);
    m.setAttribute("markerWidth", "8");
    m.setAttribute("markerHeight", "8");
    m.setAttribute("refX", "6");
    m.setAttribute("refY", "3");
    m.setAttribute("orient", "auto");
    var poly = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    poly.setAttribute("points", "0 0, 6 3, 0 6");
    poly.setAttribute("fill", color);
    m.appendChild(poly);
    defs.appendChild(m);
  }

  function drawArrows() {
    var chart = document.getElementById("f1-workflow-chart");
    var svg   = document.getElementById("f1-workflow-svg");
    if (!chart || !svg) return;

    svg.setAttribute("height", chart.offsetHeight);

    // clear
    while (svg.firstChild) svg.removeChild(svg.firstChild);

    var defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    var BLUE  = "#2B5F8E";
    var GREEN = "#3A7D5A";
    var GREY  = "#aaa";
    addMarker(defs, "f1-arr-" + BLUE.replace("#",""),  BLUE);
    addMarker(defs, "f1-arr-" + GREEN.replace("#",""), GREEN);
    addMarker(defs, "f1-arr-" + GREY.replace("#",""),  GREY);
    svg.appendChild(defs);

    var s1 = document.getElementById("f1-s1-input");
    var s2 = document.getElementById("f1-s2-init");
    var p  = document.getElementById("f1-s3-proposal");
    var sz = document.getElementById("f1-s3-size");
    var st = document.getElementById("f1-s3-taking");
    var ck = document.getElementById("f1-s3-check");
    var s4 = document.getElementById("f1-s4-output");

    if (!s1||!s2||!p||!sz||!st||!ck||!s4) return;

    var c1 = getCenter(s1, chart);
    var c2 = getCenter(s2, chart);
    var cp = getCenter(p,  chart);
    var cz = getCenter(sz, chart);
    var ct = getCenter(st, chart);
    var ck_ = getCenter(ck, chart);
    var c4 = getCenter(s4, chart);

    var GAP = 6; // arrow head clearance

    // Step1 -> Step2  (vertical, blue)
    svg.appendChild(makePath(
      lineV(c1.x, c1.bottom + GAP, c2.top - GAP),
      BLUE
    ));

    // Step2 -> Step3 proposal box  (vertical, blue)
    svg.appendChild(makePath(
      lineV(c2.x, c2.bottom + GAP, cp.top - GAP),
      BLUE
    ));

    // proposal -> size  (horizontal, blue)
    svg.appendChild(makePath(
      "M " + (cp.right + GAP) + " " + cp.y + " L " + (cz.left - GAP) + " " + cz.y,
      BLUE
    ));

    // size -> taking  (horizontal, blue)
    svg.appendChild(makePath(
      "M " + (cz.right + GAP) + " " + cz.y + " L " + (ct.left - GAP) + " " + ct.y,
      BLUE
    ));

    // taking -> convergence check  (elbow: down from taking, then left to check centre)
    var midY = ct.bottom + (ck_.top - ct.bottom) / 2;
    svg.appendChild(makePath(
      "M " + ct.x + " " + (ct.bottom + GAP) +
      " L " + ct.x + " " + midY +
      " L " + ck_.x + " " + midY +
      " L " + ck_.x + " " + (ck_.top - GAP),
      BLUE
    ));

    // convergence yes -> Step4  (vertical, green)
    svg.appendChild(makePath(
      lineV(ck_.x, ck_.bottom + GAP, c4.top - GAP),
      GREEN
    ));

    // convergence no -> back to proposal  (dashed grey, left side elbow)
    var loopX = Math.min(cp.left, ck_.left) - 28;
    svg.appendChild(makePath(
      "M " + (ck_.left - GAP) + " " + ck_.y +
      " L " + loopX + " " + ck_.y +
      " L " + loopX + " " + cp.y +
      " L " + (cp.left - GAP) + " " + cp.y,
      GREY,
      "5,3"
    ));
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", drawArrows);
  } else {
    drawArrows();
  }
  window.addEventListener("resize", drawArrows);
})();
</script>
```

## Reading order

The chapters that follow cover the mathematics behind the secondary fan and the
algorithm used to navigate its chamber structure.  For a self-contained code
walkthrough see the demo in `demo/volume_finder.py`, which subclasses `FanRoots`
to target prescribed divisor volumes and illustrates the full
proposal–size–taking pipeline in a realistic setting.
