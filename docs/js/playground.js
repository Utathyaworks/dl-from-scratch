/* Interactive demos, one registry entry per lesson.
   Each entry: { id, title, render(container) }. Add a lesson by pushing
   another object onto DEMOS -- nothing else needs to change. */

const DEMOS = [];

/* ------------------------------------------------------------------ utils */
const h = (html) => {
  const t = document.createElement('template');
  t.innerHTML = html.trim();
  return t.content.firstElementChild;
};

/** Render a 2-D array as a table, with an optional per-cell class function. */
function tensorTable(M, caption, cls = () => '') {
  const rows = M.map((row, i) =>
    `<tr>${row.map((v, j) =>
      `<td class="${cls(i, j)}">${typeof v === 'number' ? v : v}</td>`).join('')}</tr>`
  ).join('');
  return `<table class="tensor"><caption>${caption}</caption><tbody>${rows}</tbody></table>`;
}

/* ============================================================ LESSON 01 ===
   Two demos: reducing along an axis, and the broadcasting rule. */

DEMOS.push({
  id: 1,
  title: 'Lesson 01 &mdash; Tensors, Vectors and Shapes',
  render(root) {

    /* ---------------------------------------------- demo A: axis reduction */
    const a = h(`<div class="demo">
      <h3>Which axis disappears?</h3>
      <p class="lede">The index you sum over is the one that vanishes. Change the
        matrix size and the axis, and watch the output shape follow the rule
        &mdash; not your intuition.</p>
      <div class="ctl-row"><label for="rows">rows (samples)</label>
        <input type="range" id="rows" min="1" max="5" value="2"><output id="rows-o"></output></div>
      <div class="ctl-row"><label for="cols">columns (features)</label>
        <input type="range" id="cols" min="1" max="6" value="3"><output id="cols-o"></output></div>
      <div class="ctl-row"><label>reduce along</label>
        <button class="chip" data-ax="0" aria-pressed="true">axis=0 (down rows)</button>
        <button class="chip" data-ax="1" aria-pressed="false">axis=1 (across cols)</button>
        <button class="chip" data-ax="n" aria-pressed="false">everything</button></div>
      <div class="panels" id="a-view"></div>
      <pre class="out" id="a-out"></pre>
    </div>`);

    let axis = '0';

    function drawA() {
      const m = +a.querySelector('#rows').value;
      const n = +a.querySelector('#cols').value;
      a.querySelector('#rows-o').textContent = m;
      a.querySelector('#cols-o').textContent = n;

      // A[i][j] = i*n + j + 1 -- consecutive integers, easy to add in your head
      const A = Array.from({ length: m }, (_, i) =>
        Array.from({ length: n }, (_, j) => i * n + j + 1));

      const sum0 = Array.from({ length: n }, (_, j) =>
        A.reduce((s, row) => s + row[j], 0));
      const sum1 = A.map((row) => row.reduce((s, v) => s + v, 0));
      const total = sum1.reduce((s, v) => s + v, 0);

      let result, shape, expr;
      if (axis === '0') { result = [sum0]; shape = `(${n},)`; expr = `sum over i  ->  i disappears`; }
      else if (axis === '1') { result = sum1.map((v) => [v]); shape = `(${m},)`; expr = `sum over j  ->  j disappears`; }
      else { result = [[total]]; shape = '()'; expr = 'sum over i and j  ->  both disappear'; }

      a.querySelector('#a-view').innerHTML =
        tensorTable(A, `A &nbsp; shape (${m}, ${n})`) +
        `<div style="align-self:center;font-size:22px;color:var(--text-mute)">&rarr;</div>` +
        tensorTable(result, `result &nbsp; shape ${shape}`, () => 'hl');

      a.querySelector('#a-out').innerHTML =
        `shape (${m}, ${n})  ->  ${shape}     <b>${expr}</b>\n` +
        `axis=0 gives one number per FEATURE : [${sum0}]\n` +
        `axis=1 gives one number per SAMPLE  : [${sum1}]\n` +
        `both must total the same:  ${sum0.reduce((s, v) => s + v, 0)} == ${total}  ` +
        `<span class="ok">consistent</span>`;
    }

    a.querySelectorAll('input[type=range]').forEach((r) => r.addEventListener('input', drawA));
    a.querySelectorAll('.chip[data-ax]').forEach((c) => c.addEventListener('click', () => {
      axis = c.dataset.ax;
      a.querySelectorAll('.chip[data-ax]').forEach((o) =>
        o.setAttribute('aria-pressed', String(o === c)));
      drawA();
    }));

    /* ------------------------------------------ demo B: broadcasting rule */
    const SHAPES = [[2, 3], [1, 3], [3], [2], [2, 1], [1, 1], [3, 1], [1, 4]];
    const opt = (s, i) => `<option value="${i}">(${s.join(', ')}${s.length === 1 ? ',' : ''})</option>`;

    const b = h(`<div class="demo">
      <h3>Will this broadcast?</h3>
      <p class="lede">Right-align the shapes, pad the shorter with 1s on the
        <em>left</em>, then every axis pair must be equal or contain a 1. Pick two
        shapes and see the rule run &mdash; including the cases that silently give
        you an answer you never asked for.</p>
      <div class="ctl-row"><label for="sa">shape A</label>
        <select id="sa" class="chip">${SHAPES.map(opt).join('')}</select>
        <label for="sb" style="min-width:auto">&nbsp;+&nbsp; shape B</label>
        <select id="sb" class="chip">${SHAPES.map(opt).join('')}</select></div>
      <div class="panels" id="b-view"></div>
      <pre class="out" id="b-out"></pre>
    </div>`);

    b.querySelector('#sa').value = '0';   // (2,3)
    b.querySelector('#sb').value = '2';   // (3,)

    function pad(s, rank) { return Array(rank - s.length).fill(1).concat(s); }

    function drawB() {
      const sa = SHAPES[+b.querySelector('#sa').value];
      const sb = SHAPES[+b.querySelector('#sb').value];
      const rank = Math.max(sa.length, sb.length);
      const pa = pad(sa, rank), pb = pad(sb, rank);

      const out = [];
      const lines = [];
      let failed = null;
      for (let k = 0; k < rank; k++) {
        const x = pa[k], y = pb[k];
        if (x === y) { out.push(x); lines.push(`  axis ${k}:  ${x} vs ${y}  equal          -> ${x}`); }
        else if (x === 1) { out.push(y); lines.push(`  axis ${k}:  ${x} vs ${y}  A stretches    -> ${y}`); }
        else if (y === 1) { out.push(x); lines.push(`  axis ${k}:  ${x} vs ${y}  B stretches    -> ${x}`); }
        else { failed = k; lines.push(`  axis ${k}:  ${x} vs ${y}  <span class="err">INCOMPATIBLE</span>`); break; }
      }

      const fmt = (s) => `(${s.join(', ')}${s.length === 1 ? ',' : ''})`;
      let head =
        `A  ${fmt(sa).padEnd(10)} pad -> ${fmt(pa)}\n` +
        `B  ${fmt(sb).padEnd(10)} pad -> ${fmt(pb)}\n` +
        `${'-'.repeat(34)}\n${lines.join('\n')}\n${'-'.repeat(34)}\n`;

      if (failed !== null) {
        head += `<span class="err">ValueError: operands could not be broadcast ` +
                `together with shapes ${fmt(sa)} ${fmt(sb)}</span>\n\n` +
                `This one is loud, which makes it the easy case.`;
        b.querySelector('#b-view').innerHTML = '';
      } else {
        const [m, n] = out.length === 2 ? out : [1, out[0]];
        const valA = (i, j) => (pa[0] === 1 ? 0 : i) * (pa[1] || 1) + (pa[1] === 1 ? 0 : j) + 1;
        const valB = (i, j) => ((pb[0] === 1 ? 0 : i) * (pb[1] || 1) + (pb[1] === 1 ? 0 : j) + 1) * 10;
        const A = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => valA(i, j)));
        const B = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => valB(i, j)));
        const C = A.map((row, i) => row.map((v, j) => v + B[i][j]));

        const ghostA = (i, j) => (pa[0] === 1 && i > 0) || (pa[1] === 1 && j > 0) ? 'ghost' : '';
        const ghostB = (i, j) => (pb[0] === 1 && i > 0) || (pb[1] === 1 && j > 0) ? 'ghost' : '';

        b.querySelector('#b-view').innerHTML =
          tensorTable(A, `A stretched to ${fmt(out)}`, ghostA) +
          `<div style="align-self:center;font-size:20px;color:var(--text-mute)">+</div>` +
          tensorTable(B, `B stretched to ${fmt(out)}`, ghostB) +
          `<div style="align-self:center;font-size:20px;color:var(--text-mute)">=</div>` +
          tensorTable(C, `result ${fmt(out)}`, () => 'hl2');

        const grew = out.reduce((p, v) => p * v, 1) >
                     Math.max(sa.reduce((p, v) => p * v, 1), sb.reduce((p, v) => p * v, 1));
        head += `<span class="ok">result shape ${fmt(out)}</span>` +
          (grew
            ? `\n\n<span class="err">Careful.</span> The result is LARGER than either input.\n` +
              `Faded cells are values being reused, not new data. If you expected\n` +
              `an elementwise add, this is the silent bug -- you have an outer\n` +
              `operation instead, and nothing will warn you.`
            : `\n\nFaded cells are the same value being reused, never copied in memory.`);
      }
      b.querySelector('#b-out').innerHTML = head;
    }

    b.querySelectorAll('select').forEach((s) => s.addEventListener('change', drawB));

    root.append(a, b);
    drawA();
    drawB();
  },
});

/* ============================================================ LESSON 06 ===
   Gradient descent on the real loss surface of the lesson's three points.
   Everything is computed live -- no precomputed frames. */

DEMOS.push({
  id: 6,
  title: 'Lesson 06 &mdash; Linear Regression from Scratch',
  render(root) {
    const XS = [1, 2, 3], YS = [2, 3, 5];
    const W_STAR = 1.5, B_STAR = 1 / 3, L_STAR = 1 / 18;
    const ETA_CRIT = 0.18027756377319946;   // 2 / lambda_max, from section 4.3

    const mse = (w, b) =>
      XS.reduce((s, x, i) => s + (w * x + b - YS[i]) ** 2, 0) / XS.length;

    const grads = (w, b) => {
      const m = XS.length;
      const e = XS.map((x, i) => w * x + b - YS[i]);
      return [
        (2 / m) * e.reduce((s, ei, i) => s + ei * XS[i], 0),
        (2 / m) * e.reduce((s, ei) => s + ei, 0),
      ];
    };

    const descend = (w, b, lr, steps) => {
      const path = [[w, b, mse(w, b)]];
      for (let k = 0; k < steps; k++) {
        const [gw, gb] = grads(w, b);
        w -= lr * gw; b -= lr * gb;                // both from the same old values
        if (!isFinite(w) || !isFinite(b) || Math.abs(w) > 1e6) {
          path.push([NaN, NaN, NaN]); break;
        }
        path.push([w, b, mse(w, b)]);
      }
      return path;
    };

    const el = h(`<div class="demo">
      <h3>Drive gradient descent yourself</h3>
      <p class="lede">The real loss surface for the three points in the lesson,
        computed live. The star is the exact optimum <code>w=1.5, b=0.333</code>.
        Push the learning rate past <b>0.1803</b> and watch the whole thing
        detonate &mdash; there is no gentle warning, which is the point.</p>
      <div class="ctl-row"><label for="lr">learning rate</label>
        <input type="range" id="lr" min="1" max="250" value="100"><output id="lr-o"></output></div>
      <div class="ctl-row"><label for="st">steps</label>
        <input type="range" id="st" min="1" max="80" value="25"><output id="st-o"></output></div>
      <div class="ctl-row"><label for="w0">start w</label>
        <input type="range" id="w0" min="-100" max="300" value="0"><output id="w0-o"></output></div>
      <div class="ctl-row"><label for="b0">start b</label>
        <input type="range" id="b0" min="-200" max="300" value="0"><output id="b0-o"></output></div>
      <div class="panels" style="gap:16px">
        <canvas id="surf" width="380" height="330"></canvas>
        <canvas id="fit"  width="330" height="330"></canvas>
      </div>
      <pre class="out" id="out"></pre>
    </div>`);

    const surf = el.querySelector('#surf'), fit = el.querySelector('#fit');
    const sc = surf.getContext('2d'), fc = fit.getContext('2d');
    const W_LO = -0.5, W_HI = 3.0, B_LO = -2.0, B_HI = 3.0;

    const sx = (w) => ((w - W_LO) / (W_HI - W_LO)) * surf.width;
    const sy = (b) => surf.height - ((b - B_LO) / (B_HI - B_LO)) * surf.height;

    function css(name) {
      return getComputedStyle(document.body).getPropertyValue(name).trim();
    }

    function drawSurface(path) {
      // loss as a heatmap, log-scaled so the valley floor stays visible
      const img = sc.createImageData(surf.width, surf.height);
      for (let py = 0; py < surf.height; py++) {
        for (let px = 0; px < surf.width; px++) {
          const w = W_LO + (px / surf.width) * (W_HI - W_LO);
          const b = B_HI - (py / surf.height) * (B_HI - B_LO);
          const t = Math.min(1, Math.log10(mse(w, b) / L_STAR + 1) / 3.2);
          const i = (py * surf.width + px) * 4;
          img.data[i] = 30 + t * 205;
          img.data[i + 1] = 60 + t * 140;
          img.data[i + 2] = 130 + t * 110;
          img.data[i + 3] = 46;
        }
      }
      sc.putImageData(img, 0, 0);

      // the optimum
      sc.fillStyle = '#f59e0b';
      sc.beginPath(); sc.arc(sx(W_STAR), sy(B_STAR), 6, 0, 7); sc.fill();
      sc.strokeStyle = '#000'; sc.lineWidth = 1; sc.stroke();

      // the descent path
      sc.strokeStyle = '#ef4444'; sc.lineWidth = 1.6;
      sc.beginPath();
      let started = false;
      for (const [w, b] of path) {
        if (!isFinite(w)) break;
        const X = sx(w), Y = sy(b);
        started ? sc.lineTo(X, Y) : (sc.moveTo(X, Y), started = true);
      }
      sc.stroke();
      for (const [w, b] of path) {
        if (!isFinite(w)) break;
        sc.fillStyle = '#ef4444';
        sc.beginPath(); sc.arc(sx(w), sy(b), 2.4, 0, 7); sc.fill();
      }
      // start marker
      if (isFinite(path[0][0])) {
        sc.fillStyle = '#16a34a';
        sc.beginPath(); sc.arc(sx(path[0][0]), sy(path[0][1]), 5, 0, 7); sc.fill();
      }
      sc.fillStyle = css('--text-mute'); sc.font = '11px system-ui';
      sc.fillText('w  (slope) →', 8, surf.height - 8);
      sc.save(); sc.translate(12, 18); sc.fillText('b  (intercept) →', 0, 0); sc.restore();
    }

    function drawFit(w, b) {
      fc.clearRect(0, 0, fit.width, fit.height);
      const X_LO = 0, X_HI = 4, Y_LO = -1, Y_HI = 7;
      const fx = (x) => ((x - X_LO) / (X_HI - X_LO)) * fit.width;
      const fy = (y) => fit.height - ((y - Y_LO) / (Y_HI - Y_LO)) * fit.height;

      fc.strokeStyle = css('--border'); fc.lineWidth = 1;
      for (let g = 0; g <= 4; g++) {
        fc.beginPath(); fc.moveTo(fx(g), 0); fc.lineTo(fx(g), fit.height); fc.stroke();
      }

      if (isFinite(w) && isFinite(b)) {
        // squared errors, drawn as squares -- the loss IS their mean area
        XS.forEach((x, i) => {
          const p = w * x + b, side = Math.abs(fy(YS[i]) - fy(p));
          fc.fillStyle = 'rgba(239,68,68,0.16)';
          fc.fillRect(fx(x), Math.min(fy(YS[i]), fy(p)), side, side);
          fc.strokeStyle = '#ef4444'; fc.setLineDash([3, 3]);
          fc.beginPath(); fc.moveTo(fx(x), fy(YS[i])); fc.lineTo(fx(x), fy(p)); fc.stroke();
          fc.setLineDash([]);
        });
        fc.strokeStyle = '#4f46e5'; fc.lineWidth = 2.4;
        fc.beginPath(); fc.moveTo(fx(X_LO), fy(w * X_LO + b));
        fc.lineTo(fx(X_HI), fy(w * X_HI + b)); fc.stroke();
      } else {
        fc.fillStyle = '#ef4444'; fc.font = 'bold 15px system-ui';
        fc.fillText('diverged — no line to draw', 40, fit.height / 2);
      }

      fc.fillStyle = '#ef4444';
      XS.forEach((x, i) => {
        fc.beginPath(); fc.arc(fx(x), fy(YS[i]), 5.5, 0, 7); fc.fill();
      });
      fc.fillStyle = css('--text-mute'); fc.font = '11px system-ui';
      fc.fillText('the data and the current line', 8, 16);
    }

    function draw() {
      const lr = +el.querySelector('#lr').value / 1000;     // 0.001 .. 0.250
      const steps = +el.querySelector('#st').value;
      const w0 = +el.querySelector('#w0').value / 100;
      const b0 = +el.querySelector('#b0').value / 100;

      el.querySelector('#lr-o').textContent = lr.toFixed(3);
      el.querySelector('#st-o').textContent = steps;
      el.querySelector('#w0-o').textContent = w0.toFixed(2);
      el.querySelector('#b0-o').textContent = b0.toFixed(2);

      const path = descend(w0, b0, lr, steps);
      const last = path[path.length - 1];
      const blew = !isFinite(last[0]);

      drawSurface(path);
      drawFit(last[0], last[1]);

      const [gw0, gb0] = grads(w0, b0);
      let msg =
        `start      w = ${w0.toFixed(4)}   b = ${b0.toFixed(4)}   loss = ${mse(w0, b0).toFixed(6)}\n` +
        `gradient   dL/dw = ${gw0.toFixed(4)}   dL/db = ${gb0.toFixed(4)}\n` +
        `after ${String(steps).padStart(2)}   `;

      if (blew) {
        msg += `<span class="err">DIVERGED after ${path.length - 1} steps</span>\n\n` +
          `<span class="err">eta = ${lr.toFixed(3)} exceeds the threshold ${ETA_CRIT.toFixed(4)}.</span>\n` +
          `Each step overshoots further than the last, so the error grows\n` +
          `geometrically. In a real network this is the run that prints a\n` +
          `perfectly normal loss and then, one step later, nan.`;
      } else {
        const gap = Math.hypot(last[0] - W_STAR, last[1] - B_STAR);
        msg += `w = ${last[0].toFixed(6)}   b = ${last[1].toFixed(6)}   loss = ${last[2].toFixed(6)}\n` +
          `optimum    w = ${W_STAR.toFixed(6)}   b = ${B_STAR.toFixed(6)}   loss = ${L_STAR.toFixed(6)}\n` +
          `distance to optimum: ${gap.toExponential(2)}   ` +
          (gap < 1e-3 ? '<span class="ok">converged</span>'
                      : lr < 0.02 ? 'still crawling &mdash; try a larger eta'
                                  : 'getting there');
        if (lr > ETA_CRIT * 0.9 && lr < ETA_CRIT) {
          msg += `\n\n<span class="err">Note</span> eta = ${lr.toFixed(3)} is within 10% of the ` +
            `threshold ${ETA_CRIT.toFixed(4)}.\nIt still converges, but look at the zig-zag: ` +
            `every step overshoots\nthe valley floor and has to come back.`;
        }
      }
      el.querySelector('#out').innerHTML = msg;
    }

    el.querySelectorAll('input[type=range]').forEach((r) => r.addEventListener('input', draw));
    root.append(el);
    draw();
  },
});

/* ============================================================ LESSON 11 ===
   Live backpropagation through the 2-2-1 network from the lesson. Every
   number is computed here in JS from the same formulas the notebook derives;
   the defaults reproduce the hand-worked numbers in section 3 exactly. */

DEMOS.push({
  id: 11,
  title: 'Lesson 11 &mdash; Backpropagation, Derived in Full',
  render(root) {

    /* ---------------------------------------------------- the network */
    const DEFAULTS = {
      w100: 0.5, w101: -1.0, w110: -0.5, w111: 0.5,
      b10: 0.0, b11: 1.0, w20: 1.0, w21: 2.0, b2: -0.5,
      x0: 1.0, x1: 2.0, y: 1,
    };
    let p = { ...DEFAULTS };

    const tanh = Math.tanh;
    const sig = (z) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));

    /** Forward and backward, exactly as derived in sections 2.3 and 2.4. */
    function compute(q) {
      const x = [q.x0, q.x1];
      const W1 = [[q.w100, q.w101], [q.w110, q.w111]];
      const b1 = [q.b10, q.b11];
      const W2 = [q.w20, q.w21];

      // forward
      const z1 = [0, 1].map((j) => x[0] * W1[0][j] + x[1] * W1[1][j] + b1[j]);
      const h = z1.map(tanh);
      const z2 = h[0] * W2[0] + h[1] * W2[1] + q.b2;
      const yhat = sig(z2);
      const yc = Math.min(Math.max(yhat, 1e-15), 1 - 1e-15);
      const loss = -(q.y * Math.log(yc) + (1 - q.y) * Math.log(1 - yc));

      // backward
      const delta2 = yhat - q.y;                       // the L08 cancellation
      const dW2 = h.map((hj) => delta2 * hj);          // delta x activation
      const db2 = delta2;
      const tp = h.map((hj) => 1 - hj * hj);           // tanh' = 1 - h^2
      const delta1 = [0, 1].map((j) => delta2 * W2[j] * tp[j]);
      const dW1 = [[delta1[0] * x[0], delta1[1] * x[0]],
                   [delta1[0] * x[1], delta1[1] * x[1]]];
      const db1 = delta1.slice();

      return { x, z1, h, z2, yhat, loss, delta2, dW2, db2, tp, delta1, dW1, db1 };
    }

    /** Numerical gradient of the loss w.r.t. one parameter -- the honesty check. */
    function numericGrad(key, eps = 1e-5) {
      const hi = { ...p, [key]: p[key] + eps };
      const lo = { ...p, [key]: p[key] - eps };
      return (compute(hi).loss - compute(lo).loss) / (2 * eps);
    }

    const GRAD_OF = {
      w100: (s) => s.dW1[0][0], w101: (s) => s.dW1[0][1],
      w110: (s) => s.dW1[1][0], w111: (s) => s.dW1[1][1],
      b10: (s) => s.db1[0], b11: (s) => s.db1[1],
      w20: (s) => s.dW2[0], w21: (s) => s.dW2[1], b2: (s) => s.db2,
    };
    const LABEL = {
      w100: 'W1[0][0]', w101: 'W1[0][1]', w110: 'W1[1][0]', w111: 'W1[1][1]',
      b10: 'b1[0]', b11: 'b1[1]', w20: 'W2[0]', w21: 'W2[1]', b2: 'b2',
    };

    /* ---------------------------------------------------- the markup */
    const slider = (key, label, min, max) => `
      <div class="ctl-row" style="margin:3px 0">
        <label for="${key}" style="min-width:74px;font-family:var(--mono);font-size:12px">${label}</label>
        <input type="range" id="${key}" min="${min * 100}" max="${max * 100}"
               step="5" value="${DEFAULTS[key] * 100}">
        <output id="${key}-o" style="min-width:52px"></output>
      </div>`;

    const el = h(`<div class="demo">
      <h3>Watch the gradient flow backwards</h3>
      <p class="lede">The exact 2&ndash;2&ndash;1 network from the lesson. Blue numbers
        flow forward, red numbers flow back. Every gradient is computed live from
        the formulas in section 2 &mdash; and checked against a numerical
        derivative, so you can see for yourself that the derivation is right.
        The defaults reproduce the hand-worked numbers in section 3.</p>

      <div class="panels" style="gap:26px;align-items:flex-start">
        <div style="flex:1 1 320px;min-width:290px">
          <div style="font-size:12px;font-weight:600;color:var(--text-dim);margin-bottom:4px">input and target</div>
          ${slider('x0', 'x₁', -3, 3)}
          ${slider('x1', 'x₂', -3, 3)}
          <div class="ctl-row" style="margin:3px 0">
            <label style="min-width:74px;font-family:var(--mono);font-size:12px">y</label>
            <button class="chip" id="y-toggle" style="flex:0 0 auto">target = 1</button>
          </div>

          <div style="font-size:12px;font-weight:600;color:var(--text-dim);margin:12px 0 4px">hidden layer</div>
          ${slider('w100', 'W1[0][0]', -3, 3)}
          ${slider('w101', 'W1[0][1]', -3, 3)}
          ${slider('w110', 'W1[1][0]', -3, 3)}
          ${slider('w111', 'W1[1][1]', -3, 3)}
          ${slider('b10', 'b1[0]', -3, 3)}
          ${slider('b11', 'b1[1]', -3, 3)}

          <div style="font-size:12px;font-weight:600;color:var(--text-dim);margin:12px 0 4px">output layer</div>
          ${slider('w20', 'W2[0]', -3, 3)}
          ${slider('w21', 'W2[1]', -3, 3)}
          ${slider('b2', 'b2', -3, 3)}

          <div class="ctl-row" style="margin-top:14px;gap:6px">
            <button class="btn primary" id="step">Take one step</button>
            <button class="btn" id="run">Train 200</button>
            <button class="btn" id="reset">Reset</button>
          </div>
          <div class="ctl-row" style="margin:3px 0">
            <label for="lr" style="min-width:74px;font-size:12px">learning rate</label>
            <input type="range" id="lr" min="1" max="300" value="100">
            <output id="lr-o" style="min-width:52px"></output>
          </div>
        </div>

        <div style="flex:1 1 380px;min-width:330px">
          <svg id="net" viewBox="0 0 430 250" style="width:100%;height:auto"></svg>
        </div>
      </div>

      <pre class="out" id="chain"></pre>
      <pre class="out" id="grads"></pre>
    </div>`);

    /* ---------------------------------------------------- the diagram */
    const NODES = {
      x0: [40, 55], x1: [40, 160], h0: [190, 55], h1: [190, 160], out: [345, 107],
    };

    function drawNet(s) {
      const edge = (a, b, w, label, gradLabel) => {
        const [x1_, y1_] = NODES[a], [x2_, y2_] = NODES[b];
        const mx = x1_ + (x2_ - x1_) * 0.5, my = y1_ + (y2_ - y1_) * 0.5;
        const width = Math.min(6, 0.6 + Math.abs(w) * 1.5);
        return `<line x1="${x1_}" y1="${y1_}" x2="${x2_}" y2="${y2_}"
                  stroke="${w >= 0 ? '#60a5fa' : '#fca5a5'}" stroke-width="${width}"
                  opacity="0.85"/>
                <text x="${mx}" y="${my - 5}" text-anchor="middle" font-size="10"
                  font-family="ui-monospace,monospace" fill="currentColor">${label}</text>
                <text x="${mx}" y="${my + 9}" text-anchor="middle" font-size="9"
                  font-family="ui-monospace,monospace" fill="#ef4444">${gradLabel}</text>`;
      };

      const node = (k, main, sub, colour) => {
        const [cx, cy] = NODES[k];
        return `<circle cx="${cx}" cy="${cy}" r="24" fill="${colour}" opacity="0.9"/>
                <text x="${cx}" y="${cy + 4}" text-anchor="middle" font-size="11"
                  font-weight="700" fill="#fff" font-family="ui-monospace,monospace">${main}</text>
                <text x="${cx}" y="${cy + 39}" text-anchor="middle" font-size="9.5"
                  fill="#ef4444" font-family="ui-monospace,monospace">${sub}</text>`;
      };

      const f = (v, n = 3) => v.toFixed(n);
      el.querySelector('#net').innerHTML =
        edge('x0', 'h0', p.w100, f(p.w100, 2), f(s.dW1[0][0])) +
        edge('x0', 'h1', p.w101, f(p.w101, 2), f(s.dW1[0][1])) +
        edge('x1', 'h0', p.w110, f(p.w110, 2), f(s.dW1[1][0])) +
        edge('x1', 'h1', p.w111, f(p.w111, 2), f(s.dW1[1][1])) +
        edge('h0', 'out', p.w20, f(p.w20, 2), f(s.dW2[0])) +
        edge('h1', 'out', p.w21, f(p.w21, 2), f(s.dW2[1])) +
        node('x0', f(p.x0, 1), '', '#0ea5e9') +
        node('x1', f(p.x1, 1), '', '#0ea5e9') +
        node('h0', f(s.h[0]), 'δ₁=' + f(s.delta1[0]), '#16a34a') +
        node('h1', f(s.h[1]), 'δ₁=' + f(s.delta1[1]), '#16a34a') +
        node('out', f(s.yhat), 'δ₂=' + f(s.delta2), '#a855f7') +
        `<text x="40" y="20" text-anchor="middle" font-size="10" fill="currentColor">input</text>
         <text x="190" y="20" text-anchor="middle" font-size="10" fill="currentColor">tanh</text>
         <text x="345" y="20" text-anchor="middle" font-size="10" fill="currentColor">sigmoid</text>
         <text x="345" y="200" text-anchor="middle" font-size="10" fill="currentColor">y = ${p.y}</text>
         <text x="60" y="243" font-size="9.5" fill="#60a5fa">blue = weight</text>
         <text x="200" y="243" font-size="9.5" fill="#ef4444">red = gradient</text>`;
    }

    /* ---------------------------------------------------- the readouts */
    function render() {
      const s = compute(p);

      for (const k of Object.keys(DEFAULTS)) {
        if (k === 'y') continue;
        const inp = el.querySelector('#' + k);
        if (inp) {
          inp.value = p[k] * 100;
          el.querySelector('#' + k + '-o').textContent = p[k].toFixed(2);
        }
      }
      const lr = +el.querySelector('#lr').value / 100;
      el.querySelector('#lr-o').textContent = lr.toFixed(2);
      el.querySelector('#y-toggle').textContent = `target = ${p.y}`;

      drawNet(s);

      const f = (v, n = 7) => (v >= 0 ? ' ' : '') + v.toFixed(n);
      el.querySelector('#chain').innerHTML =
`<b>FORWARD</b>
  z1 = [${f(s.z1[0])}, ${f(s.z1[1])}]        h = tanh(z1) = [${f(s.h[0])}, ${f(s.h[1])}]
  z2 = ${f(s.z2)}                 yhat = sigma(z2) = ${f(s.yhat)}
  loss = ${f(s.loss)}   ${s.loss < Math.LN2 ? '<span class="ok">better than log 2</span>' : '<span class="err">worse than knowing nothing (log 2 = 0.693147)</span>'}

<b>BACKWARD</b>  &mdash; section 2.4, one hop at a time
  delta2 = yhat - y                      = ${f(s.delta2)}
  tanh'  = 1 - h^2                       = [${f(s.tp[0])}, ${f(s.tp[1])}]

  unit 1:  delta2 x W2[0] x tanh'[0]
           ${f(s.delta2)} x ${f(p.w20, 2)} x ${f(s.tp[0])} = ${f(s.delta1[0])}
  unit 2:  delta2 x W2[1] x tanh'[1]
           ${f(s.delta2)} x ${f(p.w21, 2)} x ${f(s.tp[1])} = ${f(s.delta1[1])}`;

      let rows = '';
      let worst = 0;
      for (const k of Object.keys(GRAD_OF)) {
        const a = GRAD_OF[k](s), n = numericGrad(k);
        const rel = Math.abs(a - n) / Math.max(1e-300, Math.abs(a) + Math.abs(n));
        const ok = rel < 1e-5 || Math.abs(a - n) < 1e-8;
        if (!ok) worst = Math.max(worst, rel);
        rows += `  ${LABEL[k].padEnd(9)} ${f(a).padStart(13)} ${f(n).padStart(15)}` +
                `   ${ok ? '<span class="ok">ok</span>' : '<span class="err">FAIL</span>'}\n`;
      }

      // the structural check from section 3.5
      const ratio = p.x0 !== 0 ? (p.x1 / p.x0) : NaN;
      const shown = Number.isFinite(ratio)
        ? `  dW1 row 2 / row 1 = ${ratio.toFixed(4)} = x2/x1, because dW1[i][j] = delta1[j] * x[i]`
        : '  (x1 = 0, so row 1 of dW1 is all zeros -- a zero input learns no weight)';

      el.querySelector('#grads').innerHTML =
`<b>ALL NINE GRADIENTS</b>, each checked against a numerical derivative
  ${'parameter'.padEnd(9)} ${'backprop'.padStart(13)} ${'numerical'.padStart(15)}
  ${'-'.repeat(52)}
${rows}${worst === 0 ? '  <span class="ok">every gradient confirmed</span>' : '  <span class="err">mismatch ' + worst.toExponential(2) + '</span>'}

${shown}`;
    }

    /* ---------------------------------------------------- interaction */
    function gdStep(lr) {
      const s = compute(p);
      for (const k of Object.keys(GRAD_OF)) p[k] -= lr * GRAD_OF[k](s);
    }

    el.querySelectorAll('input[type=range]').forEach((r) => {
      r.addEventListener('input', () => {
        if (r.id !== 'lr') p[r.id] = +r.value / 100;
        render();
      });
    });
    el.querySelector('#y-toggle').addEventListener('click', () => {
      p.y = p.y === 1 ? 0 : 1;
      render();
    });
    el.querySelector('#step').addEventListener('click', () => {
      gdStep(+el.querySelector('#lr').value / 100);
      render();
    });
    el.querySelector('#run').addEventListener('click', () => {
      const lr = +el.querySelector('#lr').value / 100;
      for (let i = 0; i < 200; i++) gdStep(lr);
      render();
    });
    el.querySelector('#reset').addEventListener('click', () => {
      p = { ...DEFAULTS };
      el.querySelector('#lr').value = 100;
      render();
    });

    root.append(el);
    render();
  },
});

/* ------------------------------------------------------------------ boot */
function boot() {
  const host = document.getElementById('demos');
  const chips = document.getElementById('lesson-chips');

  if (!DEMOS.length) {
    host.innerHTML = '<div class="empty">No demos yet.</div>';
    return;
  }

  const wanted = Number(new URLSearchParams(location.search).get('lesson'));
  const active = DEMOS.find((d) => d.id === wanted) || DEMOS[0];

  DEMOS.forEach((d) => {
    const c = h(`<button class="chip" aria-pressed="${d === active}">${String(d.id).padStart(2, '0')}</button>`);
    c.addEventListener('click', () => { location.search = `?lesson=${d.id}`; });
    chips.appendChild(c);
  });

  host.innerHTML = `<h2 style="font-size:18px;margin:8px 0 0">${active.title}</h2>`;
  active.render(host);
}

boot();
