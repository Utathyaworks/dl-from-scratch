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
