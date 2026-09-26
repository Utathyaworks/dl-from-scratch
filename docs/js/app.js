/* Deep Learning From Scratch -- curriculum index.
   Reads data/progress.json, which tools/track.py regenerates on every
   status change, so the site can never drift from the repo. */

const DATA_URL = 'data/progress.json';
const state = { data: null, filter: 'all', query: '' };

const el = (id) => document.getElementById(id);

/* Where a finished notebook can be opened. Kaggle imports straight from a
   GitHub blob URL, which is why the notebooks are committed executed. */
function urls(d, lesson) {
  const blob = `https://github.com/${d.github_user}/${d.repo}/blob/main/${lesson.notebook}`;
  return {
    github: blob,
    kaggle: `https://kaggle.com/kernels/welcome?src=${encodeURIComponent(blob)}`,
    colab: `https://colab.research.google.com/github/${d.github_user}/${d.repo}/blob/main/${lesson.notebook}`,
    nbviewer: `https://nbviewer.org/github/${d.github_user}/${d.repo}/blob/main/${lesson.notebook}`,
  };
}

function matches(lesson) {
  if (state.filter !== 'all' && lesson.status !== state.filter) return false;
  if (!state.query) return true;
  const hay = `${lesson.id} ${lesson.title} ${lesson.goal} ${lesson.slug}`.toLowerCase();
  return hay.includes(state.query);
}

function card(d, lesson) {
  const node = document.createElement('article');
  node.className = `card ${lesson.status}`;

  const label = { done: 'done', 'in-progress': 'in progress', todo: 'to do' }[lesson.status];
  const top = `
    <div class="top">
      <span class="num">${String(lesson.id).padStart(2, '0')}</span>
      <h3>${lesson.title}</h3>
      <span class="badge ${lesson.status}">${label}</span>
    </div>
    <p class="goal">${lesson.goal}</p>`;

  let actions = '';
  if (lesson.status === 'done') {
    const u = urls(d, lesson);
    actions = `<div class="actions">
      <a class="btn primary" href="${u.kaggle}" target="_blank" rel="noopener">Run on Kaggle</a>
      <a class="btn" href="${u.colab}" target="_blank" rel="noopener">Colab</a>
      <a class="btn" href="${u.nbviewer}" target="_blank" rel="noopener">Read</a>
      <a class="btn" href="${u.github}" target="_blank" rel="noopener">Source</a>
      ${DEMO_LESSONS.has(lesson.id)
        ? `<a class="btn play" href="playground.html?lesson=${lesson.id}">Play</a>` : ''}
    </div>`;
  } else if (lesson.status === 'in-progress') {
    actions = `<div class="actions"><span class="btn" style="cursor:default">being written</span></div>`;
  }

  node.innerHTML = top + actions;
  return node;
}

/* Lessons that have an interactive demo on playground.html. */
const DEMO_LESSONS = new Set([1, 6, 11]);

function render() {
  const d = state.data;
  const host = el('phases');
  host.textContent = '';
  let shown = 0;

  for (const phase of d.phases) {
    const lessons = d.lessons.filter((l) => l.phase === phase.id);
    const visible = lessons.filter(matches);
    if (!visible.length) continue;
    shown += visible.length;

    const section = document.createElement('section');
    section.className = 'phase';
    const done = lessons.filter((l) => l.status === 'done').length;
    section.innerHTML = `<h2>
        <span class="dot" style="background:${phase.color}"></span>
        Phase ${phase.id} &middot; ${phase.name}
        <span class="count">${done}/${lessons.length} done</span>
      </h2>`;

    const grid = document.createElement('div');
    grid.className = 'grid';
    visible.forEach((l) => grid.appendChild(card(d, l)));
    section.appendChild(grid);
    host.appendChild(section);
  }

  el('empty').hidden = shown > 0;
}

function renderStats() {
  const ls = state.data.lessons;
  const total = ls.length;
  const done = ls.filter((l) => l.status === 'done').length;
  const prog = ls.filter((l) => l.status === 'in-progress').length;
  const pct = Math.round((done / total) * 100);

  el('s-done').textContent = done;
  el('s-prog').textContent = prog;
  el('s-left').textContent = total - done - prog;
  el('s-pct').textContent = `${pct}%`;
  el('bar-done').style.width = `${(done / total) * 100}%`;
  el('bar-prog').style.width = `${(prog / total) * 100}%`;

  el('repo-link').href = `https://github.com/${state.data.github_user}/${state.data.repo}`;
}

function wireControls() {
  el('q').addEventListener('input', (e) => {
    state.query = e.target.value.trim().toLowerCase();
    render();
  });
  document.querySelectorAll('.chip[data-filter]').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.chip[data-filter]').forEach((c) =>
        c.setAttribute('aria-pressed', String(c === chip)));
      state.filter = chip.dataset.filter;
      render();
    });
  });
}

fetch(DATA_URL)
  .then((r) => {
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  })
  .then((data) => {
    state.data = data;
    renderStats();
    wireControls();
    render();
  })
  .catch((err) => {
    el('phases').innerHTML =
      `<div class="empty">Could not load <code>${DATA_URL}</code> (${err.message}).<br>
       If you opened this file directly, serve it instead:
       <code>python -m http.server -d docs 8000</code></div>`;
  });
