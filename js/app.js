/* The figures. The model itself lives in engine.js; this file only asks it questions and draws the answers. */
(() => {
  'use strict';
  LANG.init(LEXICON);
  const M = TF.load(MODEL_PACK);
  const HEADS = ['A', 'B'], H = M.cfg.h;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---------- small helpers ---------- */
  const $ = (s, r = document) => r.querySelector(s);
  const h = (tag, attrs = {}, ...kids) => {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else if (k.startsWith('aria-')) e.setAttribute(k, String(v));
      else if (v !== false && v != null) e.setAttribute(k, v === true ? '' : v);
    }
    for (const k of kids.flat()) if (k != null) e.append(k);
    return e;
  };
  const svg = (tag, attrs = {}) => { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v); return e; };
  const pct = v => Math.round(v * 100) + '%';
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const INK = [css('--head-a'), css('--head-b')];
  const rnd = LANG.rng((Date.now() ^ (Math.random() * 1e9)) | 0);
  const onResize = (node, fn) => { if ('ResizeObserver' in window) new ResizeObserver(fn).observe(node); else addEventListener('resize', fn); };

  // Which words of a wrong translation are wrong (longest common subsequence against the rulebook answer)
  function diff(hyp, ref) {
    const n = hyp.length, m = ref.length, T = Array.from({ length: n + 1 }, () => new Uint8Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) T[i][j] = hyp[i] === ref[j] ? T[i + 1][j + 1] + 1 : Math.max(T[i + 1][j], T[i][j + 1]);
    const ok = Array(n).fill(false); let i = 0, j = 0;
    while (i < n && j < m) { if (hyp[i] === ref[j]) { ok[i] = true; i++; j++; } else if (T[i + 1][j] >= T[i][j + 1]) i++; else j++; }
    return ok;
  }
  const marked = (hyp, ref) => { const ok = diff(hyp, ref); return hyp.map((w, i) => [h('span', { class: ok[i] ? '' : 'bad', text: w }), ' ']).flat(); };
  const same = (a, b) => a.length === b.length && a.every((w, i) => w === b[i]);
  const list = a => a.length < 2 ? a.join('') : a.slice(0, -1).join(', ') + ' and ' + a[a.length - 1];

  // Bars above and below a zero line: how a short list of numbers is drawn throughout
  function spark(vals, scale, fill) {
    const bw = 7, gap = 2, hh = 30, s = svg('svg', { class: 'spark', width: vals.length * (bw + gap), height: hh, viewBox: `0 0 ${vals.length * (bw + gap)} ${hh}`, 'aria-hidden': 'true' });
    s.append(svg('line', { x1: 0, x2: vals.length * (bw + gap), y1: hh / 2, y2: hh / 2, stroke: css('--rule'), 'stroke-width': 1 }));
    vals.forEach((v, i) => {
      const len = Math.max(1, Math.min(1, Math.abs(v) / scale) * (hh / 2 - 1)), f = fill(i);
      s.append(svg('rect', { x: i * (bw + gap) + .5, y: v >= 0 ? hh / 2 - len : hh / 2, width: bw - 1, height: len, fill: f || 'none', stroke: f ? 'none' : css('--ink-2'), 'stroke-width': 1 }));
    });
    return s;
  }

  /* ---------- the sentence every figure shares ---------- */
  const store = { words: null, R: null, ref: null, subs: [] };
  function setSentence(words) { store.words = words; store.R = TF.translate(M, words); store.ref = LANG.reference(words).out; store.subs.forEach(f => f()); }
  const kinds = ['pre', 'post'];
  function parse(words) {
    let i = 0;
    const np = () => { const o = { det: words[i++], pre: '', post: '' }; while (kinds.includes(LANG.kind(words[i]))) { o[LANG.kind(words[i])] = words[i]; i++; } o.noun = words[i++]; return o; };
    const s = { np1: np(), v1: words[i++], np2: np(), it: null };
    if (words[i] === 'and') { i += 2; s.it = { v: words[i++], np: np() }; }
    return s;
  }
  function flat(s) {
    const np = o => [o.det, o.pre, o.post, o.noun].filter(Boolean), w = [...np(s.np1), s.v1, ...np(s.np2)];
    if (s.it) w.push('and', 'it', s.it.v, ...np(s.it.np));
    return [...w, '.'];
  }

  /* ---------- attention view: two columns of words with weighted lines between ---------- */
  function AttnView(root, opts = {}) {
    const leftCol = h('ol', { class: 'av-col av-left' }), rightCol = h('ol', { class: 'av-col av-right' }), lines = svg('svg', { class: 'av-lines', 'aria-hidden': 'true' });
    root.classList.add('av');
    root.append(h('div', { class: 'av-head' }, h('span', { text: opts.leftLabel || '' }), h('span'), h('span', { text: opts.rightLabel || '' })), h('div', { class: 'av-body' }, leftCol, lines, rightCol));
    const st = { left: [], right: [], heads: [], shown: Infinity, emph: null, hover: null, lock: null, on: Array(H).fill(true), pending: false };
    const ROW = parseFloat(css('--row')) || 27, PAD = 5, y = (i, hh) => i * ROW + ROW / 2 + (hh - (H - 1) / 2) * 3.4;
    const focus = () => st.lock || st.hover;

    function draw() {
      const W = lines.clientWidth || 120, n = Math.max(st.left.length, st.right.length), f = focus();
      lines.setAttribute('viewBox', `0 0 ${W} ${n * ROW}`); lines.style.height = n * ROW + 'px';
      const frag = document.createDocumentFragment(), heat = Array(st.left.length).fill(0);
      st.heads.forEach((hd, hh) => {
        if (!st.on[hh]) return;
        hd.forEach((row, r) => {
          if (!row || r >= st.shown) return;
          row.forEach((w, j) => {
            if (f && f.side === 'right' && f.i === r) heat[j] = Math.max(heat[j], w);
            if (w < 0.04) return;
            let k = 1;
            if (f) k = (f.side === 'right' ? f.i === r : f.i === j) ? 1 : 0.04;
            else if (st.emph != null) k = r === st.emph ? 1 : 0.2;
            frag.append(svg('path', { class: 'ln h' + hh, d: `M${PAD},${y(j, hh)} C${W * .5},${y(j, hh)} ${W * .5},${y(r, hh)} ${W - PAD},${y(r, hh)}`, 'stroke-width': (0.6 + 2.6 * w).toFixed(2), 'stroke-opacity': ((0.14 + 0.86 * w) * k).toFixed(3) }));
          });
        });
      });
      lines.replaceChildren(frag);
      [...rightCol.children].forEach((li, r) => {
        const pend = st.pending && r === st.shown;
        li.classList.toggle('hidden', r >= st.shown && !pend); li.classList.toggle('pending', pend);
        li.firstChild.textContent = pend ? '?' : st.right[r];
        li.classList.toggle('on', f ? f.side === 'right' && f.i === r : st.emph === r);
      });
      [...leftCol.children].forEach((li, j) => { li.classList.toggle('on', !!f && f.side === 'left' && f.i === j); li.firstChild.style.setProperty('--w', heat[j].toFixed(2)); });
    }
    function tell() { if (opts.onFocus) opts.onFocus(focus()); }
    function word(text, side, i) {
      const b = h('button', { type: 'button', class: 'tok', text });
      b.addEventListener('pointerenter', e => { if (e.pointerType === 'mouse' && !opts.pick) { st.hover = { side, i }; draw(); tell(); } });
      b.addEventListener('pointerleave', () => { if (st.hover) { st.hover = null; draw(); tell(); } });
      b.addEventListener('focus', () => { if (!opts.pick && b.matches(':focus-visible')) { st.hover = { side, i }; draw(); tell(); } });
      b.addEventListener('blur', () => { if (st.hover) { st.hover = null; draw(); tell(); } });
      b.addEventListener('click', () => {
        if (opts.pick) { if (side === 'left') opts.pick(i); return; }
        st.lock = st.lock && st.lock.side === side && st.lock.i === i ? null : { side, i }; st.hover = null; draw(); tell();
      });
      return h('li', {}, b);
    }
    const api = {
      set({ left, right, heads, startLeft = false }) {
        st.left = left; st.right = right; st.heads = heads; st.lock = st.hover = null; st.shown = Infinity; st.emph = null;
        leftCol.replaceChildren(...left.map((w, i) => word(w, 'left', i))); rightCol.replaceChildren(...right.map((w, i) => word(w, 'right', i)));
        if (startLeft) leftCol.firstChild.classList.add('start');
        draw(); return api;
      },
      show(shown, emph = null, pending = false) { st.shown = shown; st.emph = emph; st.pending = pending; draw(); return api; },
      lock(side, i) { st.lock = i == null ? null : { side, i }; draw(); tell(); return api; },
      heads: st.on, redraw: draw, focus, root,
    };
    if (opts.pick) root.classList.add('pick');
    onResize(lines, draw);
    return api;
  }
  // "head A / head B" switches for a view
  function legend(box, view, after) {
    HEADS.forEach((name, hh) => box.append(h('button', {
      type: 'button', class: 'hd ' + 'ab'[hh], 'aria-pressed': 'true', title: 'Show or hide this head',
      onclick: e => { view.heads[hh] = !view.heads[hh]; e.currentTarget.setAttribute('aria-pressed', view.heads[hh]); view.redraw(); if (after) after(); },
    }, h('i'), 'head ' + name)));
  }
  const crossHeads = R => HEADS.map((_, hh) => R.steps.map(s => s.cross[hh]));

  /* ---------- Figure 1: the model at work ---------- */
  (() => {
    const readout = $('#live-readout'), box = $('#builder');
    let timer = null, shown = 0;
    const view = AttnView($('#live-attn'), { leftLabel: 'English in', rightLabel: 'French out', onFocus: renderReadout });
    legend($('#live-legend'), view);

    function select(label, value, options, blank, change) {
      const s = h('select', { 'aria-label': label, class: value ? '' : 'empty', onchange: e => change(e.target.value) });
      if (blank) s.append(h('option', { value: '', text: blank }));
      for (const o of options) s.append(h('option', { value: o, text: o }));
      s.value = value; return s;
    }
    function renderBuilder() {
      const s = parse(store.words), go = () => setSentence(flat(s));
      const np = (o, name) => [
        select(name + ': article', o.det, Object.keys(LEXICON.dets), null, v => { o.det = v; go(); }),
        select(name + ': adjective', o.pre, Object.keys(LEXICON.pre), '+ adjective', v => { o.pre = v; go(); }),
        select(name + ': colour', o.post, Object.keys(LEXICON.post), '+ colour', v => { o.post = v; go(); }),
        select(name + ': noun', o.noun, Object.keys(LEXICON.nouns), null, v => { o.noun = v; go(); }),
      ];
      const verbs = Object.keys(LEXICON.verbs), kids = [...np(s.np1, 'Subject'), select('Verb', s.v1, verbs, null, v => { s.v1 = v; go(); }), ...np(s.np2, 'Object')];
      if (s.it) kids.push(h('span', { class: 'fixed', text: 'and it' }), select('Second verb', s.it.v, verbs, null, v => { s.it.v = v; go(); }), ...np(s.it.np, 'Second object'),
        h('button', { type: 'button', class: 'btn', 'aria-label': 'Remove the second clause', text: '×', onclick: () => { s.it = null; go(); } }));
      else kids.push(h('button', { type: 'button', class: 'btn', text: '+ and it …', onclick: () => { s.it = { v: 'eats', np: { det: 'a', pre: '', post: 'red', noun: 'pear' } }; go(); } }));
      kids.push(h('span', { class: 'fixed', text: '.' }));
      box.replaceChildren(...kids);
    }
    function renderReadout() {
      const R = store.R, n = R.out.length, f = view.focus();
      const k = f && f.side === 'right' && f.i < shown ? f.i : Math.min(shown, n) - 1, kids = [];
      kids.push(h('div', {}, h('h3', { text: 'Written so far' }), h('p', { class: 'fr', text: R.out.slice(0, shown).join(' ') || '…' })));
      if (k >= 0) {
        const top = R.steps[k].probs.map((p, v) => [p, M.tgt[v]]).sort((a, b) => b[0] - a[0]).slice(0, 3), g = h('div', { class: 'odds' });
        for (const [p, w] of top) g.append(h('span', { text: w }), h('span', { class: 'bar', style: `--v:${p}` }, h('i')), h('span', { class: 'num', text: pct(p) }));
        kids.push(h('div', {}, h('h3', { text: `Odds for word ${k + 1} of ${n}` }), g));
      }
      if (shown >= n) kids.push(h('div', {}, h('h3', { text: 'Rulebook check' }), h('p', { text: same(R.out, store.ref) ? 'Matches the rulebook translation.' : 'Differs from the rulebook: ' + store.ref.join(' ') })));
      readout.replaceChildren(...kids);
    }
    function play(animate) {
      clearInterval(timer); const R = store.R, n = R.out.length;
      view.set({ left: store.words, right: R.out, heads: crossHeads(R) });
      if (!animate || reduced) { shown = n; view.show(n); renderReadout(); return; }
      shown = 0; view.show(0); renderReadout();
      timer = setInterval(() => {
        shown++; view.show(shown, shown - 1); renderReadout();
        if (shown >= n) { clearInterval(timer); timer = setTimeout(() => { view.show(n); }, 1100); }
      }, 300);
    }
    let first = true;
    store.subs.push(() => { renderBuilder(); play(!first); first = false; });
    $('#live-replay').addEventListener('click', () => play(true));
    $('#live-shuffle').addEventListener('click', () => setSentence(LANG.sentence(rnd, 0.45, 0.6, 0.3)));
    // Rest state first (complete without motion), then one run so the page opens with the model writing.
    setTimeout(() => { if (!reduced && document.visibilityState === 'visible' && scrollY < 400) play(true); }, 900);
  })();

  /* ---------- Figure 2: path length ---------- */
  (() => {
    const WORDS = ['the', 'cow', 'sees', 'my', 'hat', 'and', 'it', 'eats', '.'], DST = 6;
    const rnn = $('#path-rnn'), att = $('#path-att'), arc = svg('svg', { 'aria-hidden': 'true' }), path = svg('path');
    let src = 1, timers = [];
    const chips = box => WORDS.map((w, i) => { const b = h('button', { type: 'button', class: 'chip', text: w, disabled: i > DST || i === DST, onclick: () => choose(i) }); box.append(b); return b; });
    const R = chips(rnn), A = chips(att); arc.append(path); att.append(arc);
    function drawArc() { const W = att.clientWidth, x = i => (i + .5) * W / WORDS.length; path.setAttribute('d', `M${x(src)},34 C${x(src)},2 ${x(DST)},2 ${x(DST)},34`); }
    function choose(i) {
      src = i; timers.forEach(clearTimeout); timers = [];
      for (const set of [R, A]) set.forEach((b, k) => { b.classList.toggle('src', k === src); b.classList.toggle('dst', k === DST); b.classList.remove('lit', 'passed'); });
      const hops = DST - src; drawArc();
      $('#path-att-n').textContent = '1 step, whatever the distance';
      const done = () => { $('#path-rnn-n').textContent = `${hops} hand-off${hops > 1 ? 's' : ''}, one after another`; };
      if (reduced) { for (let k = src + 1; k < DST; k++) R[k].classList.add('passed'); done(); return; }
      $('#path-rnn-n').textContent = 'passing it along …';
      for (let s = 1; s <= hops; s++) timers.push(setTimeout(() => {
        R.forEach(b => b.classList.remove('lit')); if (src + s < DST) R[src + s].classList.add('lit', 'passed'); if (s === hops) done();
      }, s * 330));
    }
    onResize(att, drawArc); choose(1);
  })();

  /* ---------- Figure 3: one real question ---------- */
  (() => {
    const stepsBox = $('#qk-steps'), table = $('#qk-table'), headBox = $('#qk-heads');
    let step = 2, head = 0;
    HEADS.forEach((name, hh) => headBox.append(h('button', { type: 'button', class: 'hd ' + 'ab'[hh], 'aria-pressed': hh === head, onclick: () => { head = hh; render(); } }, h('i'), 'head ' + name)));
    function render() {
      const R = store.R, dk = M.cfg.d / H;
      step = Math.min(step, R.out.length - 1);
      stepsBox.replaceChildren(...R.out.map((w, i) => h('button', { type: 'button', class: 'chip', 'aria-pressed': i === step, text: w, onclick: () => { step = i; render(); } })));
      [...headBox.children].forEach((b, hh) => b.setAttribute('aria-pressed', hh === head));
      const ids = [1, ...R.out.slice(0, step).map(w => M.t2i[w])], L = TF.decode(M, ids, R.enc).layers[0];
      const q = [...L.crossQ[0].slice(head * dk, (head + 1) * dk)], keys = L.crossK.map(k => [...k.slice(head * dk, (head + 1) * dk)]);
      const scores = L.crossScores[head][0], w = L.cross[head][0], qs = Math.max(...q.map(Math.abs)), scale = Math.max(...keys.flat().map(Math.abs)), best = w.indexOf(Math.max(...w));
      const g = h('div', { class: 'qk', style: `--c:${INK[head]}` });
      g.append(h('div', { class: 'q' }, h('span', {}, 'The question asked while writing ', h('b', { text: R.out[step] }), ':'), spark(q, qs, () => INK[head])));
      for (const t of ['English word', 'its key', 'score', 'share of attention', '']) g.append(h('span', { class: 'hdr', text: t }));
      store.words.forEach((word, j) => {
        const c = j === best ? 'top' : '';
        g.append(h('span', { class: c, text: word }), spark(keys[j], scale, i => q[i] * keys[j][i] > 0 ? INK[head] : null),
          h('span', { class: 'num ' + c, text: (scores[j] >= 0 ? '+' : '−') + Math.abs(scores[j]).toFixed(1) }), h('span', { class: 'bar', style: `--v:${w[j]}` }, h('i')), h('span', { class: 'num ' + c, text: pct(w[j]) }));
      });
      table.replaceChildren(g);
    }
    store.subs.push(render);
  })();

  /* ---------- Figure 4: three places to look ---------- */
  const three = (() => {
    const TABS = [['enc', 'English reads English'], ['cross', 'French reads English'], ['self', 'French reads its own past']];
    const HINT = { enc: 'Tap a word on the right to see what it consults. Try an article: the, a, my or this.', cross: 'Each French word reads the English word it translates. Follow the lines where the order changes.', self: 'Each word can only read upward, to words already written. Tap il or elle if the sentence has one.' };
    const note = $('#three-note'), tabBox = $('#three-tabs');
    let tab = 'enc', data = null;
    const view = AttnView($('#three-attn'), { onFocus: renderNote });
    legend($('#three-legend'), view, renderNote);
    TABS.forEach(([id, label]) => tabBox.append(h('button', { type: 'button', class: 'tab', role: 'tab', 'aria-selected': id === tab, 'data-tab': id, text: label, onclick: () => { tab = id; render(); } })));
    function render() {
      const R = store.R, en = store.words, fr = R.out;
      if (tab === 'enc') data = { left: en, right: en, heads: R.enc.layers[0].att, labels: ['English, consulted', 'English, asking'] };
      else if (tab === 'cross') data = { left: en, right: fr, heads: crossHeads(R), labels: ['English, consulted', 'French, being written'] };
      else data = { left: ['start', ...fr.slice(0, -1)], right: fr, heads: HEADS.map((_, hh) => R.steps.map(s => s.self[hh])), labels: ['French so far', 'French, being written'], startLeft: true };
      [...tabBox.children].forEach(b => b.setAttribute('aria-selected', b.dataset.tab === tab));
      const hd = view.root.querySelector('.av-head'); hd.firstChild.textContent = data.labels[0]; hd.lastChild.textContent = data.labels[1];
      view.set(data); renderNote();
    }
    function renderNote() {
      const f = view.focus();
      if (!f || !data) { note.replaceChildren(h('h3', { text: 'What to try' }), h('p', { text: HINT[tab] })); return; }
      const kids = [h('h3', { text: f.side === 'right' ? `“${data.right[f.i]}” reads` : `Who reads “${data.left[f.i]}”` })];
      HEADS.forEach((name, hh) => {
        if (!view.heads[hh]) return;
        const pairs = f.side === 'right' ? (data.heads[hh][f.i] || []).map((w, j) => [w, data.left[j]]) : data.heads[hh].map((row, r) => [row && row[f.i] || 0, data.right[r]]);
        const top = pairs.filter(p => p[0] >= 0.05).sort((a, b) => b[0] - a[0]).slice(0, 4), ul = h('ul', { class: 'note-list' });
        ul.append(h('li', {}, h('span', { class: 'sw', style: `background:${INK[hh]}` }), h('b', { text: 'head ' + name })));
        if (!top.length) ul.append(h('li', { text: 'nothing above 5%' }));
        for (const [w, word] of top) ul.append(h('li', { class: 'num', text: `${pct(w).padStart(4, ' ')}  ${word}` }));
        kids.push(ul);
      });
      note.replaceChildren(...kids);
    }
    store.subs.push(render);
    $('#three-shuffle').addEventListener('click', () => setSentence(LANG.sentence(rnd, 0.45, 0.6, tab === 'self' ? 1 : 0.4)));
    const PRESETS = {
      enc: ['the small black cow sees my new green hat .', R => ['right', 0]],
      cross: ['a pretty white horse follows this new blue car .', () => null],
      self: ['the small black cow sees my new green hat and it eats a red pear .', R => ['right', R.out.findIndex(w => w === 'il' || w === 'elle')]],
    };
    document.querySelectorAll('[data-preset]').forEach(b => b.addEventListener('click', () => {
      const [text, lock] = PRESETS[b.dataset.preset]; tab = b.dataset.preset; setSentence(text.split(' '));
      const l = lock(store.R); if (l) view.lock(l[0], l[1]);
      $('#fig-three').scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'center' });
    }));
  })();

  /* ---------- a shared test runner: 200 fixed sentences, run in slices so the page stays live ---------- */
  const TEST = (() => { const r = LANG.rng(7), a = []; for (let i = 0; i < 200; i++) { const s = LANG.sentence(r); a.push({ s, ref: LANG.reference(s).out }); } return a; })();
  function runTest(S, onTick, job) {
    let i = 0, ok = 0; const fails = [];
    (function slice() {
      if (job.cancelled) return;
      for (const end = Math.min(i + 20, TEST.length); i < end; i++) { const t = TEST[i], out = TF.translate(M, t.s, S).out; if (same(out, t.ref)) ok++; else if (fails.length < 3) fails.push({ ...t, out }); }
      onTick(ok, i, fails, i >= TEST.length);
      if (i < TEST.length) setTimeout(slice, 0);
    })();
  }

  /* ---------- Figure 5: the position stamp ---------- */
  (() => {
    const cv = $('#pos-canvas'), probe = $('#pos-probe'), out = $('#pos-out'), btn = $('#pos-toggle'), count = $('#pos-count');
    const N = 20, D = M.cfg.d; let col = 3, off = false, job = { cancelled: false };
    function draw() {
      const W = cv.clientWidth, Ht = cv.clientHeight, r = devicePixelRatio || 1; cv.width = W * r; cv.height = Ht * r;
      const c = cv.getContext('2d'); c.scale(r, r); const cw = W / N, ch = Ht / D;
      for (let p = 0; p < N; p++) for (let d = 0; d < D; d++) { const g = Math.round(255 - (M.pe[p][d] + 1) / 2 * 235); c.fillStyle = `rgb(${g},${g},${g + 3})`; c.fillRect(p * cw, d * ch, Math.ceil(cw), Math.ceil(ch)); }
      c.strokeStyle = INK[0]; c.lineWidth = 2; c.strokeRect(col * cw + 1, 1, cw - 2, Ht - 2);
      probe.replaceChildren(h('span', { text: `position ${col}` }), spark([...M.pe[col]], 1, () => css('--ink')), h('span', { class: 'count', text: 'its 16 numbers; tap another column' }));
    }
    const pick = e => { const b = cv.getBoundingClientRect(), c = Math.max(0, Math.min(N - 1, Math.floor((e.clientX - b.left) / b.width * N))); if (c !== col) { col = c; draw(); } };
    cv.addEventListener('pointermove', pick); cv.addEventListener('pointerdown', pick);
    function render() {
      const lab = t => h('span', { class: 'lab', text: t }), kids = [h('div', {}, lab('English'), store.words.join(' ')), h('div', {}, lab('With the stamp'), store.R.out.join(' '))];
      if (off) kids.push(h('div', {}, lab('Without it'), ...marked(TF.translate(M, store.words, { noPos: true }).out, store.ref)));
      out.replaceChildren(...kids);
    }
    btn.addEventListener('click', () => {
      off = !off; btn.setAttribute('aria-pressed', off); btn.textContent = off ? 'Restore the position stamp' : 'Erase the position stamp'; render();
      job.cancelled = true; job = { cancelled: false }; count.textContent = '';
      if (off) runTest({ noPos: true }, (ok, n, _, done) => { count.textContent = done ? `${ok} of 200 test sentences correct without it` : `testing ${n} / 200`; }, job);
    });
    store.subs.push(render); onResize(cv, draw);
  })();

  /* ---------- Figure 6: be the attention head ---------- */
  (() => {
    const msg = $('#game-msg'), status = $('#game-status'), scoreEl = $('#game-score'), bestEl = $('#game-best');
    let g = null, best = 0;
    try { best = +localStorage.getItem('aiayn-best') || 0; } catch (e) { /* storage is optional */ }
    const view = AttnView($('#game-attn'), { leftLabel: 'English: tap your guess', rightLabel: 'French', pick });
    function start() {
      const words = LANG.sentence(rnd, 0.5, 0.65, 0.3), R = TF.translate(M, words);
      g = { words, R, k: 0, total: 0, played: 0, over: false };
      view.set({ left: words, right: R.out, heads: crossHeads(R) }); view.show(0, null, true);
      msg.textContent = 'The first French word is coming. Which English word is the model reading?'; hud();
    }
    function hud() {
      const n = g.R.out.length;
      status.textContent = g.over ? 'Sentence finished' : `Word ${g.k + 1} of ${n}`;
      scoreEl.textContent = g.played ? `average ${Math.round(g.total / g.played)} / 100` : 'no guesses yet';
      bestEl.textContent = best ? `best finished sentence: ${best} / 100` : '';
    }
    function pick(j) {
      if (g.over) return;
      const st = g.R.steps[g.k], avg = g.words.map((_, i) => st.cross.reduce((a, hd) => a + hd[i], 0) / H), top = Math.max(...avg), ti = avg.indexOf(top);
      const pts = j === ti ? 100 : Math.round(100 * avg[j] / top);
      g.total += pts; g.played++; g.k++;
      const badge = h('span', { class: 'pts', text: '+' + pts });
      if (j === ti) msg.replaceChildren(badge, `Yes. It read “${g.words[ti]}” to write “${st.word}”.`);
      else msg.replaceChildren(badge, `To write “${st.word}” it gave ${pct(top)} of its attention to “${g.words[ti]}”, and ${pct(avg[j])} to “${g.words[j]}”.`);
      finishOrNext(true);
    }
    function finishOrNext(scored) {
      const n = g.R.out.length;
      if (g.k >= n) {
        g.over = true; view.show(n);
        const avg = g.played ? Math.round(g.total / g.played) : 0;
        if (scored && g.played === n) {
          if (avg > best) { best = avg; try { localStorage.setItem('aiayn-best', best); } catch (e) { /* optional */ } }
          msg.append(' ', h('b', { text: `Sentence done: ${avg} / 100. ` }), avg >= 95 ? 'You could stand in for the head.' : avg >= 75 ? 'Close. The misses are usually word order.' : 'The trick is French word order: colours move after the noun, words like big and new stay before it.');
        }
      } else view.show(g.k, g.k - 1, true);
      hud();
    }
    $('#game-new').addEventListener('click', start);
    $('#game-reveal').addEventListener('click', () => { if (g.over) return; g.k = g.R.out.length; msg.textContent = 'Revealed. Only the words you guessed count toward the average.'; finishOrNext(false); });
    start();
  })();

  /* ---------- Figure 7: the machine, with removable parts ---------- */
  (() => {
    const box = $('#machine'), result = $('#bench-result');
    const fresh = () => ({ noPos: false, noFF: false, noRes: false, uniform: { enc: false, self: false, cross: false }, kill: { enc: new Set(), self: new Set(), cross: new Set() } });
    let S = fresh(), job = { cancelled: false };
    const NAMES = { enc: 'encoder self-attention', self: 'decoder self-attention', cross: 'attention over the English' };
    const attn = (kind, title) => {
      const ctl = h('div', { class: 'ctl' });
      HEADS.forEach((name, hh) => ctl.append(h('button', { type: 'button', class: 'cut', style: `--c:${INK[hh]}`, 'data-k': `kill.${kind}.${hh}`, onclick: () => { const s = S.kill[kind]; s.has(hh) ? s.delete(hh) : s.add(hh); run(); } }, h('i'), 'head ' + name)));
      ctl.append(h('button', { type: 'button', class: 'cut', 'data-k': `uniform.${kind}`, text: 'blur', onclick: () => { S.uniform[kind] = !S.uniform[kind]; run(); } }));
      return h('div', { class: 'part', 'data-part': kind }, h('h4', { text: title }), ctl);
    };
    const whole = (key, title, sub, cls = '') => h('button', { type: 'button', class: 'part ' + cls, 'data-k': key, onclick: () => { S[key] = !S[key]; run(); } }, h('h4', { text: title }), h('small', { text: sub }));
    box.append(
      whole('noPos', 'Position stamp', 'added to every word before anything else', 'wide'),
      h('div', { class: 'stack' }, h('h3', { text: 'Encoder: reads the English' }), h('div', { class: 'io', text: 'English words' }),
        attn('enc', 'Self-attention'), whole('noFF', 'Feed-forward network', 'reworks each word on its own'), h('div', { class: 'io end', text: 'to the decoder →' })),
      h('div', { class: 'stack' }, h('h3', { text: 'Decoder: writes the French' }), h('div', { class: 'io', text: 'French written so far' }),
        attn('self', 'Masked self-attention'), attn('cross', 'Attention over the English'), whole('noFF', 'Feed-forward network', 'reworks each word on its own'), h('div', { class: 'io end', text: 'next French word' })),
      whole('noRes', 'Residual connections', 'the bypass around every block above', 'wide'));
    function removed() {
      const a = [];
      if (S.noPos) a.push('the position stamp');
      for (const k of ['enc', 'self', 'cross']) { if (S.uniform[k]) a.push(NAMES[k] + ' (blurred)'); for (const hh of [...S.kill[k]].sort()) a.push(`${NAMES[k]} head ${HEADS[hh]}`); }
      if (S.noFF) a.push('the feed-forward networks'); if (S.noRes) a.push('the residual connections');
      return a;
    }
    function paint() {
      box.querySelectorAll('[data-k]').forEach(b => {
        const p = b.dataset.k.split('.'), on = p[0] === 'kill' ? S.kill[p[1]].has(+p[2]) : p[0] === 'uniform' ? S.uniform[p[1]] : S[p[0]];
        b.setAttribute('aria-pressed', on); if (b.classList.contains('part')) b.classList.toggle('gone', on);
      });
      box.querySelectorAll('[data-part]').forEach(d => { const k = d.dataset.part; d.classList.toggle('gone', S.kill[k].size === H); });
    }
    function run() {
      paint(); job.cancelled = true; job = { cancelled: false };
      const big = h('div', { class: 'big' }), meter = h('div', { class: 'meter', style: '--v:0' }, h('i')), verdict = h('p', { class: 'verdict' }), fails = h('ul', { class: 'fails' });
      result.replaceChildren(big, meter, verdict, fails);
      const gone = removed();
      runTest(S, (ok, n, bad, done) => {
        big.replaceChildren(String(ok), h('small', { text: done ? 'of 200 test sentences correct' : `of ${n} so far` })); meter.style.setProperty('--v', ok / 200);
        if (!done) return;
        verdict.textContent = !gone.length ? 'Nothing removed. Every test sentence is translated exactly.'
          : (ok === 200 ? 'No effect on these sentences without ' : ok === 0 ? 'Nothing survives without ' : 'Damaged without ') + list(gone) + '.';
        const lab = t => h('span', { class: 'lab', text: t });
        fails.replaceChildren(...bad.map(f => h('li', {}, h('div', {}, lab('English'), h('span', { text: f.s.join(' ') })), h('div', {}, lab('Model'), h('span', {}, ...marked(f.out, f.ref))), h('div', {}, lab('Rulebook'), h('span', { text: f.ref.join(' ') })))));
      }, job);
    }
    $('#bench-reset').addEventListener('click', () => { S = fresh(); run(); });
    run();
  })();

  /* ---------- Figure 8: how it learned ---------- */
  (() => {
    const box = $('#train-chart'), curve = MODEL_PACK.cfg.curve.filter(p => p[0] <= 1000), warm = MODEL_PACK.cfg.warmup, MAXS = 1000;
    function draw() {
      const W = box.clientWidth || 600, Ht = 230, m = { l: 42, r: 14, t: 16, b: 30 }, pw = W - m.l - m.r, ph = Ht - m.t - m.b;
      const x = s => m.l + s / MAXS * pw, y = v => m.t + (1 - v) * ph, s = svg('svg', { viewBox: `0 0 ${W} ${Ht}`, height: Ht, role: 'img', 'aria-label': 'Line chart: held-out accuracy stays near zero for 300 training steps, then climbs to 100 percent' });
      const text = (t, px, py, anchor = 'middle', cls = '') => { const e = svg('text', { x: px, y: py, 'text-anchor': anchor, class: cls }); e.textContent = t; s.append(e); };
      for (const v of [0, .5, 1]) { s.append(svg('line', { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: css('--rule'), 'stroke-width': 1 })); text(pct(v), m.l - 7, y(v) + 4, 'end'); }
      for (const st of [0, 250, 500, 750, 1000]) text(st === 0 ? 'step 0' : String(st), x(st), Ht - 8, st === 0 ? 'start' : st === 1000 ? 'end' : 'middle');
      const lr = st => Math.min(1 / Math.sqrt(st), st * Math.pow(warm, -1.5)), peak = lr(warm); let d = '';
      for (let st = 1; st <= MAXS; st += 10) d += (d ? 'L' : 'M') + x(st).toFixed(1) + ',' + y(lr(st) / peak * .92).toFixed(1);
      s.append(svg('path', { d, fill: 'none', stroke: css('--ink-2'), 'stroke-width': 1.2, 'stroke-dasharray': '4 4' }));
      text('learning rate', x(warm) - 8, y(.92) - 3, 'end');
      const line = curve.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ',' + y(p[2]).toFixed(1)).join('');
      s.append(svg('path', { d: line + `L${x(MAXS)},${y(0)}L${x(0)},${y(0)}Z`, fill: INK[0], 'fill-opacity': .1 }), svg('path', { d: line, fill: 'none', stroke: INK[0], 'stroke-width': 2.5, 'stroke-linejoin': 'round' }));
      const full = curve.find(p => p[2] >= 1), half = curve.find(p => p[2] >= .5);
      for (const p of [half, full]) s.append(svg('circle', { cx: x(p[0]), cy: y(p[2]), r: 4, fill: css('--paper'), stroke: INK[0], 'stroke-width': 2 }));
      text(`step ${half[0]}: ${pct(half[2])}`, x(half[0]) + 10, y(half[2]) + 16, 'start', 'lbl'); text(`step ${full[0]}: 100%`, x(full[0]) + 8, y(1) + 18, 'start', 'lbl');
      box.replaceChildren(s);
    }
    onResize(box, draw); draw();
  })();

  setSentence('the small black cow sees my new green hat .'.split(' '));
})();
