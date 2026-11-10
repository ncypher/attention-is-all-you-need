/* ---------- the model: a Transformer forward pass in plain JavaScript ----------
   Same arithmetic as the 2017 paper, at toy size. No libraries.
   Every sabotage switch on the page is a branch in this code. */
const TF = (() => {
  const b64ToI16 = b64 => {
    const bin = atob(b64), u = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i);
    return new Int16Array(u.buffer);
  };

  function load(pack) {
    const i16 = b64ToI16(pack.blob), W = {};
    let off = 0;
    for (const [name, shape, scale] of pack.tensors) {
      const n = shape.reduce((a, b) => a * b, 1), f = new Float32Array(n);
      for (let i = 0; i < n; i++) f[i] = i16[off + i] * scale;
      off += n; W[name] = { shape, data: f };
    }
    const cfg = pack.cfg, d = cfg.d;
    const pe = [];                                   // sinusoidal positions, straight from section 3.5
    for (let pos = 0; pos < 32; pos++) {
      const r = new Float32Array(d);
      for (let i = 0; i < d; i += 2) {
        const w = Math.exp(-Math.log(10000) * i / d);
        r[i] = Math.sin(pos * w); r[i + 1] = Math.cos(pos * w);
      }
      pe.push(r);
    }
    const s2i = {}, t2i = {};
    pack.src_vocab.forEach((w, i) => s2i[w] = i); pack.tgt_vocab.forEach((w, i) => t2i[w] = i);
    return { W, cfg, pe, s2i, t2i, src: pack.src_vocab, tgt: pack.tgt_vocab };
  }

  const linear = (x, w, b) => {                      // y = W x (+ b)
    const [o, n] = w.shape, y = new Float32Array(o), D = w.data;
    for (let i = 0; i < o; i++) {
      let s = b ? b.data[i] : 0; const base = i * n;
      for (let j = 0; j < n; j++) s += D[base + j] * x[j];
      y[i] = s;
    }
    return y;
  };
  const layerNorm = (x, g, b) => {
    const n = x.length; let m = 0, v = 0;
    for (let i = 0; i < n; i++) m += x[i]; m /= n;
    for (let i = 0; i < n; i++) v += (x[i] - m) * (x[i] - m); v = Math.sqrt(v / n + 1e-5);
    const y = new Float32Array(n);
    for (let i = 0; i < n; i++) y[i] = (x[i] - m) / v * g.data[i] + b.data[i];
    return y;
  };
  const softmax = a => {
    let mx = -Infinity; for (const v of a) if (v > mx) mx = v;
    let s = 0; const y = a.map(v => { const e = v === -Infinity ? 0 : Math.exp(v - mx); s += e; return e; });
    return y.map(v => v / s);
  };

  /* Multi-head attention. rows = which query positions to compute; allow(i,j) = may i look at j?
     sab = { uniform:bool, kill:Set(head) } */
  function attention(M, pre, X, MEM, rows, allow, sab) {
    const { W, cfg } = M, h = cfg.h, dk = cfg.d / h, root = Math.sqrt(dk);
    const K = MEM.map(r => linear(r, W[pre + '.k.weight'])), V = MEM.map(r => linear(r, W[pre + '.v.weight']));
    const att = [], scores = [], out = [], Q = [];
    for (let hh = 0; hh < h; hh++) { att.push([]); scores.push([]); }
    for (const i of rows) {
      const q = linear(X[i], W[pre + '.q.weight']), cat = new Float32Array(cfg.d); Q.push(q);
      for (let hh = 0; hh < h; hh++) {
        const o = hh * dk, sc = [];
        for (let j = 0; j < MEM.length; j++) {
          if (!allow(i, j)) { sc.push(-Infinity); continue; }
          let s = 0; for (let c = 0; c < dk; c++) s += q[o + c] * K[j][o + c];
          sc.push(s / root);                          // QK^T / sqrt(d_k)
        }
        let w = softmax(sc);
        if (sab && sab.uniform) { const n = sc.filter(v => v !== -Infinity).length; w = sc.map(v => v === -Infinity ? 0 : 1 / n); }
        att[hh].push(w); scores[hh].push(sc);
        if (sab && sab.kill && sab.kill.has(hh)) continue;
        for (let j = 0; j < MEM.length; j++) { const wj = w[j]; if (!wj) continue; for (let c = 0; c < dk; c++) cat[o + c] += wj * V[j][o + c]; }
      }
      out.push(linear(cat, W[pre + '.o.weight']));
    }
    return { out, att, scores, Q, K };
  }

  const feedForward = (M, pre, x) => {
    const { W } = M, hdn = linear(x, W[pre + '.a.weight'], W[pre + '.a.bias']);
    for (let i = 0; i < hdn.length; i++) if (hdn[i] < 0) hdn[i] = 0;       // ReLU
    return linear(hdn, W[pre + '.b.weight'], W[pre + '.b.bias']);
  };
  // residual connection + layer norm: LayerNorm(x + Sublayer(x))
  const addNorm = (M, pre, x, sub, S) => {
    const y = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) y[i] = (S.noRes ? 0 : x[i]) + (sub ? sub[i] : 0);
    return layerNorm(y, M.W[pre + '.weight'], M.W[pre + '.bias']);
  };
  const embed = (M, table, ids, S) => ids.map((id, pos) => {
    const d = M.cfg.d, r = new Float32Array(d), E = M.W[table].data, sc = Math.sqrt(d);
    for (let i = 0; i < d; i++) r[i] = E[id * d + i] * sc + (S.noPos ? 0 : M.pe[pos][i]);
    return r;
  });
  const sabFor = (S, kind) => ({ uniform: !!(S.uniform && S.uniform[kind]), kill: (S.kill && S.kill[kind]) || null });

  function encode(M, ids, S = {}) {
    let x = embed(M, 'se.weight', ids, S); const layers = [], all = ids.map((_, i) => i);
    for (let l = 0; l < M.cfg.n_enc; l++) {
      const p = 'enc.' + l, a = attention(M, p + '.att', x, x, all, () => true, sabFor(S, 'enc'));
      x = x.map((r, i) => addNorm(M, p + '.n1', r, a.out[i], S));
      x = x.map(r => addNorm(M, p + '.n2', r, S.noFF ? null : feedForward(M, p + '.ff', r), S));
      layers.push({ att: a.att, scores: a.scores });
    }
    return { mem: x, layers };
  }

  // One decoder pass over the words written so far. Returns next-word probabilities.
  function decode(M, ids, enc, S = {}, lastOnly = true) {
    let y = embed(M, 'te.weight', ids, S); const n = ids.length, layers = [];
    const one = lastOnly && M.cfg.n_dec === 1, rows = one ? [n - 1] : ids.map((_, i) => i);
    for (let l = 0; l < M.cfg.n_dec; l++) {
      const p = 'dec.' + l;
      const a = attention(M, p + '.self', y, y, rows, (i, j) => S.noMask || j <= i, sabFor(S, 'self'));   // the mask: no peeking right
      let z = rows.map((i, k) => addNorm(M, p + '.n1', y[i], a.out[k], S));
      const zr = z.map((_, k) => k);
      const c = attention(M, p + '.cross', z, enc.mem, zr, () => true, sabFor(S, 'cross'));
      z = z.map((r, k) => addNorm(M, p + '.n2', r, c.out[k], S));
      z = z.map(r => addNorm(M, p + '.n3', r, S.noFF ? null : feedForward(M, p + '.ff', r), S));
      layers.push({ self: a.att, cross: c.att, crossScores: c.scores, crossQ: c.Q, crossK: c.K });
      y = z;
    }
    const last = y[y.length - 1], E = M.W['te.weight'], logits = [];
    for (let v = 0; v < M.tgt.length; v++) { let s = 0; for (let i = 0; i < M.cfg.d; i++) s += E.data[v * M.cfg.d + i] * last[i]; logits.push(s); }
    logits[0] = -Infinity; logits[1] = -Infinity;       // never emit <pad> or <s>
    return { probs: softmax(logits), logits, layers };
  }

  // Greedy translation, one word at a time, each word fed back in.
  function translate(M, words, S = {}, maxLen = 20) {
    const enc = encode(M, words.map(w => M.s2i[w]), S), ids = [1], out = [], steps = [];
    for (let t = 0; t < maxLen; t++) {
      const r = decode(M, ids, enc, S); let best = 0;
      for (let v = 1; v < r.probs.length; v++) if (r.probs[v] > r.probs[best]) best = v;
      const L = r.layers[r.layers.length - 1], row = L.cross[0].length - 1;
      steps.push({ word: M.tgt[best], probs: r.probs, cross: L.cross.map(hd => hd[row]), self: L.self.map(hd => hd[row]) });
      out.push(M.tgt[best]); ids.push(best);
      if (M.tgt[best] === '.') break;
    }
    return { out, steps, enc };
  }

  return { load, encode, decode, translate, softmax };
})();

/* ---------- the toy language: a grammar small enough to check every answer ---------- */
const LANG = (() => {
  let LEX;
  const init = lex => { LEX = lex; };
  const rng = seed => () => { seed |= 0; seed = seed + 0x6D2B79F5 | 0; let t = Math.imul(seed ^ seed >>> 15, 1 | seed); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  const pick = (r, a) => a[Math.floor(r() * a.length)];
  const nounPhrase = (r, pPre = 0.4, pPost = 0.55) => {
    const np = [pick(r, Object.keys(LEX.dets))];
    if (r() < pPre) np.push(pick(r, Object.keys(LEX.pre)));
    if (r() < pPost) np.push(pick(r, Object.keys(LEX.post)));
    np.push(pick(r, Object.keys(LEX.nouns))); return np;
  };
  const sentence = (r, pPre, pPost, pIt = 0.4) => {
    const s = [...nounPhrase(r, pPre, pPost), pick(r, Object.keys(LEX.verbs)), ...nounPhrase(r, pPre, pPost)];
    if (r() < pIt) s.push('and', 'it', pick(r, Object.keys(LEX.verbs)), ...nounPhrase(r, pPre, pPost));
    return [...s, '.'];
  };
  const kind = w => w in LEX.dets ? 'det' : w in LEX.pre ? 'pre' : w in LEX.post ? 'post' : w in LEX.nouns ? 'noun' : w in LEX.verbs ? 'verb' : w === '.' ? 'stop' : w;
  // The rulebook answer, with the index of the English word each French word comes from.
  function reference(words) {
    const out = [], from = []; let i = 0;
    let subj = -1;
    const np = () => {
      const start = i; while (kind(words[i]) !== 'noun') i++;
      const ni = i++, g = LEX.nouns[words[ni]][1] === 'm' ? 0 : 1;
      if (subj < 0) subj = ni;
      out.push(LEX.dets[words[start]][g]); from.push(start);
      for (let k = start + 1; k < ni; k++) if (kind(words[k]) === 'pre') { out.push(LEX.pre[words[k]][g]); from.push(k); }
      out.push(LEX.nouns[words[ni]][0]); from.push(ni);
      for (let k = start + 1; k < ni; k++) if (kind(words[k]) === 'post') { out.push(LEX.post[words[k]][g]); from.push(k); }
    };
    np(); out.push(LEX.verbs[words[i]]); from.push(i); i++; np();
    if (words[i] === 'and') {                      // 'it' always means the subject
      out.push(LEX.link.and, LEX.link.it[LEX.nouns[words[subj]][1] === 'm' ? 0 : 1], LEX.verbs[words[i + 2]]);
      from.push(i, i + 1, i + 2); i += 3; np();
    }
    out.push('.'); from.push(i);
    return { out, from, subj };
  }
  return { init, rng, sentence, reference, kind, pick };
})();

if (typeof module !== 'undefined') module.exports = { TF, LANG };
