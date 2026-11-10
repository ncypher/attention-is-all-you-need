/* Checks the browser engine against the PyTorch model it was exported from,
   then measures every number the page quotes.   Run:  node test/verify.js  */
const fs = require('fs'), path = require('path');
const { TF, LANG } = require('../js/engine.js');
const { MODEL_PACK, LEXICON } = require('../js/model.js');
const refs = JSON.parse(fs.readFileSync(path.join(__dirname, 'reference-outputs.json')));
LANG.init(LEXICON);
const M = TF.load(MODEL_PACK), H = M.cfg.h;
let failed = 0;
const check = (name, ok, detail = '') => { console.log((ok ? 'ok   ' : 'FAIL ') + name + (detail ? '  ' + detail : '')); if (!ok) failed++; };

// 1. Same numbers as PyTorch (weights are 16-bit quantised, so allow a small gap)
let dLogit = 0, dAtt = 0;
const gap = (A, B) => A.forEach((hd, h) => hd.forEach((row, i) => row.forEach((v, j) => { dAtt = Math.max(dAtt, Math.abs(v - B[h][i][j])); })));
for (const r of refs) {
  const enc = TF.encode(M, r.src.map(w => M.s2i[w])), ids = [1, ...r.tgt.slice(0, -1).map(w => M.t2i[w])];
  const full = TF.decode(M, ids, enc, {}, false), fast = TF.decode(M, ids, enc, {}, true);
  r.logits_last.forEach((v, i) => { if (i > 1) dLogit = Math.max(dLogit, Math.abs(v - full.logits[i]), Math.abs(v - fast.logits[i])); });
  gap(r.enc_att[0], enc.layers[0].att); gap(r.cross_att[0], full.layers[0].cross); gap(r.self_att[0], full.layers[0].self);
}
check('engine matches PyTorch logits', dLogit < 0.01, `max gap ${dLogit.toFixed(5)}`);
check('engine matches PyTorch attention', dAtt < 0.005, `max gap ${dAtt.toFixed(5)}`);

// 2. The rulebook translator agrees with the training data generator's examples
check('rulebook reproduces reference translations', refs.every(r => LANG.reference(r.src).out.join(' ') === r.tgt.join(' ')));

// 3. Accuracy on random sentences
const accuracy = (S, N, seed) => { const r = LANG.rng(seed); let ok = 0; for (let i = 0; i < N; i++) { const s = LANG.sentence(r); if (LANG.reference(s).out.join(' ') === TF.translate(M, s, S).out.join(' ')) ok++; } return ok; };
const big = accuracy({}, 20000, 2017);
check('20,000 random sentences translated exactly', big === 20000, `${big}/20000`);

// 4. Sabotage bench numbers (same 200 sentences the page uses: seed 7)
const all = new Set([...Array(H).keys()]), bench = (label, S) => console.log('     ' + label.padEnd(34) + accuracy(S, 200, 7) + '/200');
console.log('\nsabotage bench (200 sentences, seed 7)');
bench('intact', {}); bench('no position stamp', { noPos: true });
bench('encoder self-attention blurred', { uniform: { enc: true } }); bench('decoder self-attention blurred', { uniform: { self: true } });
bench('cross-attention blurred', { uniform: { cross: true } }); bench('no feed-forward', { noFF: true }); bench('no residual connections', { noRes: true });
for (const k of ['enc', 'self', 'cross']) { for (let h = 0; h < H; h++) bench(`${k} head ${'AB'[h]} removed`, { kill: { [k]: new Set([h]) } }); bench(`${k} both heads removed`, { kill: { [k]: all } }); }

// 5. The attention habits the page describes, measured over 1,000 sentences
const r = LANG.rng(99), det = Array(H).fill(0), pro = Array(H).fill(0), proAdj = Array(H).fill(0), align = Array(H).fill(0); let nd = 0, np = 0, na = 0, minTop = 1;
for (let i = 0; i < 1000; i++) {
  const s = LANG.sentence(r, 0.4, 0.55, 0.6), ref = LANG.reference(s), t = TF.translate(M, s);
  s.forEach((w, j) => { if (LANG.kind(w) === 'det') { let n = j; while (LANG.kind(s[n]) !== 'noun') n++; nd++; for (let h = 0; h < H; h++) det[h] += t.enc.layers[0].att[h][j][n]; } });
  t.steps.forEach((st, k) => {
    minTop = Math.min(minTop, Math.max(...st.probs)); na++; for (let h = 0; h < H; h++) align[h] += st.cross[h][ref.from[k]];
    if (st.word === 'il' || st.word === 'elle') { np++; const subjLen = ref.from.indexOf(ref.subj) + 1; for (let h = 0; h < H; h++) { pro[h] += st.self[h][1]; let m = 0; for (let q = 1; q <= subjLen; q++) m += st.self[h][q]; proAdj[h] += m; } }
  });
}
const pct = a => a.map(v => (100 * v).toFixed(0) + '%').join(' / ');
console.log('\nattention habits, head A / head B');
console.log('     encoder: article -> its own noun            ', pct(det.map(v => v / nd)));
console.log('     decoder self: il/elle -> first French word   ', pct(pro.map(v => v / np)));
console.log('     decoder self: il/elle -> subject noun phrase ', pct(proAdj.map(v => v / np)));
console.log('     cross: French word -> the English word it translates', pct(align.map(v => v / na)));
console.log('     lowest winning probability seen:', minTop.toFixed(2));

// 6. Size of the toy language
const c = o => Object.keys(o).length, nps = c(LEXICON.dets) * (c(LEXICON.pre) + 1) * (c(LEXICON.post) + 1) * c(LEXICON.nouns);
const total = nps * c(LEXICON.verbs) * nps * (1 + c(LEXICON.verbs) * nps);
console.log(`\nnoun phrases ${nps.toLocaleString()}  sentences ${total.toLocaleString()}  vocab ${M.src.length - 1} English / ${M.tgt.length - 2} French`);
console.log(failed ? `\n${failed} check(s) FAILED` : '\nall checks passed'); process.exit(failed ? 1 : 0);
