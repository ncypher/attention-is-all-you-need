"""A tiny Transformer, built the way Vaswani et al. (2017) describe it,
trained on a toy English->French grammar. Exports weights + reference outputs.

usage: python3 model/train.py N_ENC N_DEC D H DFF STEPS SEED OUT.json
       (the model on the page:  python3 model/train.py 1 1 16 2 32 3000 3 trained.json)
"""
import json, math, random, sys, hashlib, time
import torch, torch.nn as nn, torch.nn.functional as F

N_ENC, N_DEC, D, H, DFF, STEPS, SEED = [int(x) for x in sys.argv[1:8]]
OUT = sys.argv[8]
torch.manual_seed(SEED); random.seed(SEED)
torch.set_num_threads(2)

import os
LEX = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lexicon.json'), encoding='utf-8'))
DETS, PRE, POST, NOUNS, VERBS = LEX['dets'], LEX['pre'], LEX['post'], LEX['nouns'], LEX['verbs']

SRC_VOCAB = ['<pad>'] + list(DETS) + list(PRE) + list(POST) + list(NOUNS) + list(VERBS) + ['and', 'it', '.']
tv = []
for a, b in DETS.values(): tv += [a, b]
for a, b in PRE.values(): tv += [a, b]
for a, b in POST.values(): tv += [a, b]
tv += [v[0] for v in NOUNS.values()] + list(VERBS.values()) + ['et', 'il', 'elle']
TGT_VOCAB = ['<pad>', '<s>'] + list(dict.fromkeys(tv)) + ['.']
S2I = {w: i for i, w in enumerate(SRC_VOCAB)}; T2I = {w: i for i, w in enumerate(TGT_VOCAB)}

def rand_np():
    det = random.choice(list(DETS)); noun = random.choice(list(NOUNS))
    pre = random.choice(list(PRE)) if random.random() < 0.4 else None
    post = random.choice(list(POST)) if random.random() < 0.55 else None
    return [w for w in (det, pre, post, noun) if w]

def translate_np(words):
    noun = words[-1]; fr, g = NOUNS[noun]; gi = 0 if g == 'm' else 1
    out = [DETS[words[0]][gi]]
    pre = [w for w in words[1:-1] if w in PRE]; post = [w for w in words[1:-1] if w in POST]
    out += [PRE[w][gi] for w in pre] + [fr] + [POST[w][gi] for w in post]
    return out

def rand_pair():
    a, v, b = rand_np(), random.choice(list(VERBS)), rand_np()
    s, t = a + [v] + b, translate_np(a) + [VERBS[v]] + translate_np(b)
    if random.random() < 0.4:      # "... and it <verb> <noun phrase>": 'it' is always the subject
        v2, c = random.choice(list(VERBS)), rand_np()
        s += ['and', 'it', v2] + c
        t += ['et', 'il' if NOUNS[a[-1]][1] == 'm' else 'elle', VERBS[v2]] + translate_np(c)
    return s + ['.'], t + ['.']

def held_out(src):  # one sentence in ten is never trained on
    return int(hashlib.md5(' '.join(src).encode()).hexdigest(), 16) % 10 == 0

def batch(n, want_held=False):
    S, T = [], []
    while len(S) < n:
        s, t = rand_pair()
        if held_out(s) != want_held: continue
        S.append([S2I[w] for w in s]); T.append([T2I['<s>']] + [T2I[w] for w in t])
    ls, lt = max(map(len, S)), max(map(len, T))
    S = torch.tensor([x + [0] * (ls - len(x)) for x in S]); T = torch.tensor([x + [0] * (lt - len(x)) for x in T])
    return S, T

def pos_enc(n, d):
    pe = torch.zeros(n, d); pos = torch.arange(n).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe

class MHA(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
    def forward(s, x, mem, mask):
        B, n, _ = x.shape; m = mem.shape[1]; dk = D // H
        q = s.q(x).view(B, n, H, dk).transpose(1, 2)
        k = s.k(mem).view(B, m, H, dk).transpose(1, 2)
        v = s.v(mem).view(B, m, H, dk).transpose(1, 2)
        sc = q @ k.transpose(-1, -2) / math.sqrt(dk)
        sc = sc.masked_fill(~mask, -1e9)
        a = sc.softmax(-1); s.last = a.detach()
        return s.o((F.dropout(a, 0.1, s.training) @ v).transpose(1, 2).reshape(B, n, D))

class FF(nn.Module):
    def __init__(s):
        super().__init__(); s.a = nn.Linear(D, DFF); s.b = nn.Linear(DFF, D)
    def forward(s, x): return s.b(F.relu(s.a(x)))

class Enc(nn.Module):
    def __init__(s):
        super().__init__(); s.att, s.ff, s.n1, s.n2 = MHA(), FF(), nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(s, x, m):
        x = s.n1(x + F.dropout(s.att(x, x, m), 0.1, s.training))
        return s.n2(x + F.dropout(s.ff(x), 0.1, s.training))

class Dec(nn.Module):
    def __init__(s):
        super().__init__(); s.self, s.cross, s.ff = MHA(), MHA(), FF()
        s.n1, s.n2, s.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(s, y, mem, tm, sm):
        y = s.n1(y + F.dropout(s.self(y, y, tm), 0.1, s.training))
        y = s.n2(y + F.dropout(s.cross(y, mem, sm), 0.1, s.training))
        return s.n3(y + F.dropout(s.ff(y), 0.1, s.training))

class Transformer(nn.Module):
    def __init__(s):
        super().__init__()
        s.se = nn.Embedding(len(SRC_VOCAB), D); s.te = nn.Embedding(len(TGT_VOCAB), D)
        nn.init.normal_(s.se.weight, 0, D ** -0.5); nn.init.normal_(s.te.weight, 0, D ** -0.5)
        s.enc = nn.ModuleList(Enc() for _ in range(N_ENC)); s.dec = nn.ModuleList(Dec() for _ in range(N_DEC))
        s.register_buffer('pe', pos_enc(32, D))
    def encode(s, S):
        sm = (S != 0)[:, None, None, :]
        x = F.dropout(s.se(S) * math.sqrt(D) + s.pe[:S.shape[1]], 0.1, s.training)
        for l in s.enc: x = l(x, sm)
        return x, sm
    def decode(s, T, mem, sm):
        n = T.shape[1]
        tm = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None] & (T != 0)[:, None, None, :]
        y = F.dropout(s.te(T) * math.sqrt(D) + s.pe[:n], 0.1, s.training)
        for l in s.dec: y = l(y, mem, tm, sm)
        return y @ s.te.weight.T          # output projection shares the embedding matrix
    def forward(s, S, T):
        mem, sm = s.encode(S); return s.decode(T, mem, sm)

model = Transformer()
nparams = sum(p.numel() for p in model.parameters())
WARM = 400
opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: D ** -0.5 * min((st + 1) ** -0.5, (st + 1) * WARM ** -1.5))

@torch.no_grad()
def greedy(S, maxlen=20):
    model.eval(); mem, sm = model.encode(S)
    T = torch.full((S.shape[0], 1), T2I['<s>'])
    for _ in range(maxlen):
        nxt = model.decode(T, mem, sm)[:, -1].argmax(-1, keepdim=True); T = torch.cat([T, nxt], 1)
    return T

def exact(n=2000, held=True):
    S, T = batch(n, held); P = greedy(S); ok = 0
    for i in range(n):
        ref = [x for x in T[i].tolist() if x]; hyp = P[i].tolist()[:len(ref)]
        ok += ref == hyp
    return ok / n

train_secs = 0.0; curve = []
for step in range(1, STEPS + 1):
    t0 = time.time(); model.train(); S, T = batch(64)
    logits = model(S, T[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), T[:, 1:].reshape(-1), ignore_index=0, label_smoothing=0.1)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step(); train_secs += time.time() - t0
    if step == 1 or (step <= 1000 and step % 50 == 0) or step % 500 == 0 or step == STEPS:
        acc = exact(200); curve.append([step, round(loss.item(), 3), round(acc, 4)])
        print(step, round(loss.item(), 3), acc, flush=True)
secs = train_secs
final = exact(3000, True); seen = exact(500, False)
print('params', nparams, 'held-out exact', final, 'train-dist exact', seen, 'secs', round(secs, 1))

# ---- export ----
sd = {k: v for k, v in model.state_dict().items() if k != 'pe'}
weights = {k: {'shape': list(v.shape), 'data': [round(x, 5) for x in v.flatten().tolist()]} for k, v in sd.items()}
refs = []
random.seed(123)
model.eval()
for _ in range(6):
    s, t = rand_pair(); S = torch.tensor([[S2I[w] for w in s]]); T = torch.tensor([[T2I['<s>']] + [T2I[w] for w in t]])
    with torch.no_grad(): lg = model(S, T[:, :-1])
    refs.append({'src': s, 'tgt': t, 'logits_last': [round(x, 4) for x in lg[0, -1].tolist()],
                 'enc_att': [l.att.last[0].tolist() for l in model.enc],
                 'cross_att': [l.cross.last[0].tolist() for l in model.dec],
                 'self_att': [l.self.last[0].tolist() for l in model.dec]})
json.dump({'cfg': {'n_enc': N_ENC, 'n_dec': N_DEC, 'd': D, 'h': H, 'dff': DFF, 'steps': STEPS, 'seed': SEED, 'warmup': WARM,
                   'params': nparams, 'secs': round(secs, 1), 'held_out_exact': final, 'curve': curve},
           'src_vocab': SRC_VOCAB, 'tgt_vocab': TGT_VOCAB, 'weights': weights, 'refs': refs}, open(OUT, 'w'))
torch.save(model.state_dict(), OUT.replace('.json', '.pt'))
