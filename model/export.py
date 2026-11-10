"""Pack a trained model for the browser.

    python3 model/export.py trained.json

Writes js/model.js (16-bit quantised weights as one base64 blob, plus the lexicon)
and test/reference-outputs.json (PyTorch outputs that test/verify.js checks the browser engine against).
"""
import base64, json, os, struct, sys

here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
m = json.load(open(sys.argv[1], encoding='utf-8'))
tensors, blob = [], bytearray()
for name, t in m['weights'].items():
    scale = (max(abs(x) for x in t['data']) or 1.0) / 32767.0
    q = [max(-32767, min(32767, round(x / scale))) for x in t['data']]
    blob += struct.pack('<%dh' % len(q), *q)
    tensors.append([name, t['shape'], float('%.8g' % scale)])
pack = {'cfg': m['cfg'], 'src_vocab': m['src_vocab'], 'tgt_vocab': m['tgt_vocab'], 'tensors': tensors,
        'blob': base64.b64encode(bytes(blob)).decode()}
lex = json.load(open(os.path.join(here, 'lexicon.json'), encoding='utf-8'))
dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(',', ':'))
with open(os.path.join(root, 'js', 'model.js'), 'w', encoding='utf-8') as f:
    f.write('/* The trained model: %s weights, 16-bit quantised, base64. Produced by model/train.py + model/export.py. */\n' % format(m['cfg']['params'], ','))
    f.write('const MODEL_PACK = ' + dump(pack) + ';\nconst LEXICON = ' + dump(lex) + ';\n')
    f.write("if (typeof module !== 'undefined') module.exports = { MODEL_PACK, LEXICON };\n")
json.dump(m['refs'], open(os.path.join(root, 'test', 'reference-outputs.json'), 'w'))
print('wrote js/model.js (%d KB) and test/reference-outputs.json' % (os.path.getsize(os.path.join(root, 'js', 'model.js')) // 1024))
