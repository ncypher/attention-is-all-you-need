# Attention Is All You Need · Playable Edition

The 2017 Transformer paper as a working model. A Transformer of 7,136 numbers runs in your browser and translates a toy English into French. Every attention line on the page is the model's real arithmetic, so you can watch it read, bet on where it will look next, and remove its parts to see which ones it needs.

**[Open the page](https://ncypher.github.io/attention-is-all-you-need/)** (needs GitHub Pages switched on for this repository: Settings → Pages → Deploy from a branch → `main` / root)

Part of Tom Harwood's [Conversational Artifacts](https://ncypher.github.io/tomfoolery/). Tom chose the subject. Everything else here was written by Claude Fable 5.2 in one session in November 2026: the prose, the toy grammar, the training script, the model, the JavaScript engine and the tests.

## What is on the page

| Figure | What you do |
| --- | --- |
| 1. The model at work | Build a sentence from menus and watch it translate, word by word, with both attention heads drawn |
| 2. Path length | Send a fact from one word to another through a recurrent chain, then through attention |
| 3. One real question | See a query, every key, the scores and the softmax for one decoder step |
| 4. Three places to look | Switch between encoder self-attention, cross-attention and masked decoder self-attention |
| 5. The position stamp | Read the sinusoid pattern, then erase it and watch the output fall apart |
| 6. Be the attention head | Guess which English word the model reads before each French word appears |
| 7. The machine | Remove heads, blur attention, cut the feed-forward layers or residuals; 200 test sentences rerun each time |
| 8. How it learned | The held-out accuracy curve from the actual training run |

## The model

| | Paper (base), 2017 | This page |
| --- | --- | --- |
| Encoder / decoder layers | 6 / 6 | 1 / 1 |
| d_model | 512 | 16 |
| Heads per layer | 8 | 2 |
| Feed-forward width | 2,048 | 32 |
| Parameters | about 65 million | 7,136 |
| Training | 12 hours on 8 P100 GPUs | 28 seconds on 2 CPU cores |

It follows the paper's design: sinusoidal positions, scaled dot-product attention, post-layer-norm residual blocks, and an output layer that shares the target embedding matrix. Training uses Adam with the paper's warm-up schedule (400 steps here), dropout 0.1 and label smoothing 0.1, for 3,000 steps of 64 sentences.

The toy language has 105 words and about 2.2 trillion possible sentences. Articles and adjectives agree with the noun's gender, colours move after the noun, and *it* in a second clause refers to the subject. One sentence in ten is held out of training.

Measured by `test/verify.js`:

- 20,000 of 20,000 random sentences translated exactly.
- The browser engine matches PyTorch to within 0.001 on logits and attention weights (the gap is 16-bit weight quantisation).
- Sabotage results on the page's 200 test sentences: no position stamp 0, cross-attention blurred 0, no residuals 0, encoder self-attention blurred 102, no feed-forward 116, decoder self-attention head A removed 149. Every one of those 51 failures is a wrong *il* or *elle*.
- In the encoder, head A puts 97% of an article's attention on its own noun. When the decoder writes *il* or *elle*, self-attention head A puts 95% of its attention on the subject noun phrase at the start of its own output.

The sabotage bench cuts parts out of a model that was trained with them. The paper's own variations (its Table 3) retrain each time, so the two are not comparable.

## Files

```
index.html               the page
css/page.css             styles
js/engine.js             the Transformer forward pass (about 150 lines) and the toy grammar, no dependencies
js/model.js              trained weights (16-bit, base64) and the lexicon; generated
js/app.js                the figures
model/train.py           PyTorch training script
model/export.py          packs a trained model into js/model.js
model/lexicon.json       the toy language
test/verify.js           checks the engine against PyTorch and measures every number quoted above
test/reference-outputs.json   PyTorch outputs for the check; generated
```

## Run it

Serve the folder with any static server and open `index.html`:

```sh
python3 -m http.server
```

Run the checks (Node 18 or later, no packages):

```sh
node test/verify.js
```

Retrain the model (PyTorch, CPU is enough):

```sh
python3 model/train.py 1 1 16 2 32 3000 3 trained.json
python3 model/export.py trained.json
node test/verify.js
```

Seed 3 on PyTorch 2.15 (CPU, 2 threads) reproduced the shipped weights byte for byte. Other versions or thread counts may give a different but equally accurate model, in which case the specific head behaviours described on the page can move to the other head or change. The seed was chosen from six for the clearest head behaviour; all six translated at least 19,992 of 20,000 test sentences.

## Source

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin, "Attention Is All You Need," [arXiv:1706.03762](https://arxiv.org/abs/1706.03762), 2017.

This is an independent explanation. It is not affiliated with the authors or with Google.
