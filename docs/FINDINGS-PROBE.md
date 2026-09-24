# A probe on internal states: the signal exists, and it is still not enough

Chapter 7's negative result rests on AUROC 0.5742 — neither token-level
uncertainty nor sampling-based semantic divergence ranks a globally-wrong step.
Chapter 9 named the untested alternative and committed to a prediction before
running it: ReProbe (Ni et al., [arXiv:2511.06209](https://arxiv.org/abs/2511.06209))
trains a sub-10M-parameter probe on frozen internal states and matches PRMs up
to 810× larger, but its margin is largest *out of domain* and strong PRMs reach
parity on GSM8K — so a probe here should land **near 0.9033, not above it**.

Reproduce:

```bash
python scripts/gpu_probe_states.py          # Kaggle T4x2, ~15 min
python scripts/exp_gate_pipeline.py --probe runs/probe_qwen25_7b.json
```

One teacher-forced pass per solution over the committed 500-solution corpus,
hidden states captured at each step's final token for all 29 hidden-state
tensors (the embedding output plus 28 transformer layers), logistic
probe per layer, same hash splits as Chapter 7.

---

## 1. The signal is there

| score | AUROC (test) |
|---|---|
| token-level only | 0.5589 |
| semantic divergence only | 0.5740 |
| both | 0.5742 |
| **probe on layer 25** | **0.6968** |
| ch. 5 task PRM (scope, not AUROC) | 0.9033 |

**+0.12 AUROC over everything Chapter 7 measured.** This is the first score in
the project that is clearly better than the token/sampling family, and it
settles the question Chapter 9 posed: the AUROC 0.56 result was about *those
signals*, not about step-level uncertainty in general. The generator's internal
states do encode something about whether the step it just wrote was sound.

It also lands **below** Chapter 9's prediction of ~0.9033, not near it. See §4.

### The layer profile says this is not a lucky draw

AUROC on the selection split, by layer:

```
layer  0   0.6099   (embeddings)
layer  6   0.8126
layer 12   0.7977
layer 18   0.8414
layer 25   0.8748   <- selected
layer 28   0.8596   (final)
```

A smooth rise from the embedding layer through the late-middle of the network,
peaking around layers 24–27 and easing slightly at the last layer. That shape is
what a real encoded property looks like; selection noise over 29 candidates would be
jagged and would not peak coherently.

---

## 2. And it still does not rescue risk control

Measured selective risk on the accepted set, same corpus, same calibrator, same
budget — only the score changes:

| α | target | old score | **probe** | oracle | binds? |
|---|---|---|---|---|---|
| 0.05 | 0.05 | 0.1491 | **0.1432** | 0.0885 | yes |
| 0.10 | 0.10 | 0.1494 | **0.1363** | 0.0885 | yes |
| 0.30 | 0.30 | 0.1538 | **0.1394** | 0.0885 | no |

The probe improves risk at every α **and uses fewer calls** (0.96/question
against 1.09 at α = 0.30). That is a genuine improvement, the first in this
project.

It also misses **every binding target**: 2.9× at α = 0.05, 1.4× at α = 0.10.

> Chapter 7's conclusion survives, and is now better supported. The earlier
> version could be read as "the signal happened to be absent." This shows the
> signal is present, measurable, and improves the gate — and the gate still
> does not hold its target.

The three scores line up against the oracle roughly in proportion to their
AUROC. The probe closes ~29% of the AUROC gap to a perfect score and ~22% of
the risk gap, which is the consistency one would want before believing either
number.

---

## 3. It does not rescue the verifier either

Projected accuracy against the 0.8022 no-gate baseline, task PRM at its
measured 9.87% false-alarm rate:

| score | calls/q | recall | projected accuracy |
|---|---|---|---|
| old score, split conformal | 1.09 | 0.1761 | 0.7912 |
| **probe** | **0.96** | **0.2465** | **0.7802** |
| oracle | 0.37 | 0.4366 | 0.9780 |

The probe catches **40% more wrong steps** than the old score on fewer calls —
and projected accuracy is still below the baseline. At AUROC 0.70 the gate is
still pointing the verifier at correct steps often enough that false alarms
dominate.

This sharpens §7.4 rather than contradicting it. The relevant quantity is
`P(wrong | verified)`, and 0.70 does not raise it far enough. Only the oracle
escapes, which puts the useful threshold for that particular problem somewhere
well above 0.70 — and locates it as a question about score quality, not about
the verifier.

---

## 4. Two things that stop this being over-read

### The winner's curse is large and visible

```
AUROC on the selection split   0.8748
AUROC on TEST                  0.6968
```

A **0.18 gap**, from choosing the best of 29 layers × 5 regularisation
strengths on 287 selection steps. Had the layer been chosen on test — the
obvious shortcut — this document would be reporting ~0.87 and claiming the
probe beats the PRM.

That is the entire reason `select_and_fit` does not take test indices as a
parameter, and why a test asserts its signature cannot grow one. The discipline
is not ceremony: it is worth 0.18 AUROC of wrongness here.

### The probe is data-starved, so 0.6968 is a floor

| training steps | AUROC (selection split) |
|---|---|
| 167 | 0.7374 |
| 335 | 0.8090 |
| 502 | 0.8453 |
| 670 | 0.8748 |

Monotone, steep, and **nowhere near a plateau**. ReProbe trains on far more than
670 steps. A better-trained probe on this corpus would score higher, and how
much higher is unknown.

So **0.6968 is a lower bound on what this signal class can do**, and every
conclusion in §2 and §3 should be read as "at this level of probe training."
The one conclusion that does *not* depend on it is the oracle's: even a perfect
score misses α = 0.05 by 1.8×, because at two calls per question over 5.15
steps the budget binds regardless of ranking.

---

## What changes in the thesis

**Chapter 7 gains a row and loses an ambiguity.** "The signal does not rank the
risk" becomes "the signals *measured there* do not; a probe on internal states
does, by +0.12 AUROC, and the gate still misses every binding α."

**Chapter 9's prediction was wrong in a specific, useful direction.** It
predicted ~0.9033; the measured value is 0.6968 with the training curve still
climbing. The prediction assumed ReProbe-scale training and got probe-scale
evidence from 670 steps, so the miss is attributable and the prediction is worth
re-running rather than retracting.

**The bottleneck ordering is unchanged and better evidenced.** Signal first,
budget second, verifier downstream of both.
