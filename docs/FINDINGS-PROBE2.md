# Probe, round two: it was not starved, and AUROC was the wrong target

Round one ([FINDINGS-PROBE.md](FINDINGS-PROBE.md)) reported AUROC **0.6968**
against Chapter 7's 0.5742 and left two claims resting on a learning curve.
This run settles both from one forward pass, and **one of them was wrong**.

Reproduce:

```bash
python scripts/gpu_probe_states.py --save-states runs/probe_states.npz  # GPU, ~15 min
python scripts/exp_probe_variants.py                                    # CPU, minutes
```

Round one wrote out only the per-step scores, which is the sole reason
answering any follow-up needed a second GPU session. The states are now saved
(fp16, 535 MB, gitignored — over GitHub's file limit), so every further probe
question is CPU-only.

---

## 1. The probe was not starved. C9c is refuted.

Round one reported the learning curve climbing **0.7374 → 0.8748** across
167 → 670 training steps and concluded that 0.6968 was *a floor on the signal
class, not a ceiling* (claim C9c).

**That curve was evaluated on the selection split** — the same data the layer
and the regularisation strength were chosen on. It is therefore optimistically
biased by the selection *and* measured on a different population from test.
`learning_curve(..., eval_idx=sel_idx, ...)` in the round-one call is where it
happened.

Re-measured on the held-out test split, with up to twice the training data:

| training steps | 136 | 340 | 680 | 1020 | 1361 |
|---|---|---|---|---|---|
| **test AUROC** | 0.6583 | 0.6901 | 0.6811 | 0.7043 | 0.6896 |

Flat and noisy between 0.66 and 0.70. Final slope **−0.043 AUROC per 1000
training steps**. And the head-to-head:

| probe | training steps | test AUROC | 95% CI |
|---|---|---|---|
| A, dev only (as published) | 670 | 0.6968 | [0.631, 0.756] |
| B, dev + calibration | 1361 | 0.6896 | [0.590, 0.768] |

**Doubling the training data does not move it.** C9c does not survive, and the
recommendation that followed from it — "generate more solutions" — was aimed at
the wrong constraint.

> The correct reading of 0.6968 is not "a floor, pending more data". It is
> roughly what a linear probe on this model's frozen states gives for this
> target on this corpus.

*Variant B saw the calibration split, so it is not gate-safe and Chapter 7 does
not quote it as a deployable score. It is reported here as a probe-quality
measurement only, which is all the learning-curve question needs.*

## 2. AUROC was the wrong training target, and training on the right one works

§7.4 measured that two scores of equal AUROC are worth different amounts,
because the propagation model only pays for the **first** globally-wrong step
in a solution — everything downstream inherits corruption a later repair does
not undo — and the probe's score correlates **+0.18** with step position while
the composite's correlates −0.28. The probe spends its ranking power on late
steps no repair can rescue.

Three ways at it, all on the pooled training set, all evaluated on test:

| variant | AUROC | AUROC vs first-bad | **first-bad recall** | score–position corr |
|---|---|---|---|---|
| B global target | 0.6896 | 0.5692 | 0.2250 | +0.163 |
| **C first-bad target** | 0.5735 | **0.6956** | **0.4500** | −0.472 |
| D position-residualised | 0.6811 | 0.6255 | 0.2750 | −0.066 |
| E selected on first-bad recall | **0.7060** | 0.6138 | 0.3500 | +0.208 |

First-bad recall is measured at a **fixed 19.1% verification rate** — the
probe's own measured rate in §7.4 — so no variant can win by flagging more
steps.

**Training on the first-bad label doubles the quantity that pays**: 0.2250 →
0.4500, a paired solution-clustered bootstrap difference of **+0.2232, 95% CI
[+0.056, +0.393], P(better) = 0.99**. The interval excludes zero.

And it costs global AUROC, 0.6896 → 0.5735. That is not a defect of the
experiment; it is §7.4's thesis made concrete. **These are different
objectives, and optimising one sacrifices the other.** A reader who ranks step
scores by AUROC would reject variant C as the worst of the four, and variant C
is the one that best does the job the system is for.

Variant D is the cleanest evidence for the mechanism: projecting step position
out of the states before fitting drops the score–position correlation from
+0.163 to −0.066 and raises first-bad recall 0.2250 → 0.2750, at almost no cost
in AUROC. The positional component is real, and it is not where the value is.

## 3. In the actual gate

Run through the real pipeline with the **gate-safe** probes — trained on dev
only, so the calibration split the conformal threshold uses stays clean:

| score | calls/q | selective risk | first-bad recall | PROJ accuracy |
|---|---|---|---|---|
| no gate | 0.00 | 0.1554 | — | **0.8022** |
| split conformal (token+semantic) | 1.09 | 0.1538 | 0.3590 | 0.7912 |
| probe, global target | 0.96 | **0.1394** | 0.3077 | 0.7802 |
| **probe, first-bad target** | 1.18 | 0.1600 | **0.3846** | **0.7912** |
| oracle | 0.37 | 0.0885 | 0.9231 | 0.9780 |

The first-bad-trained probe moves both quantities §7.4 predicts it should:
first-bad recall 0.3077 → 0.3846 and projected accuracy 0.7802 → 0.7912. It
also *worsens* selective risk, 0.1394 → 0.1600, because its global AUROC is
near chance and selective risk is defined over all accepted steps.

**It still does not clear the 0.8022 baseline.** So §7.4's diagnosis is
actionable — training against first-bad recall really does improve first-bad
recall — and acting on it at this corpus size does not make the gate worth
running.

## 4. The limit, stated rather than buried

**The corpus contains 108 first-bad steps. 40 are in test.**

That ceiling was computed before the run rather than discovered after it, and
it is what separates the two configurations:

| configuration | training positives | first-bad recall gain | 95% CI | P(better) |
|---|---|---|---|---|
| pooled (dev + calibration) | 68 | **+0.2232** | [+0.056, +0.393] | **0.99** |
| gate-safe (dev only) | 27 | +0.0812 | [−0.108, +0.278] | 0.76 |

The significant result is the one that saw the calibration split. The
deployable one points the same way and its interval spans zero. Both are
reported; neither is presented as the other.

So the honest status of §2 is: **the effect is real at 68 training positives
and unproven at 27**, and the way to settle it is more first-bad steps, which
means more generated solutions — the thing §1 just showed would *not* help the
global-target probe. The two gaps want opposite things from the same budget,
and that is a useful thing to know before spending it.

## 5. What changes in the thesis

| | before | after |
|---|---|---|
| C9c, "0.6968 is a floor, not a ceiling" | learning curve 0.7374 → 0.8748, no plateau | **refuted** — that curve was on the selection split; the held-out curve is flat at 2× the data |
| Ch. 9 future work, "train the probe properly" | the cheapest open question | **answered, negatively** — more data does not move it |
| §7.4, "AUROC is not a sufficient figure of merit" | a diagnosis | **actionable** — training on first-bad recall doubles it, CI excludes zero |
| Figure 7.6 right panel | the selection-split curve, "still climbing" | the held-out curve, flat |

See [FINDINGS-PROBE.md](FINDINGS-PROBE.md) for round one and
[FINDINGS-SCORE-QUALITY.md](FINDINGS-SCORE-QUALITY.md) for the §7.4 result this
acts on.
