# Probe, round two: C9c's evidence was invalid, and AUROC is the wrong target

Round one ([FINDINGS-PROBE.md](FINDINGS-PROBE.md)) reported AUROC **0.6968** and
left two claims resting on a learning curve. This settles the first negatively
— not by showing the claim false, but by showing its evidence never supported
it — and gives the second a real answer.

> **This document has itself been corrected once.** Its first version claimed
> "doubling the data does not move it, C9c refuted". That comparison confounded
> training size with re-running layer selection, and its learning curve
> confounded size with split composition. Both are fixed below, and the
> correction reverses the sign. The errors are described rather than deleted,
> because they are the same class of mistake C9c itself was.

Reproduce:

```bash
python scripts/gpu_probe_states.py --save-states runs/probe_states.npz  # GPU, ~15 min
python scripts/exp_probe_variants.py                                    # CPU, ~20 min
```

---

## 1. C9c is unsupported — which is not the same as refuted

Round one reported the learning curve climbing **0.7374 → 0.8748** over
167 → 670 training steps and concluded 0.6968 was *a floor, not a ceiling*
(claim C9c).

**That curve was evaluated on the selection split** — the same data the layer
and the regularisation strength were chosen on. It is biased by that selection
and measured on the wrong population. So C9c never had evidence. The question
of whether it is *true* is separate, and needs measuring properly.

### The clean test: hold the configuration fixed

Comparing two independently *selected* probes does not measure data quantity.
It cannot: re-running selection on a bigger training set can land on a
different layer, and then the difference is partly the layer. It did — the
dev-only probe selects layer 25 and the pooled one selects layer 28.

Holding layer and C fixed and only changing how much data the probe sees:

| configuration | 670 → 1,361 steps | change | 95% CI |
|---|---|---|---|
| layer 25, C = 0.1 (the published probe) | 0.6968 → **0.7086** | **+0.0105** | [−0.035, +0.057] |
| layer 28, C = 0.1 | 0.6817 → 0.6896 | +0.0072 | [−0.041, +0.055] |

**The sign is positive at every configuration, and both intervals span zero.**

### And the learning curve, with the composition confound removed

The first version built the pooled training set as `concat(dev_shuffled,
calibration_unshuffled)`, so walking its prefixes changed *who* was in training
as well as how many: calibration share ran 0.00 → 0.00 → 0.015 → 0.34 → 0.51
while the wrong-step rate drifted 0.213 → 0.177. The "slope" was measured
exactly where the population moved most. Shuffling the pool fixes it:

| training steps | 136 | 340 | 680 | 1020 | 1361 |
|---|---|---|---|---|---|
| test AUROC | 0.5686 | 0.6691 | 0.6590 | 0.6856 | 0.6896 |
| calibration share | 0.53 | 0.53 | 0.50 | 0.51 | 0.51 |

End to end the slope is **+0.099 AUROC per 1000 steps**. The curve climbs,
and most of the climb is the rise out of a very small sample rather than
anything the doubling buys: the fixed-configuration test above is the
measurement to quote, not any single pair of points on this curve. (The
680 → 1361 segment reads +0.031 raw AUROC, but 680 is one random prefix of the
pool — resampling 680-step subsets at the same layer and C gives 0.6810 ±
0.0216 — so that segment is a draw, not an effect.)

### So what is the honest status?

| | |
|---|---|
| Round one's evidence for C9c | **invalid** — scored on the selection split |
| Is more data worth anything? | Yes, a little: +0.01 at fixed config, +0.03 across the doubling, intervals spanning zero |
| Is 0.6968 "a floor"? | **Unproven either way.** Directionally supported, nowhere near the 0.7374 → 0.8748 trajectory the leaked curve implied |
| Should C9c be marked refuted? | **No — unsupported.** Refuting it needs evidence more data does *not* help, and the point estimates are positive |

The practical reading: doubling the training data buys roughly a hundredth of
AUROC. Generating more solutions is not the lever the leaked curve made it look
like, and it is not worthless either.

## 2. AUROC is the wrong training target

§7.4 measured that two scores of equal AUROC are worth different amounts,
because the propagation model only pays for the **first** globally-wrong step
in a solution, and this probe's score correlates positively with step position —
it spends its ranking power on late steps no repair can rescue. (§7.4 reports
**+0.1813** and the table below **+0.2327**. Same probe, same 925 test steps,
Spearman 1.0 between them: the two committed artifacts store different monotone
transforms of one score, and Pearson is not invariant to that. AUROC and
first-bad recall are, which is why the comparisons rest on those.)

Variants B–E are trained on the pooled set and A is the dev-only published
probe; all are evaluated on test with first-bad recall measured at a **fixed
19.1% verification rate**, so nothing wins by flagging more:

| variant | AUROC | **first-bad recall** | score–position corr |
|---|---|---|---|
| A global target, dev only (published) | 0.6968 | 0.3000 | +0.233 |
| B global target, pooled | 0.6896 | 0.2250 | +0.163 |
| **C first-bad target** | 0.5735 | **0.4500** | −0.472 |
| D position-residualised | 0.6811 | 0.2750 | −0.066 |
| E selected on first-bad recall | **0.7060** | 0.3500 | +0.208 |

Training on the first-bad label raises the quantity that pays and costs global
AUROC. At 0.5735 it lands level with the 0.5742 token+semantic baseline whose
failure is Chapter 7's central negative result — a 0.0007 gap, far inside the
interval on either number, so the honest statement is that **the probe's AUROC
advantage is gone**, not that it is measurably worse than the baseline. §7.2
declines to rank a 0.0114 gap for exactly this reason and the same restraint
applies here.

### Against which comparator?

B has the lowest first-bad recall of any global-target probe here, so quoting C
against B alone picks the flattering baseline. Against both:

| comparison | gain | 95% CI | P(better) | |
|---|---|---|---|---|
| C vs **B** (pooled global) | +0.2216 | [+0.051, +0.390] | 0.99 | excludes zero |
| C vs **A** (the deployed probe) | +0.1412 | [−0.065, +0.333] | 0.90 | **spans zero** |

**The effect is directionally consistent and significant against only one of
two reasonable comparators.** A reader has the published probe A in mind, and
against A it is 0.3000 → 0.4500 on the point estimates (+50%) with a bootstrap
interval that includes no effect. "Doubles" is true of C vs B and not of
C vs A.

One caveat that does *not* apply: variant C is the best of 145 configurations
chosen on a selection split holding just 19 first-bad positives, which is where
a winner's curse would live. Its selection-split recall is 0.4211 against
0.4500 on test — test is higher, so no curse is visible here.

Variant D is the mechanism check: projecting step position out of the states
drops the score–position correlation from +0.163 to −0.066 and moves first-bad
recall 0.2250 → 0.2750. That movement is two test steps and its interval spans
zero, and D still sits *below* the un-residualised published probe A. It
supports the mechanism weakly; it is not evidence that residualising produces a
better probe.

## 3. In the gate, at honestly unmatched budgets

§2's comparisons are budget-matched by construction. The gate is not — each
score picks its own operating point through the conformal threshold — so the
verification rate is reported alongside, and so is recall per call:

| score | calls (calls/q) | selective risk | first-bad recall | per call | PROJ accuracy |
|---|---|---|---|---|---|
| no gate | 0 (0.00) | 0.1554 | — | — | **0.8022** |
| split conformal (token+semantic) | 199 (1.0934) | 0.1538 | 0.3590 | **0.3283** | 0.7912 |
| probe, global target | 175 (0.9615) | **0.1394** | 0.3077 | 0.3200 | 0.7802 |
| probe, first-bad target | 214 (1.1758) | 0.1600 | **0.3846** | 0.3271 | 0.7912 |
| oracle | 67 (0.3681) | 0.0885 | 0.9231 | — | 0.9780 |

The first-bad probe has the highest raw first-bad recall and it spends 23% more
calls to get it. Per verification call the ordering reverses and the existing
token+semantic baseline comes out top — but by **0.0012**, which is about one
verification call, so that reversal is a tie rather than a result. What is not
a tie: the baseline reaches the same 0.7912 projected accuracy on *fewer* calls
(199 against 214). Against what the chapter already had, the first-bad probe is
not an improvement.

The two moved quantities are also small in absolute terms: 0.3077 → 0.3846 is
3 first-bad steps of 39, and 0.7802 → 0.7912 is 2 questions of 182.

## 4. The limit, and it is the label

**108 first-bad steps exist in the corpus. 40 are in test.** Training positives:
27 dev-only, **49** pooled (the other 19 sit in the selection split).

| configuration | training positives | gain vs its comparator | 95% CI |
|---|---|---|---|
| pooled (not gate-safe) | 49 | +0.2216 vs B | [+0.051, +0.390] |
| gate-safe (dev only) | 27 | +0.0805 | [−0.119, +0.262] |

The significant number comes from the configuration that saw the calibration
split; the deployable one points the same way and its interval spans zero.
Neither stands in for the other.

## 5. What this changes in the thesis

| | before | after |
|---|---|---|
| C9c "0.6968 is a floor" | asserted, on a selection-split curve | **unsupported** — evidence invalid; a clean test gives +0.01, CI spanning zero |
| "doubling the data does not move it" | this document, first version | **withdrawn** — confounded with layer re-selection; the sign is positive |
| §7.4 "AUROC is not a sufficient figure of merit" | a diagnosis | **actionable, partially demonstrated** — significant against one comparator, not the other |
| the gate table | probe rows only | the token+semantic baseline restored, with per-call recall |

See [FINDINGS-PROBE.md](FINDINGS-PROBE.md) for round one and
[FINDINGS-SCORE-QUALITY.md](FINDINGS-SCORE-QUALITY.md) for the §7.4 result this
acts on.
