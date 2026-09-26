# The instrument is not identifiable, because layer choice is not resolvable

Chapter 7.6 varied two things about the probe and never varied the
**instrument**: every probe in this thesis is a logistic regression on one
layer, and the ReProbe prediction that a probe here would reach ~0.9033 was
made about a setting whose probes are not linear.

This varies it, and the answer is a null with a reason attached. The reason is
more useful than the null.

> **This document was wrong in draft and never published.** Its first version
> reported "+0.0250, the largest of three levers" by comparing each arm's own
> best layer — instrument *plus* layer re-selection, the identical confound
> that `fixed_config_doubling` exists to prevent for the data lever and that
> this section had already corrected twice. An adversarial verification pass
> caught it before commit. The matched-layer numbers are below; the confounded
> one is kept in the artifact so the difference is visible.

```bash
python scripts/exp_probe_nonlinear.py          # CPU, ~20 min
```

No GPU: `runs/probe_states.npz` already holds the frozen states.

---

## 1. Matched at a fixed layer, there is no instrument effect

The only comparison that isolates the instrument holds the layer fixed and
lets each arm choose its own hyper-parameters on the selection split:

| training set | layer | linear | non-linear | difference | 95% CI | |
|---|---|---|---|---|---|---|
| dev (gate-safe) | 25 | 0.6968 | 0.7024 | +0.0051 | [−0.031, +0.042] | spans zero |
| dev (gate-safe) | 26 | 0.6897 | 0.7222 | +0.0315 | [−0.007, +0.070] | spans zero |
| pooled | 28 | 0.6896 | 0.7195 | +0.0300 | [−0.002, +0.066] | spans zero |
| pooled | 19 | **0.7645** | 0.7331 | **−0.0315** | [−0.051, −0.015] | **favours linear** |

**The point estimates run −0.0315 to +0.0315. The sign depends on which layer
is held fixed.** Three of four intervals span zero, and the only one that does
not favours the *linear* probe.

At the published probe's own layer — 25, the configuration Chapter 7.6 actually
reports — the instrument is worth **+0.0051, CI [−0.031, +0.042]**. That is
*half* the data lever (+0.0105), not a multiple of it.

## 2. Why: the selection split cannot resolve layers

| training set | selection spread over the top 5 layers | their test spread | ratio |
|---|---|---|---|
| dev | 0.0152 | 0.0285 | 1.9× |
| pooled | 0.0173 | **0.0750** | **4.3×** |

287 selection steps separate the candidate layers by at most 0.017, while their
held-out AUROCs differ by up to 0.075. The argmax is therefore close to
arbitrary, and any instrument effect of ±0.03 is smaller than the noise in the
thing being held constant.

The pooled arm shows exactly what that costs:

```
layer 28   selection 0.8667   test 0.6896   <- chosen
layer 19   selection 0.8655   test 0.7645
```

**A 0.0012 margin on the selection split bought a 0.0749 loss on test.** The
linear probe's own layer choice was worse than the alternative it rejected, by
thirty times the margin it was rejected on.

> The probe series has reached the resolution limit of this corpus. The binding
> constraint is not which probe, or how much training data, or what target — it
> is that 287 selection steps and 925 test steps cannot separate effects of
> this size.

## 3. What the confounded comparison looked like, and why it is kept

Comparing each arm's own argmax gives +0.0250 (dev) and +0.0452 (pooled), both
spanning zero. Those numbers are in the artifact under `delta_auroc` and they
are *not* the instrument effect: the arms land on different layers (25 vs 26,
28 vs 19), so the delta carries layer re-selection.

The pooled row is the clearest illustration. Its +0.0452 is almost entirely the
linear arm's unlucky layer draw — matched at layer 19 the same comparison is
−0.0315 in the *opposite* direction, with the interval excluding zero.

The draft additionally leaned on two things that do not hold:

- **"The sign replicates at both training sizes."** The pooled training set is
  a strict superset of the dev one, the selection and test splits are identical,
  and the bootstrap resamples the same solutions. The second row cannot
  independently confirm the first.
- **"Lifting the layer restriction changed nothing, so the handicap did not
  bind."** If best-of-80 equals best-of-464 then the reported model *is* the
  argmax of a 464-configuration search, so the restriction supplied no
  protection against the curse. The check refutes the safeguard rather than
  validating it.

## 4. Seed sensitivity

`MLPClassifier(random_state=)` is a nuisance parameter the first draft never
varied. Holding the split, grid and selection rule fixed and changing only it:

| seed | argmax layer | test AUROC | delta vs linear@25 |
|---|---|---|---|
| 0 | 26 | 0.7222 | +0.0250 |
| 1 | 25 | 0.6987 | +0.0013 |
| 2 | 25 | 0.7232 | +0.0262 |
| 3 | 27 | 0.7045 | +0.0061 |

The headline was one draw of four, and the 0.0065 by which its interval missed
excluding zero is well inside this spread.

## 5. What this changes in the thesis

| | before | after |
|---|---|---|
| §7.6 "the untested variable is the instrument" | an argument for future work | **tested, and null** — no matched-layer comparison separates the instruments |
| the proposed C13, "the instrument is the largest lever" | drafted | **withdrawn before publication** — it was layer re-selection |
| ch. 9, the ReProbe prediction | missed, instrument excuse open | the excuse is closed: at matched layer the instrument is worth +0.005, against the +0.21 the prediction needs |
| what limits the probe series | open | **the corpus** — a 0.0012 selection margin moves test AUROC by 0.0749 |

The honest summary of the three levers is now that **none of them is
identifiable at 925 test steps**, and that the measurement which *is* solid is
the negative one: nothing about the probe — its training data, its target, or
its functional form — closes the gap to the ch. 5 PRM's 0.9033.

See [FINDINGS-PROBE.md](FINDINGS-PROBE.md) for round one and
[FINDINGS-PROBE2.md](FINDINGS-PROBE2.md) for the data and target levers.
