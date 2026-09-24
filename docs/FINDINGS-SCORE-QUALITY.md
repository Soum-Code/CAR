# How good does the score have to be? The crossing is at AUROC ≈ 0.65

Chapter 7 ends with a verifier that detects 90.3% of globally-wrong steps and
is still a *net loss* — behind the measured score it costs 4 points of
projected accuracy, behind the probe it costs 2, and behind an oracle it gains
17.6. So there is a score quality at which the task PRM stops being a
liability, and two places in the draft could only say "somewhere above 0.70 and
unmeasured".

This measures it, and the answer is **lower than the draft guessed** — which
matters, because it puts the threshold within reach of a signal that already
exists rather than beyond it.

Reproduce:

```bash
python scripts/exp_score_quality_threshold.py --seeds 64
```

CPU, about 70 seconds, deterministic. Writes `runs/score_quality_threshold.json`.

## Method

`SyntheticAUROCScorer` makes score quality a dial. Under the binormal model a
globally-wrong step draws its score from N(d, 1) and a correct step from
N(0, 1), so

```
AUROC = P(S_wrong > S_correct) = P(N(d, 2) > 0) = Phi(d / sqrt(2))
d     = sqrt(2) * Phi^-1(AUROC)
```

Validated against 200k draws at five targets; the realised AUROC matches the
requested one to four decimals.

Everything else is held fixed and imported from `exp_gate_pipeline.py` rather
than reimplemented — same corpus, same hash splits, same split-conformal
calibrator, same 2-call budget, same verifier at its measured scope and
false-alarm rate. If this script ran the loop its own way its rows would not be
comparable to the Chapter 7 tables, which is the entire point. 64 seeds per
grid point; the confidence intervals below are over seeds.

---

## 1. The crossing

α = 0.30, task PRM at its measured 9.87% false-alarm rate. Baseline (no gate)
is **0.8022**.

| AUROC | verify % | sel. risk | recall | 1st-bad recall | PROJ acc | 95% CI | |
|---|---|---|---|---|---|---|---|
| 0.550 | 22.5% | 0.1536 | 0.2106 | 0.2853 | 0.7782 | [0.7725, 0.7840] | net − |
| 0.600 | 22.9% | 0.1489 | 0.2354 | 0.3413 | 0.7879 | [0.7818, 0.7939] | net − |
| 0.625 | 23.1% | 0.1463 | 0.2500 | 0.3786 | 0.7940 | [0.7880, 0.8001] | net − |
| 0.650 | 23.3% | 0.1441 | 0.2600 | 0.4111 | 0.8010 | [0.7953, 0.8067] | ~same |
| 0.675 | 23.4% | 0.1425 | 0.2685 | 0.4375 | 0.8055 | [0.7993, 0.8116] | ~same |
| **0.700** | 23.6% | 0.1405 | 0.2787 | 0.4635 | **0.8119** | [0.8057, 0.8181] | **net +** |
| 0.750 | 23.9% | 0.1368 | 0.2943 | 0.5120 | 0.8219 | [0.8156, 0.8281] | net + |
| 0.800 | 24.1% | 0.1335 | 0.3095 | 0.5753 | 0.8341 | [0.8282, 0.8400] | net + |
| 0.900 | 24.4% | 0.1285 | 0.3370 | 0.7039 | 0.8566 | [0.8506, 0.8627] | net + |
| 0.990 | 24.6% | 0.1258 | 0.3491 | 0.7736 | 0.8712 | [0.8650, 0.8775] | net + |

**Crossing at AUROC ≈ 0.657 on the means.** At 95% the sweep is net-negative up
to 0.625 and net-positive from 0.700; between those two it cannot separate the
gated system from doing nothing, so the honest statement is an interval rather
than a point.

The draft's guess of "well above 0.70" was **wrong in the direction that
matters**. The threshold is roughly 0.65, and the probe measured 0.6968 — it is
sitting on the boundary, not far below it.

## 2. The crossing exists only because of false alarms

Same sweep, the verifier's false-alarm rate switched off:

| AUROC | 0.550 | 0.650 | 0.750 | 0.900 | 0.990 |
|---|---|---|---|---|---|
| PROJ acc, FA = 0.0987 | 0.7782 | 0.8010 | 0.8219 | 0.8566 | 0.8712 |
| PROJ acc, FA = 0 | 0.8589 | 0.8844 | 0.9045 | 0.9421 | 0.9563 |

With false alarms off the verifier is net-positive at **every** score quality
tested, including 0.55 — a score barely better than a coin. There is no
crossing to find.

> The threshold is not a property of detection. A verifier that never breaks a
> correct step is worth having however badly it is aimed; the whole question of
> "how good does the score need to be" is created by the false-alarm rate.

This is the sharpest available statement of §7.4's correction, and it is the
one a practitioner can act on: **halving a verifier's false-alarm rate moves
the score quality you need more than raising its detection rate does.**

## 3. No score quality holds a binding α at this budget

α = 0.05, the same verifier:

| AUROC | 0.550 | 0.700 | 0.850 | 0.950 | 0.990 |
|---|---|---|---|---|---|
| selective risk | 0.1531 | 0.1399 | 0.1193 | 0.1024 | **0.0956** |

**Never held.** The best any score quality manages is 0.0956 at AUROC 0.99 —
still **1.9× the target**. §7.3 showed this at one point, using a binary
oracle; this shows it across the whole range, and rules out the reading that
the oracle's failure was an artifact of its degenerate score distribution.

Score quality is not the binding constraint on risk control. The budget is.
Two verification calls over a mean of 5.15 steps leaves most globally-wrong
steps unverified however perfectly they are ranked.

---

## 4. The unplanned result: AUROC is not a sufficient description of a score

Sweep A lets the calibrator choose its own operating point, so a better score
changes both the ranking *and* the number of calls. Pinning verification at
19.1% — the probe's measured rate — isolates the ranking. At that matched
budget:

| | AUROC | calls/q | recall | 1st-bad recall | PROJ acc |
|---|---|---|---|---|---|
| synthetic score | 0.6974 | 0.79 | 0.2416 | **0.3413** | **0.8206** [0.8154, 0.8258] |
| **probe, layer 25** | 0.6968 | 0.96 | 0.2465 | **0.3077** | **0.7802** |

The probe sits at the **0th percentile of 64 synthetic draws** on the same 182
test questions. It catches slightly *more* wrong steps and rescues *fewer*
answers.

### The mechanism, measured rather than inferred

```
corr(normalised step position, probe score)      +0.1813
corr(normalised step position, composite score)  -0.2766

among globally-wrong steps only:  probe +0.2030,  composite -0.2242
```

**The probe flags late steps. The composite flags early ones.** Chapter 6
measured local error rate rising with position — `corr(position, local error)
= +0.950`, doubling from 11% at step 1 to 22% at step 8 — so a score that
chases positional difficulty is rewarded on AUROC. But the projection only pays
for the **first** bad step in a solution, because everything downstream inherits
corruption that repairing a later step does not undo (C2, and 0 recoveries in
25,971 solutions).

So the probe's AUROC advantage is spent in the wrong place. `SyntheticAUROCScorer`
draws independently per step, making its position correlation zero by
construction, which is exactly why it converts its AUROC into accuracy more
efficiently than a real score of the same quality does.

> Two scores with identical AUROC are worth different amounts. What a
> step-level score is worth depends on **which** errors it ranks highly, and
> under a propagation objective the ones that count are the earliest.

### What this costs the rest of the thesis

It does not overturn anything, and it sharpens two things:

1. **The crossing in §1 above is an estimate for a well-behaved score.** A real
   score of AUROC 0.65 with a positional bias will land below the synthetic
   curve, so 0.65 is a floor on the requirement, not a specification.
2. **Chapter 7's AUROCs are the right measurement of the wrong quantity.**
   They are comparable to each other and to the literature, and they are not
   sufficient to predict what a score is worth in the loop. First-bad-step
   recall is closer to the quantity that matters.

### The honest limits

**First-bad recall is coarse.** Only **40 of 182** test trajectories contain a
globally-wrong step at all, so the denominator is 39–40 and any single pair of
rows differs by two or three trajectories. The claim rests on the monotone
trend across 13 grid points × 64 seeds (0.2853 → 0.7736), not on the
probe-vs-synthetic pair in isolation.

**The binormal draw is a shape, not a signal.** A real score need not be
binormal, and this experiment cannot say what a real score of a given AUROC
would do beyond "at least as badly as the synthetic one, if it has structure
the projection does not reward".

**Projected accuracy is MODELLED throughout**, with the same assumptions §7.5
sets out. Selective risk, verification rate, recall and first-bad recall are
measured.

---

## 5. What to take from this

| question | answer |
|---|---|
| Where does the PRM stop being a liability? | AUROC ≈ 0.65 (net − to 0.625, net + from 0.700) |
| Is that reachable? | The probe is at 0.6968 — on the boundary already |
| Why is there a threshold at all? | Entirely the 9.87% false-alarm rate; with FA = 0 there is none |
| Can a better score hold α = 0.05? | No. Best is 0.0956 at AUROC 0.99, 1.9× the target |
| Is AUROC the right target to optimise? | **No.** First-bad-step recall is what the objective pays for |

See also [FINDINGS-PROBE.md](FINDINGS-PROBE.md) for the score this compares
against, and [FINDINGS-PIPELINE.md](FINDINGS-PIPELINE.md) for the gate result
it extends.
