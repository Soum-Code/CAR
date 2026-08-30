# CAR — Literature Positioning

Status: step 4 of the agreed workplan. Compiled 2026-08-31 from primary sources
(full PDFs read, not abstracts, except where noted).

**Headline: the contribution proposed on 2026-08-31 — adaptive conformal
calibration under censored feedback, fixed by Bernoulli forced exploration with
inverse-propensity weighting — is already published.** See §1. What survives is
narrower and is set out in §4.

---

## 1. The scoop: Conformal Selective Acting (CSA)

Khosravi & Huo, *Conformal Selective Acting: Anytime-Valid Risk Control for
RLVR-Trained LLMs*, [arXiv:2605.20270](https://arxiv.org/abs/2605.20270),
18 May 2026.

CSA controls **selective risk** — error rate among released outputs — with
anytime-pathwise validity, via a Ville-type e-process per candidate threshold on
a Bonferroni grid. Main bound:

```
R^act_T  <=  alpha + O( sqrt( log(1/delta) / N_T ) )     for all T,  w.p. >= 1-2delta
```

Algorithm 1 line 12 runs the verifier on **every** round, so the main method has
full feedback. The problem CAR was built around appears in the appendices:

**Theorem E.1 (Sparse-verifier validity and delay).** Subsample verifier labels
by predictable `B_t ~ Bernoulli(pi_t)` with `pi_t >= pi_min > 0`, and use the
**importance-weighted increment** `X̃_t(q) := (B_t / pi_t) · X_t(q)`. Then
(i) `Ẽ` remains a nonnegative supermartingale, (ii) all anytime-valid guarantees
hold verbatim, (iii) expected certification delay inflates by at most `1/pi_min`,
exactly `1/pi` when `pi_t = pi`.

That is Bernoulli exploration at rate pi with `1/pi` propensity weighting — the
`epsilon` / IPW scheme in `car/conformal/adaptive.py`, with a **stronger**
guarantee than the `O(1/sqrt(epsilon·T))` long-run rate we hypothesised.

They also cover the fully-censored case (Appendix B.5, "Partial evaluation"):
run the verifier only on acted rounds, updates restrict to `q <= q_deploy`,
higher thresholds never certify, "the controller stalls at the current threshold
but stays valid." That is our `naive` failure mode, characterised.

Empirical sparse-verifier results (MATH-200, alpha=0.40, T=2000):

| pi  | risk  | action rate | 1st cert | V-calls | delay |
|-----|-------|-------------|----------|---------|-------|
| 1.0 | 30.7% | 92.8%       | 70       | 2000    | 1.0x  |
| 0.5 | 29.1% | 86.1%       | 138      | 1000    | 2.0x  |
| 0.2 | 24.7% | 70.6%       | 312      | 405     | 4.5x  |
| 0.1 | 17.8% | 46.7%       | 705      | 199     | 10.1x |

**The one soft spot.** Validity survives but utility degrades hard: action rate
collapses 92.8% -> 46.7% as pi drops to 0.1, and risk falls to 17.8% against a
target of 40% — badly over-conservative. Uniform subsampling spends labels on
the reject region, where they carry no information about accept-region risk. A
scheme that concentrates the exploration budget on the accept region should
dominate this. That is a real but **incremental** opening, not a paper on its own.

### What CSA does not do

- Operates on **independent rounds** — one query, one answer, release or abstain.
  No intermediate steps, no dependency structure, no error propagation.
- Verifier assumed **deterministic**; imperfect/probabilistic verifiers are
  stated as open.
- Single scalar threshold. No notion that some items matter more than others.

---

## 2. The impossibility result — use it, do not fight it

Kotte, *When Can Conformal Risk Control Certify LLM Outputs?*,
[arXiv:2606.29054](https://arxiv.org/abs/2606.29054), 27 Jun 2026.

**Proposition 3.** When base risk `mu > alpha`, any distribution-free method
must abstain on at least

```
(mu - alpha) / (1 - alpha)
```

of examples. Closed form, checkable **before** running anything.

Direct consequence for CAR: this is a **hard floor on the verification rate**.
At a step error rate of `mu = 0.30` and `alpha = 0.10`, no gate — CAR's or
anyone's — can verify fewer than `(0.30-0.10)/0.90 = 22.2%` of steps. Any
reported result below that floor is a bug or a leak.

Also from that paper, and it is a warning about CAR's defaults:

- "hard NER/QA/CLS configurations are **uncertifiable at alpha = 0.10**;
  relaxing to alpha = 0.30-0.40 unlocks practical certification"
- ACI under cross-dataset shift cut risk-target violations 71% -> 21%, with
  "residual failures concentrated exactly where the impossibility bound predicts"

**Actions.** (a) Run the feasibility test as experiment zero and report `mu` per
dataset. (b) `alpha = 0.10` in `configs/default.yaml` is likely infeasible for
StrategyQA — expect to report a sweep, not a single alpha. (c) Add the
impossibility floor as a reference line on every accuracy-vs-cost plot; it makes
the Pareto argument much stronger than an unanchored curve.

---

## 3. Full comparison table

| Work | Gates what | On what unit | Trigger signal | Guarantee | Feedback model | Cost-aware | Propagation |
|---|---|---|---|---|---|---|---|
| **CSA** (2605.20270) | release vs abstain | whole output, indep. rounds | calibrated surrogate score | anytime-pathwise selective risk (e-process) | full; **sparse Bernoulli + IPW in Thm E.1** | verifier cost | no |
| **Kotte** (2606.29054) | abstain | whole output | nonconformity | CRC + **impossibility bound** | full | no | no |
| **ConfSpec** (2602.18447) | escalate draft->target model | reasoning step | draft-model confidence `p_D` vs hand-set `gamma` | none | full (target always answers) | latency (2.24x speedup) | no |
| **UHeads** (2511.06209) | nothing — scores only | reasoning step | trained head on frozen internal states | none | trained offline | ~free at inference | no |
| **Sherlock** (2511.00330) | verifier placement | workflow DAG node | **fan-in**, static, offline | none | full | budget `k` | topology-aware |
| **PRM line** (Lightman; Math-Shepherd; ProcessBench) | nothing — scores only | reasoning step | trained reward model | none | offline labels | no | no |
| **KnowNo / Ren 2023** | act vs ask for help | plan step | CP set size | marginal coverage | full | help cost | no |
| **CCPO** (2511.11828) | model/tool escalation | whole query | learned cost-aware policy | conformal constraint | full | 30% cost cut | no |
| **ARES** (2507.12948) | nothing — scores only | step, given verified premises | probabilistic soundness | no | — | no | **yes — propagated errors** |
| **Snowball** (2608.14588) | boundary gate at handoff | agent handoff | deterministic numeric match | none | full | ~zero (no LLM) | **yes — Markov model** |
| **CAR** (proposed) | continue / verify / abstain | **dependent step within a trajectory** | calibrated uncertainty **x downstream influence** | target: selective risk under dependence | **censored, exploration-corrected** | per-trajectory budget | **yes — objective** |

---

## 4. What actually survives

Ranked by how defensible each is after this review.

### 4.1 Dependent steps within a trajectory — STRONGEST

Every conformal gating paper above treats rounds as **independent items**. CSA's
supermartingale is over a filtration where round `t`'s verifier outcome does not
alter the correctness distribution of round `t+1`.

In CAR that is false by construction. Accepting a wrong step at `t=1` **changes
the distribution of steps `t=2..T`** — they are generated conditioned on a
corrupted premise. This is not a technicality; it is the phenomenon the project
is about, and it breaks the exchangeability *and* the independent-increment
structure that both CSA and CRC rely on.

Nobody has done risk control on a sequence where the gate's own accept decision
changes the data-generating process for later elements. That is a genuine open
problem, and CAR is naturally positioned on it.

**Risk:** it is also genuinely hard. It may resist a clean theorem. Plan for an
empirical characterisation with a theorem as upside, not the reverse.

### 4.2 Influence-weighted allocation — SURVIVES, with a caveat

`verification_value = P(wrong) x downstream_influence` still appears unclaimed
as a **calibrated, online** rule. Confirmed by the search: existing work
"optimizes scoring every step, not allocating a verification budget across steps
by expected downstream error reduction."

Nearest prior work is **Sherlock** ([arXiv:2511.00330](https://arxiv.org/pdf/2511.00330)),
which does treat verifier placement as a topology-plus-budget problem — but:

| | Sherlock | CAR |
|---|---|---|
| when | offline, before execution | online, during generation |
| topology | known workflow DAG | chain built as it is generated |
| criterion | fan-in, static ordering | uncertainty x descendants, calibrated |
| signal | none — pure structure | conformal score |

Cite Sherlock explicitly. The distinction is real but must be argued, not
assumed.

### 4.3 The impossibility floor as an evaluation instrument — CHEAP AND STRONG

Not novel (it is Kotte's), but nobody has used it as a **reference line for
selective verification in reasoning**. Adding it costs almost nothing and
converts the accuracy-cost plot from "our curve is above theirs" into "here is
the achievable region and here is where we sit in it."

### 4.4 Accept-region-concentrated exploration — INCREMENTAL

CSA's uniform Bernoulli subsampling wastes labels on the reject region and its
action rate collapses at low pi. Concentrating exploration on the accept region
should beat it. Worth a section; not worth a paper.

---

## 5. Dead, or badly weakened

| Claim from the original spec | Status |
|---|---|
| "Adaptive conformal under censored feedback is novel" | **Dead** — CSA Thm E.1 |
| The `O(1/sqrt(epsilon·T))` rate hypothesis | **Dead** — CSA proves a stronger anytime bound |
| "Uncertainty-triggered selective verification" as the contribution | **Crowded** — ConfSpec, UHeads, KnowNo, CCPO |
| "Semantic entropy at intermediate steps is the key signal" | **Contested** — [arXiv:2602.02427](https://arxiv.org/html/2602.02427) reports sampling-agreement methods "struggle to pinpoint the intermediate uncertainty"; UHeads beats Semantic Entropy as a baseline |
| Composite uncertainty as the novel score | **Weak** — UHeads (<10M params) matches PRMs 750-810x larger; a cheap trained head is the thing to beat |
| Verification before commitment as an architectural claim | **Partly taken** — Snowball's "verify before handoff, not before generation" |
| `alpha = 0.1` as a working target | **Probably infeasible** — see §2 |

---

## 6. Recommended re-pitch

> Existing conformal gating controls selective risk over **independent** items
> — one query, one answer, act or abstain. Multi-step reasoning breaks the
> assumption those methods rest on: accepting a wrong step does not merely incur
> local loss, it **changes the distribution of every step that follows**. CAR
> studies risk control when the gate's own decisions alter the data-generating
> process downstream, and shows that under this dependence the right allocation
> rule is not uncertainty but uncertainty weighted by downstream influence.

One problem, one mechanism, an evaluation anchored to a known achievable region.
The censored-feedback machinery stays in the system — it is needed and it works
— but it is cited to CSA, not claimed.

---

## 7. Reading still outstanding

- Sherlock (2511.00330) — full read. Now the closest prior work on 4.2.
- ARES (2507.12948) — propagated-error step evaluation; overlaps the propagation framing.
- *When does verification pay off?* (2512.02304) — cost-benefit, not yet read.
- CAP (Bao et al. 2025) — online selective conformal with FCR control; CSA compares against it.
- Barber et al. 2023, conformal beyond exchangeability — the formal route for §4.1.
