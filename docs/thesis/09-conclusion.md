# 9. Limitations, negative results, and conclusion

## 9.1 The negative results, collected

This thesis refuted more claims than it established, including most of its own.
Each refutation carries a regression test so it cannot quietly stop
reproducing.

| claim | source | outcome | evidence |
|---|---|---|---|
| Adaptive conformal under censored feedback is novel | this project's specification | **scooped** | CSA Thm E.1, a stronger anytime guarantee |
| Composite token-level uncertainty is the key signal | specification | **refuted** | AUROC 0.5589 |
| Semantic entropy at intermediate steps is the key signal | specification | **refuted** | AUROC 0.5740; combining buys 0.0002 |
| Influence weighting (descendant count) beats uniform | specification §4.2 | **refuted 5×** | loses to uniform on chains, 5 DAG families, 2 real corpora |
| H3: verify early beats verify late | specification | **refuted** | worst policy at every scope > 0; corr(position, error) = +0.950 |
| α = 0.10 is a workable target | specification | **infeasible** | 32.3% Kotte entry fee at μ = 0.3908 |
| StrategyQA is a suitable primary benchmark | specification | **wrong choice** | 72.9% of graphs one hop deep |
| The best structural signal is benchmark-dependent | this thesis, earlier draft | **overturned** | artifact of the extraction bug; depth wins on both |
| Corruption is near-absorbing | this thesis, ch. 4 | **generator-specific** | 95.9% on Mistral, 66.4% on Qwen |
| C3's `net = scope − FA` is the figure of merit | this thesis, ch. 5 | **corrected** | base-rate error; see §7.4 |
| Selective verification with a calibrated gate controls risk | the whole premise | **refuted** | selective risk misses α by 3× at every binding α |

The last row is the thesis. The design this project set out to build does not
work, and the reasons are three independent failures rather than one fixable
defect.

## 9.2 Threats to validity

**Generator mismatch, partly addressed.** The headline rates come from
Mistral-7B-SFT via Math-Shepherd. C1 has been reproduced on Qwen2.5-7B-Instruct
and comes out *higher*, with disjoint Wilson intervals, so the gap is not an
artifact of a weak generator. Two numbers do not transfer and are labelled
per-generator: μ (0.3908 → 0.1221) and corruption persistence (95.9% → 66.4%).
Neither has been measured on a third model.

**A "step" is not a model-invariant unit.** Math-Shepherd's step is one
calculator operation. Qwen writes 5.15 steps per solution of which 59% are
narration asserting no arithmetic at all. Per-step rates across generators are
rates over different units. This is a limitation of the unit of analysis, not
of the measurement, and it is not fixable by better extraction.

**Label semantics.** Math-Shepherd's `+`/`-` are automatic Monte-Carlo
estimates of "leads to a correct answer", not proofs. A lucky wrong step can be
labelled `+`, which makes measured global error a lower bound. The generated
corpus inherits the same estimator (K = 4), deliberately, so the two are
comparable.

**Constructed corpus.** Math-Shepherd is a PRM *training* set with a deliberate
class mix, so no unconditional rate can be read off it. Every number is
stratified and post-stratified. The generated corpus does not have this
problem — one solution per problem is an unbiased draw — which is why its
post-stratification is a sensitivity check rather than a correction.

**Derived graphs.** GSM8K dependencies are inferred from operand matching.
Hand-validated at **5.6% edge error** (16.5% in graphs containing an ambiguous
link, 4.2% elsewhere). That is a lower bound — dependencies routed through
unannotated solution lines are invisible to any operand-matching scheme — and
the adjudication was by LLM, not blind human annotation.

**Uncheckable steps.** 12.3% of Math-Shepherd steps and 59% of Qwen steps carry
no arithmetic. They are reported separately rather than assumed correct, but
every local rate is conditioned on checkability.

**Modelling versus measurement.** The propagation model's assumptions — full
repair on detection, i.i.d. per-step error — are known to be wrong. The i.i.d.
one is wrong in a direction that makes the allocation conclusions
*conservative*. Chapter 7's projected accuracy is pessimistic about false
alarms; the direction is isolated by the FA = 0 ablation, the magnitude is not
robust.

**Sample size in the end-to-end run.** 182 test questions, 940 test steps. The
headline gap is far outside sampling noise; the between-condition differences
are not.

**Single task family.** Everything measured here is grade-school arithmetic
word problems. Whether the local/global gap has the same size on code, on
multi-hop retrieval, or on agentic tool use is untested. The *mechanism* has no
obvious arithmetic dependence, but that is an argument, not a measurement.

## 9.3 What a practitioner should take from this

**Measure your base risk before choosing α.** It is two numbers and a closed
form, and it will frequently tell you the target you wrote down is either
infeasible or vacuous. Both failure modes look like success in a results table.

**Do not assume verifiers are interchangeable.** Calculator, retrieval, sandbox
and PRM are listed as alternatives in a great deal of system design. They span
0.00 to 0.90 detection rate on the population that matters. Which one you pick
is the design decision; the gate around it is not.

**Evaluate a verifier on the population you will deploy it against.** A
detection rate measured on a positive-only set is not a deployment number. With
84% of steps correct, a 10% false-alarm rate destroys more than a 90% detection
rate saves.

**Coverage is not risk.** A conformal guarantee will hold, and be reported as
holding, while the quantity you care about is untouched. Nothing in the
procedure warns you. Measure selective risk on the accepted set directly.

**Verify late, not early.** The intuition that early errors are cheaper to
catch is correct and the allocation conclusion that follows from it is wrong,
because later steps are measurably harder. If you must use structure, use
ancestor count.

## 9.4 Future work

**A better step-level signal is the bottleneck**, and two candidates were left
untested — both with a concrete, checkable prediction attached.

*Probes on frozen internal states.* ReProbe (Ni et al.) trains a sub-10M-
parameter probe that beats PRMs 150x larger. But its margin is largest **out of
domain**, and the strongest PRMs reach parity with it on GSM8K. The prediction
for this setup is therefore specific: a probe should land near the ch. 5 PRM's
0.9033, not above it — which would make the signal good enough to calibrate on
while leaving §7.4's false-alarm problem entirely intact. Ni et al. also find
probe and PRM combine better than either alone, so the productive object may be
a hybrid rather than a replacement.

*Embedding perturbation.* Wen et al. argue this reflects intermediate-step
uncertainty better than sampling-based agreement does. This thesis measured only
the family they argue against, so their alternative is untouched by the negative
result here — it is the cheapest remaining test of whether the AUROC 0.56 result
is about *these* signals or about step-level uncertainty generally.

A signal reaching even 0.75 would make the rest of this pipeline worth
rebuilding. Chapter 7 is a result about token-level and sampling-based signals,
not about all possible signals.

**Bidirectional entailment clustering.** Semantic divergence was measured under
one equivalence relation. The raw K = 5 samples are committed, so a
entailment-based relation can be evaluated with no GPU at all — and Chapter 7
shows the relation is load-bearing, so this is not a detail.

**Verifier calibration rather than verifier reach.** §7.4 suggests the
productive direction is not a higher-scope verifier but a *better-calibrated*
one: at 90% detection, halving the false-alarm rate is worth more than any
further detection gain.

**Verify against verified premises, under a budget.** ARES (You et al., §2.7)
detects propagated errors at 90.3% F1 by scoring each step solely against
*previously-verified* premises. That is the structural change C1 implies is
necessary, and it works. It also assumes something a budgeted gate cannot
supply: a verified prefix. Every verifier measured in Chapter 5 reads the
generator's uncorrected context, which is why they top out where they do.

What happens to ARES-style entailment scoring when only a fraction of the prefix
has been verified — and how the gate should choose *which* fraction to make the
conditioning set as useful as possible — is the most promising open question
this thesis can hand on. It also reframes Chapter 6: allocation would no longer
be about catching errors, but about building a trustworthy prefix to condition
later checks on, and ancestor count is a much more natural signal for that
objective than it is for the one tested here.

**Risk control under gate-induced dependence.** The theoretical problem remains
open. Barber et al. (§2.1) relax exchangeability for *exogenous* drift and for
asymmetric fitting algorithms; the violation here is endogenous — the gate's own
accept decision changes the data-generating process for later elements. Their
weighted-quantile machinery is the closest available starting point, not a
solution. This thesis characterises the phenomenon empirically and does not
solve it.

**Cross-domain replication.** The mechanism should not be arithmetic-specific.
Testing it on code or multi-hop retrieval is the clearest way to find out.

## 9.5 Conclusion

The thesis set out to build a calibrated gate for step-level verification and
instead measured why one does not work. The final statement:

> Conformal machinery applied to multi-step reasoning certifies whether a step
> is **locally valid** — that is what a verifier reports and therefore what can
> be calibrated. In multi-step reasoning that is not the quantity of interest:
> 78.5% of globally-wrong steps are arithmetically perfect on a weak generator,
> and **90.4% on a strong one**, so the gap widens rather than closing as
> models improve.
>
> The gap does not close by widening the verification window — arithmetic
> lookback saturates at 0.1999 because 80% of inherited corruption has no
> upstream arithmetic error at all. It does not close by asking the model to
> check itself, which detects zero errors. It closes only for a verifier that
> is both independent of the generator and specialised for the task, and
> *that* verifier is net-negative in deployment because its false-alarm rate
> acts on a population that is overwhelmingly correct.
>
> And the gate that would allocate its budget cannot be built, because neither
> token-level uncertainty (AUROC 0.5589) nor sampling-based semantic divergence
> (0.5740) ranks the risk, and combining them buys 0.0002. Split conformal
> calibration then holds its coverage guarantee while missing its selective
> risk target by a factor of three, and reports nothing amiss.
>
> Selective verification of LLM reasoning fails at the signal, at the
> calibration, and at the verifier. Repairing any one of them is not
> sufficient.

That is a claim about the design space rather than about one system, each leg
has a measured number and a regression test, and it is falsifiable in the only
way that matters: a signal that ranks step error would break it.

The project is more useful for having failed. A working gate on GSM8K would
have been one more entry in a crowded area. A measurement of *why* the obvious
design cannot work, with the three failure points separated and quantified, is
a result that transfers.
