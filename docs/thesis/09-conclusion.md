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
| The three failures are independent | this thesis, ch. 7 first draft | **partly wrong** | the oracle shows the verifier result is downstream of the score |
| Internal states do not carry a usable step signal | implied by ch. 7's first draft | **refuted** | a probe reaches AUROC 0.6968 against 0.5742, and improves the gate |
| A probe here will land near the PRM's 0.9033 | this thesis, ch. 9 prediction | **missed** | 0.6968 measured; the flat held-out learning curve removes the training-scale excuse |
| 0.6968 is a floor, pending more training data | this thesis, ch. 7.6 first draft | **refuted** | that curve was scored on the selection split; doubling the data gives 0.6896 |
| The verifier needs a score "well above 0.70" | this thesis, ch. 7 earlier draft | **wrong, and low** | the crossing is AUROC ≈ 0.65; the probe already clears it |
| AUROC is the figure of merit for a step score | implicit everywhere in chs. 7 and 9 | **refuted** | equal-AUROC scores differ by 0.04 projected accuracy; what counts is first-bad-step recall |
| The cheap equivalence relation was hiding the signal | the obvious objection to §7.2 | **refuted** | entailment 0.5625, CI [0.512, 0.614], across a 0.1%–78% permissiveness range |

The row refuting *selective verification with a calibrated gate controls risk*
is the thesis. The row below it is the correction the oracle baseline forced: of the three failure points, only two are separate. The
verifier's net-negative result is a consequence of aiming it with a near-chance
score, and disappears when the score is perfect. What remains genuinely
independent is the **signal** and the **budget**.

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

**Sample size in the end-to-end run.** 182 test questions, 925 test steps. The
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

**Evaluate a verifier on the population your gate will actually send it.** A
detection rate measured on a positive-only set is not a deployment number — and
neither is one measured over the whole population. What matters is
`P(wrong | verified)`, and the *score* sets that. The same PRM at the same 9.87%
false-alarm rate is worth −4 points of accuracy behind a near-chance score and
+17.6 behind a perfect one. A verifier is not good or bad; it is well or badly
aimed.

**Do not tune a step-level score on AUROC.** It is the natural metric and it is
not the quantity a reasoning system is paid on. Under error propagation only the
*first* bad step in a solution can be repaired, so a score that ranks late
errors highly scores well on AUROC and rescues nothing. Measured here: a trained
probe at AUROC 0.6968 converts to 0.7802 projected accuracy where a synthetic
score of the same AUROC reaches 0.8206, because the probe's score correlates
+0.18 with step position and the synthetic one 0.00. Rank by AUROC if you must
compare to the literature; select on first-bad-step recall.

**Coverage is not risk.** A conformal guarantee will hold, and be reported as
holding, while the quantity you care about is untouched. Nothing in the
procedure warns you. Measure selective risk on the accepted set directly.

**Verify late, not early.** The intuition that early errors are cheaper to
catch is correct and the allocation conclusion that follows from it is wrong,
because later steps are measurably harder. If you must use structure, use
ancestor count.

## 9.4 Future work

**A better step-level signal is the bottleneck.** Two candidates were named here
with a concrete, checkable prediction attached to each. One has since been run,
and the result is recorded below alongside the prediction it missed.

*Probes on frozen internal states — now tested, and worth continuing.* §7.6
reports AUROC **0.6968**, +0.12 over everything else measured here, improving
selective risk at every α on fewer calls. The prediction that it would land near
0.9033 was wrong, and an earlier draft excused it as probe-scale training on
670 steps against ReProbe's far larger sets. That excuse is gone: §7.6 doubles
the training data and the held-out AUROC does not move (0.6968 → 0.6896, final
slope −0.043 per 1000 steps). The first curve had been scored on the selection
split. So either the prediction was wrong about this setting, or a *linear*
probe on frozen states is the wrong instrument — ReProbe's are not linear, and
that is the untested half. Ni et al. also find probe and PRM combine better
than either alone, so the productive object may still be a hybrid.

The more useful finding is what 0.70 was *not* enough for. It did not make the
gate hold any binding α, and it did not make the PRM worth having — projected
accuracy 0.7802, still under the 0.8022 baseline, where the oracle reaches
0.9780.

§7.4 then locates the threshold that earlier drafts could only gesture at. A
9.87%-false-alarm verifier turns from liability into gain at **AUROC ≈ 0.65**,
net-negative to 0.625 and net-positive from 0.700 — *below* the probe's own
0.6968. Two things follow, and they cut in opposite directions. The signal
bottleneck is nearly closed already, which is more encouraging than this thesis
expected. And it is closed faster by fixing the verifier's false alarms than by
improving the score at all: at FA = 0 the PRM is worth having at every score
quality down to 0.55.

*Embedding perturbation.* Wen et al. argue this reflects intermediate-step
uncertainty better than sampling-based agreement does. This thesis measured only
the family they argue against, so their alternative is untouched by the negative
result here. The probe has already settled the general question — step-level
uncertainty *is* readable — so what embedding perturbation would add is a
cheaper route to it, one that needs no labelled training steps at all.

Chapter 7 is a result about token-level and sampling-based signals, not about
all possible signals, and §7.6 is the demonstration of that. But §7.4 also
changes what "better" should mean. The target is not a higher AUROC: the
verifier already turns positive at ≈ 0.65, and no AUROC at all holds α = 0.05
at this budget. The target is a score that ranks the **earliest** bad step
highly, because that is the only one a repair can rescue. A signal evaluated on
AUROC alone can improve on that metric while getting worse at the thing the
system is for — which is what the probe did.

**Bidirectional entailment clustering — run, and it closed negatively.** This
was listed here as the cheapest untested alternative, on the grounds that
Chapter 7 shows the relation is load-bearing. §7.2 now reports it: the
reference relation scores **0.5625** on test, 95% CI [0.512, 0.614], on the
same 12,865 generations. It does not find signal the cheap relation missed. The
0.0114 gap to numeric equivalence is *not* a ranking — its interval spans zero
— and the claim worth making is the other one: the three relations nest on a
single permissiveness axis, calling 0.1%, 31.9% and 78.0% of pairs equal, and
across that entire range the measurement does not move.

The relation is also the wrong tool, which is worth recording separately from
the result. An MNLI model judges 86.6% of pairs entailing and merges steps
asserting different quantities — it checks whether two sentences are about the
same thing, not whether they compute the same number. A relation that *is*
sensitive to the asserted quantity, and that also handles the 59% of steps
carrying no arithmetic, remains unbuilt.

**Verifier calibration rather than verifier reach.** §7.4 no longer merely
suggests this, it measures it. The score quality a 90.3%-scope verifier needs
before it is worth calling is AUROC ≈ 0.65 at a 9.87% false-alarm rate and
*nothing at all* at a 0% one — the PRM is net-positive down to AUROC 0.55 once
its false alarms are removed. The threshold is made entirely of false alarms,
so halving that rate buys more than any further detection gain.

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
> check itself, which detects zero errors. Among verifiers reading the
> generator's own unverified context, it closes only for one that is both
> independent of the generator and specialised for the task.
>
> (Scoring against *verified* premises instead lifts the ceiling further — You
> et al. reach 90.3% F1 on propagated errors that way — but it assumes a
> verified prefix, which is precisely what a budgeted gate cannot supply.)
>
> And the gate that would allocate that verifier's budget cannot be built,
> because neither token-level uncertainty (AUROC 0.5589) nor sampling-based
> semantic divergence (0.5740) ranks the risk, and combining them buys 0.0002.
> Split conformal calibration then holds its coverage guarantee while missing
> its selective-risk target by a factor of three, and reports nothing amiss.
>
> An oracle score locates the damage precisely. It takes selective risk from
> 0.154 to 0.089 and projected accuracy from 0.79 to 0.98 with a third of the
> verification calls — so the verifier was never the problem, and the score is.
> But the oracle still misses α = 0.05 by 1.8×, because at two calls per
> question most wrong steps go unverified however well they are ranked.
>
> And a better signal does exist: a probe on the generator's own frozen hidden
> states reaches 0.6968 against 0.5742, and improves the gate at every α on
> fewer calls. It is not enough. It misses α = 0.05 by 2.9×, and leaves the
> task PRM still net-negative. The failure is not that step-level uncertainty
> is unreadable — it is that reading it better than anything here managed still
> does not buy risk control at a realistic budget.
>
> Selective verification of LLM reasoning is bottlenecked at the **signal** and
> at the **budget**. The verifier is downstream of the first, and the
> calibration cannot help while the first is unfixed.

That is a claim about the design space rather than about one system, each leg
has a measured number and a regression test, and it is falsifiable in the only
way that matters: a signal that ranks step error would break it.

The project is more useful for having failed. A working gate on GSM8K would
have been one more entry in a crowded area. A measurement of *why* the obvious
design cannot work, with the three failure points separated and quantified, is
a result that transfers.
