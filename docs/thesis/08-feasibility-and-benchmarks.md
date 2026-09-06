# 8. Feasibility and benchmarks

> Reproduce: `python scripts/exp_measure_error_rate.py`,
> `python scripts/exp_strategyqa_topology.py`,
> `python scripts/exp_benchmark_compare.py`

Two questions that should be answered before any method is designed, and were
not answered in the original specification: is the risk target attainable at
all, and can the benchmark exhibit the phenomenon being studied?

Both answers are negative for the choices the specification made.

## 8.1 The risk target is constrained before any method is chosen

Kotte's Proposition 3: when base risk μ exceeds target α, any distribution-free
method must verify or abstain on at least `(μ − α)/(1 − α)` of items.

With the μ measured in Chapter 4 for Mistral-7B-SFT:

| α | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 |
|---|---|---|---|---|---|
| μ = 0.3908, floor | 35.9% | **32.3%** | 23.8% | 13.0% | none |

**α = 0.10 — the specification's value — charges a third of the entire
verification budget as an entry fee** before any method is admissible. It is
not a modest target; against global risk it is close to infeasible, and it is
only defensible against *local* risk, which Chapter 4 showed is not the
quantity of interest.

The configuration was changed to α = 0.30, where the 13.0% floor is affordable
against a 37.5% budget and which sits inside the 0.30–0.40 band Kotte found
workable for LLM tasks. `configs/default.yaml` now **enforces the floor at
setup**: a configuration whose budget sits below its own floor raises rather
than producing a flat results table that has to be diagnosed afterwards.

### The floor is a property of the generator

Chapter 4 measured μ on a second generator, and the picture changes completely:

| α | 0.05 | 0.10 | 0.20 | 0.30 |
|---|---|---|---|---|
| Mistral-7B-SFT, μ = 0.3908 | 35.9% | 32.3% | 23.8% | 13.0% |
| Qwen2.5-7B, μ = 0.1221 | 7.6% | 2.5% | **none** | **none** |

The impossibility bound has not weakened; the base risk it applies to has. On a
strong generator α = 0.20 is attainable with no entry fee at all.

**Any statement about attainable α must name the model it was measured on.**
This thesis originally stated the floor as a property of the task, and that was
wrong.

Chapter 7 shows the sting in the tail: when μ falls below α, the target becomes
*vacuous* rather than easy. At α = 0.30 against μ = 0.1578, verifying nothing
satisfies the guarantee. A conformal result that is satisfied by the empty
policy certifies nothing, and the sweep in §7.3 exists precisely to find where
α starts to bind.

## 8.2 StrategyQA cannot exercise the phenomenon it is used for

StrategyQA was the original primary benchmark, chosen because it ships human
annotated decompositions — apparently ideal for studying step-level reasoning.
Extracting all 2,272 annotated dependency graphs shows why that was the wrong
choice.

| property | StrategyQA | GSM8K (corrected) |
|---|---|---|
| mean graph depth | 2.30 | 2.79 |
| graphs exactly one hop deep | **72.9%** | — |
| steps with a non-terminal descendant | **11.2%** | **29.9%** |
| max depth | 5 | 8 |
| questions at depth ≥ 3 | 615 (27.1%) | 3,815 |

**A step can only corrupt downstream reasoning if downstream reasoning
exists.** With 72.9% of graphs one hop deep and only 11.2% of steps having any
descendant other than the answer itself, propagation barely occurs. The
phenomenon this thesis is about is almost absent from the benchmark the
specification chose to study it on.

The consequence is visible in the allocation results. On StrategyQA the spread
between the best and worst allocation policy is **0.0126**; on GSM8K it is
**0.0338**, and it grows with depth (0.0033 at depth 2 → 0.1279 at depth 6).
Policy comparisons on StrategyQA are measuring noise.

### The switch, and what it cost

GSM8K became the primary benchmark on evidence rather than preference. It has
2.4× the propagation headroom, depth reaching 8, and five times as many
questions at depth ≥ 3.

What was given up is the annotated decomposition: GSM8K dependency graphs are
**derived** from calculator-operand matching, not annotated. Chapter 6
hand-validates that derivation and measures its error at 5.6% of edges. That is
a real cost, and it is quantified rather than waved at.

StrategyQA is retained for calibration and for evidence-grounded verification,
where its evidence paragraphs are genuinely useful — and it is the only place a
retrieval verifier arm could run at all, though §5.3 explains why that arm is
not available in practice.

## 8.3 A note on benchmark choice as a research decision

Both findings in this chapter share a shape. The specification made a
reasonable-looking choice — α = 0.10 because it sounds like a strong guarantee,
StrategyQA because it has annotations — and in each case a cheap measurement
made before any method design would have shown the choice was unworkable.

The feasibility check costs one script and two numbers. The benchmark check
costs an afternoon of graph extraction. Both were run late in this project, and
both invalidated work that had already been done.

> Run the feasibility test and the benchmark-capacity test as experiment zero.
> They are the cheapest experiments in the project and they constrain
> everything that follows.
