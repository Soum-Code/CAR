# Thesis draft

Full draft, 2026-09-06. Nine chapters plus front matter.

| ch | title | reproduces with |
|---|---|---|
| — | [Front matter and abstract](00-front-matter.md) | — |
| 1 | [Introduction](01-introduction.md) | — |
| 2 | [Background](02-background.md) | — |
| 3 | [Framework](03-framework.md) | `pytest` |
| 4 | [Measuring the gap](04-measuring-the-gap.md) | `exp_measure_error_rate.py`, `exp_generator_transfer.py` |
| 5 | [Verifier reach](05-verifier-reach.md) | `exp_verifier_scope.py`, `gpu_semantic_scope.py --analyse` |
| 6 | [Allocation](06-allocation.md) | `exp_allocation.py`, `exp_topology.py`, `validate_dependency_graphs.py score` |
| 7 | [The assembled gate](07-the-assembled-gate.md) | `exp_gate_pipeline.py` |
| 8 | [Feasibility and benchmarks](08-feasibility-and-benchmarks.md) | `exp_strategyqa_topology.py`, `exp_benchmark_compare.py` |
| 9 | [Limitations and conclusion](09-conclusion.md) | — |

## Relationship to the findings documents

The chapters are the argument; the `docs/FINDINGS-*.md` files are the lab
notebooks they draw on, and are kept because they record what was tried and
discarded, not only what worked.

| chapter | primary source |
|---|---|
| 4 | [FINDINGS-PROPAGATION.md](../FINDINGS-PROPAGATION.md) Addendum 4, [FINDINGS-GENERATOR.md](../FINDINGS-GENERATOR.md) |
| 5 | [FINDINGS-PROPAGATION.md](../FINDINGS-PROPAGATION.md) Addendum 5 |
| 6 | [FINDINGS-PROPAGATION.md](../FINDINGS-PROPAGATION.md) Addenda 1–3, [FINDINGS-DEPGRAPH.md](../FINDINGS-DEPGRAPH.md) |
| 7 | [FINDINGS-PIPELINE.md](../FINDINGS-PIPELINE.md) |
| 8 | [FINDINGS-PROPAGATION.md](../FINDINGS-PROPAGATION.md) Addenda 2–3 |
| 2, 9 | [POSITIONING.md](../POSITIONING.md) |

[THESIS.md](../THESIS.md) is the working plan and claim-to-evidence map, kept
current as the index of what is measured and what is not.

## Status

**Complete as a draft.** Every chapter is written, every number in it is
measured and reproducible, and every refuted claim has a regression test.

Known gaps, in the order they would matter to an examiner:

1. **No figures.** All results are tables. The accuracy-vs-cost Pareto plot
   with the Kotte floor as a reference line is the one figure the argument
   actually wants, and it is not drawn.
2. **Chapter 2 needs the outstanding reads.** Sherlock, ARES, CAP and
   Barber et al. on conformal beyond exchangeability are cited from secondary
   description in places; §2.7–2.8 should be tightened after a full read.
3. **No related-work chapter separate from background.** For a paper
   submission these would split.
4. **Citations are inline links, not a bibliography.** Needs converting to
   BibTeX for submission.
5. **A third generator** would settle which of the per-generator quantities
   (μ, absorption) are monotone in model strength and which are idiosyncratic.

## Venue

Realistic targets: an ACL/EMNLP short paper, or a NeurIPS/ICLR workshop on LLM
evaluation or uncertainty. The measurement plus the benchmark critique is a
credible short-paper contribution; Chapter 7's three-point failure analysis is
the part most likely to interest a main-conference audience, because it is a
negative result with a mechanism rather than a null.

For the M.Tech thesis this is comfortably sufficient: a clear question,
measured answers, refuted hypotheses recorded honestly, and a working system
whose failure is itself the result.
