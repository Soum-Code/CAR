# HANDOFF — read this first, then start

Working state for the CAR thesis. Not thesis content: this is what a new
session needs to pick the work up without re-deriving it. If something here
contradicts the code, the code wins — check before relying on a line.

**Last updated:** 2026-09-26, after the non-linear probe run and its
verification pass.

---

## 1. What the project is

**CAR (Conformalized Agentic Reasoning)** — M.Tech 3rd-semester research, at
`C:\MLOPs\3rd sem project\CAR`, mirrored to the **public** repo
https://github.com/Soum-Code/CAR (MIT + Apache-2.0).

It is a **measurement study, not a method paper.** Two proposed method
contributions were killed by literature checks and the user committed to the
measurement framing on 2026-09-02. Do not try to revive them:

1. *Adaptive conformal under censored feedback* — already published as
   Conformal Selective Acting (arXiv:2605.20270) Thm E.1, under a stronger
   guarantee.
2. *Influence-weighted allocation / "verify early"* — refuted on chains, five
   synthetic DAG families, and real extracted graphs.

The central finding: conformal machinery certifies *local* step validity, but
**69.8% of globally-wrong GSM8K steps are arithmetically perfect** — wrong only
because a premise was. A calculator cannot see any of them.

Framing lives in `README.md` and `docs/THESIS.md`. The draft is
`docs/thesis/` (9 chapters + front matter + 12 figures + `references.bib`).

---

## 2. Status

**The draft is complete.** Every chapter written, every number measured and
reproducible, every refuted claim carrying a regression test. 342 tests pass.

Experiments finished, most recent first:

| # | experiment | result | writeup |
|---|---|---|---|
| 11 | A non-linear probe on the same frozen states | null at matched layer (+0.0051); layer choice is noisier than any effect measured | `docs/FINDINGS-PROBE-NONLINEAR.md` |
| 10 | Probe round two (more data + first-bad target) | more data buys ~+0.01 (C9c unsupported, not refuted); the right target raises first-bad recall 0.30 -> 0.45 | `docs/FINDINGS-PROBE2.md` |
| 9 | Bidirectional entailment clustering | 0.5625, CI [0.512, 0.614] — reference relation does NOT rescue the signal | `docs/FINDINGS-ENTAILMENT.md` |
| 8 | Score-quality sweep | verifier turns net-positive at AUROC ≈ 0.65; AUROC is the wrong metric | `docs/FINDINGS-SCORE-QUALITY.md` |
| 7 | Internal-state probe | AUROC 0.6968 vs 0.5742; better, still not enough | `docs/FINDINGS-PROBE.md` |
| 6 | Dependency-graph hand-validation | 5.6% edge error, found a systematic bug | `docs/FINDINGS-DEPGRAPH.md` |
| 5 | End-to-end gate | selective risk misses α by 3× at every binding α | `docs/FINDINGS-PIPELINE.md` |

---

## 3. Recent history worth knowing

The entailment write-up was verified by a 3-agent adversarial workflow before
being committed, and it found **25 real problems** in the first draft —
including three that changed conclusions:

- **A double standard on noise.** The draft called entailment's +0.0063
  composite edge "inside noise" and its −0.0114 divergence gap a ranking. The
  gap's CI is [−0.087, +0.059], P(worse) = 0.63. "Slightly worse" was removed.
- **A selected denominator.** "The relations agree 89.9%" conditioned on the
  NLI model's own positive verdict. Over all comparable pairs it is 85.5% — and
  the diagnostic that matters is the reverse conditional: entailment merges
  **52%** of the pairs whose asserted numbers conflict.
- **Overclaimed corroboration.** "Three substantially different partitions" —
  they nest (0.1% ⊂ 31.9% ⊂ 78.0%, containment 100% and 93.0%). One
  permissiveness knob, not three probes.

The non-linear probe write-up (§7.6) was then caught by the same process
**before** it was committed: it claimed the instrument was the largest of three
levers, on a comparison of two probes that had landed on different layers. At
matched layer the effect is +0.0051 and the sign flips across layers. That is
the third time in this section that a correct number was used with the wrong
control.

Lesson worth repeating: **verify write-ups adversarially before committing.**
Recomputing every number from the artifacts is cheap next to publishing a wrong
one in a thesis — and cheaper still than the three correction commits it took
when the check came after the push rather than before it.

```
 M README.md                            new blockquote on the reference relation
 M docs/THESIS.md                       C11 / C11b / C11c rows
 M docs/thesis/07-the-assembled-gate.md new subsection in §7.2
 M docs/thesis/09-conclusion.md         rewrote the entailment future-work item
 M docs/thesis/README.md                findings-map row
 ?? docs/FINDINGS-ENTAILMENT.md
 ?? runs/entailment_cache.jsonl         27,936 NLI verdicts — 19.6h of CPU, COMMIT THIS
 ?? runs/uncertainty_qwen25_7b_entail.jsonl
 ?? tests/test_equivalence_relations.py
```

Before committing: `python -m pytest -q`, the broken-link check (§7), and
`python scripts/check_citations.py --all`.

---

## 4. What is actually left

**Blocked on the user — do not attempt these yourself:**

1. **Llama 3.1 8B re-measurement.** Kernel `somnath26/car-llama-transfer` is
   built and waiting on `HF_TOKEN` as a Kaggle Secret. Meta's licence
   acceptance is the user's to give; I declined to accept terms on their
   behalf and that decision stands.
2. **A Hugging Face token of the user's is exposed in another of their public
   repos.** They were told on 2026-09-24 to revoke it; unconfirmed as of this
   writing. Exact locations are deliberately not repeated here — this file is
   committed to a public repo and naming them would be a pointer to a possibly
   live credential. Ask the user, or search their repos for `hf_` yourself.
   **Never reuse that token**, and never hardcode any token —
   `tests/test_generation_checkpoint.py::test_gated_model_auth_is_not_defaulted_to_a_literal`
   enforces this.

**Open experiments, cheapest first:**

3. **More EVALUATION data — now the binding constraint on the whole probe
   series.** All three levers sit inside their own intervals: data +0.0105,
   instrument +0.0051 at matched layer, target on a different metric. Worse,
   287 selection steps separate candidate layers by 0.017 while their test
   AUROCs differ by 0.075 — a 0.0012 selection margin once cost 0.0749 of test
   AUROC. An arbitrary layer choice swamps every effect §7.6 can measure. More
   generated solutions, same as #4.
4. **More first-bad-step labels.** §7.6 *does* train a score against first-bad
   recall and it works (0.30 → 0.45), but on 49 training positives and 40 test
   ones, and the gain is significant against one comparator and not the other.
   108 such steps exist in the whole corpus — that is the binding constraint,
   and generating more solutions is the one thing that would relieve it.
5. **A third generator** would settle which per-generator quantities (μ,
   absorption) are monotone in model strength and which are idiosyncratic.
6. **ARES-style conditioning under a budget** — the clearest remaining
   scientific gap; see ch. 9 future work.

**Not an experiment, and worth asking about:** the draft is markdown. If the
department wants a LaTeX PDF on a university template, that conversion is a
real job and should start well before the deadline. The user has not said.

---

## 4b. One artifact lives outside git

`runs/probe_states.npz` is **535 MB** — over GitHub's 100 MB file limit, so it
is gitignored and exists only on this machine. It holds the frozen hidden
states for all 2,573 steps (29 layers, fp16) plus labels, roles, solution ids
and step positions. Every probe question is CPU-only while it exists.

Regenerate with:

```bash
python scripts/gpu_probe_states.py --save-states runs/probe_states.npz
```

GPU, ~15 min. Kaggle kernel `somnath26/car-probe2` does it end to end and also
runs the variants; the source dataset is `somnath26/car-source-verifier-scope`
and must be re-uploaded with `kaggle datasets version -d --dir-mode zip`
(without `--dir-mode zip` it silently uploads only the top-level files).

## 5. Traps — these cost real time

**Environment**

- **No local GPU.** `torch.cuda.is_available()` is False on this machine.
  Anything model-heavy is either Kaggle or slow CPU.
- Kaggle batch GPUs are **P100 (sm_60)** but the preinstalled torch is cu128
  (sm_70+). Install `torch==2.6.0+cu118` *before* the first torch import —
  torch cannot be reloaded in-process.
- Kaggle's transformers tokenizes SentencePiece models differently from
  training: the `▁` marker is not added, so `▁ки` 12902 → `ки` 1107. Only
  pinning `transformers==4.44.2` fixed it. This silently produced a confident
  **wrong** result before a validation gate caught it.

**Data**

- **Math-Shepherd is sorted into blocks by label.** Any prefix read is
  single-class — the first 80MB is 100% wrong-answer. Use
  `load_solutions(..., stride=N)`, never a bare `limit`.
- A **chain topology makes influence-weighting and front-loading numerically
  identical**, so chain-only experiments cannot distinguish them. Check an
  experiment can *see* the difference before trusting a negative result.

**Process — the ones I got wrong this session**

- **Do not write code containing escapes through a bash heredoc.** `\n` gets
  mangled and produces `SyntaxError: unterminated string literal`. This
  happened three times. Use the Write/Edit tools.
- **`&` background processes do not survive the Bash tool call.** Use the
  tool's `run_in_background`, not `nohup ... &`.
- **Piping a long-running command through `grep` hides its progress** — the
  pipe buffers and the output file stays empty. Run it unpiped with `python -u`.
- **Check figure numbering after inserting a figure.** Chapter 7's figures went
  out of reading order twice; captions are numbered by hand.
- **A comparison that claims to isolate one variable must hold the others
  fixed.** "Doubling the data does not move it" was published from two probes
  that had selected *different layers*; at fixed layer and C the sign reverses.
  Re-running model selection is not a control.
- **A learning curve must sample its pool.** The pooled training set was
  `concat(dev_shuffled, cal_unshuffled)`, so prefixes of it changed composition
  as well as size and the slope read negative for that reason.
- **Check which split a curve is evaluated on.** C9c ("0.6968 is a floor") stood
  for weeks on a learning curve scored on the *selection* split — the data the
  layer and C were chosen on, so it was biased and measured on the wrong
  population. Any curve, CI or score computed on data used for selection is not
  evidence.
- **Always add a harness-validation gate before reporting a model-derived
  measurement** — verify the model reproduces labels it was trained on. This
  caught a fake headline result once.

---

## 6. Useful commands

```bash
python -m pytest -q                                  # 342 tests
python scripts/check_citations.py --all              # every cited arXiv id has an entry
python scripts/make_figures.py                       # all 12 figures, png + pdf
python scripts/exp_gate_pipeline.py --probe runs/probe_qwen25_7b.json
python scripts/exp_score_quality_threshold.py --seeds 64      # ~70s, deterministic
python scripts/exp_equivalence_relations.py                   # reads committed artifacts
```

Broken-link check (there is no script; this is the inline one used):

```bash
python -c "
import re, pathlib
bad = 0
for md in list(pathlib.Path('.').glob('*.md')) + list(pathlib.Path('.').glob('docs/**/*.md')):
    for m in re.finditer(r'\[[^\]]*\]\(([^)\s]+)\)', md.read_text(encoding='utf-8')):
        t = m.group(1).split('#')[0]
        if t and not t.startswith(('http','mailto:')) and not (md.parent/t).exists():
            print('BROKEN', md, '->', t); bad += 1
print('broken:', bad)"
```

**Do not re-run** `scripts/gpu_semantic_divergence.py --equivalence entailment`
without a reason: it is 27,936 NLI pairs, **19.6 hours of CPU**. The verdicts
are cached in `runs/entailment_cache.jsonl` and it resumes from there.

---

## 7. Working agreements

- **Respond in English**, even when the user writes in Hindi/Hinglish. They
  switched working language mid-session on 2026-08-31 and it has held since.
- The user consistently prioritises **knowing whether a claim is real over
  having a claim**. Negative results are the contribution here. Report failures
  plainly, with the numbers.
- Every refuted claim gets a **regression test** so it cannot quietly stop
  reproducing. Follow that convention for anything new.
- Long runs get **checkpointing**. A 20-hour job with no resume is how this
  project lost a GPU session once, and nearly lost the entailment run.
