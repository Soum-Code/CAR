# Session handoff

*State as of 2026-09-30. Read this, then `docs/thesis/README.md`.*

Everything below is committed and pushed to `Soum-Code/CAR` on `main`.

---

## What exists

| artifact | where | state |
|---|---|---|
| **Thesis** | `thesis-latex/main.pdf` | 89 pages, compiles clean, 0 undefined refs |
| Thesis source | `docs/thesis/` | 10 chapters + front matter, markdown is the source of truth |
| **Paper** | `paper-latex/main.pdf` | 6 pages, IEEE conference format, 0 overfull boxes |
| Paper source | [docs/paper/paper.md](docs/paper/paper.md) | see [docs/paper/README.md](docs/paper/README.md) |
| **Defence deck** | `docs/CAR-final-review.pptx` | 21 slides, speaker notes on every one |
| Project report | `docs/PROJECT-REPORT.{md,docx,pdf}` | ~6,000 words, the "what is this project" document |
| Findings notebooks | `docs/FINDINGS-*.md` | what was tried and discarded, not only what worked |
| Claim ledger | [docs/THESIS.md](docs/THESIS.md) | every claim, its evidence, and its epistemic label |

`thesis-latex/` and `paper-latex/` are **build products and gitignored**. Regenerate:

```bash
python scripts/build_latex.py        # thesis
python scripts/build_paper_latex.py  # paper
python scripts/make_deck.py          # defence deck
```

A TeX toolchain (MiKTeX) is installed on this machine, so both compile locally:

```bash
cd thesis-latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## The paper has NOT been verified to the same standard as the thesis

This is the single most important thing to know before submitting it anywhere.

The thesis went through three adversarial review passes **with** verification:
84 findings claimed, 56 confirmed, all fixed. The paper got one pass whose
verification stage died on a session limit. Four findings have since been
confirmed by hand and fixed:

- the oracle table's recall column mixed first-bad recall (0.3590, 0.3077) with
  global recall (0.4366) from a different thesis table, under one heading;
- the abstract paired a no-gate risk baseline with a split-conformal accuracy
  baseline and a split-conformal call ratio;
- "gains 17.6 points on a third of the calls" — that comparison is a fifth;
- "risk barely moves across the sweep (0.1491 → 0.1587) while verification
  climbs from 4.6% to 26.9%" — those endpoints span the split-conformal AND
  the CAR column of the thesis table. The paper tabulates only split conformal,
  so its prose contradicted its own table. Now 0.1491 → 0.1538 and 4.6% → 21.8%.

**Eight remain unverified.** They are recoverable from the workflow journal at
`.claude/.../subagents/workflows/wf_4165863f-acc/journal.jsonl`. The two worth
settling first, both being hedges lost in compression:

- the AUROC ≈ 0.65 crossing may be stated without the thesis's caveat that
  0.6968 sits *inside* the [0.625, 0.700] band the sweep cannot resolve — "at
  the boundary, not past it";
- the claim that halving false alarms beats raising detection, which §10.4
  calls "an argument, not a result" because scope was pinned at 0.9033 in every
  sweep row and detection was never varied.

Note that `check_paper.py` passes on all of these. Every numeric literal in the
paper does appear in the thesis — they are attached to the wrong comparison,
which is exactly the gap a numeric checker cannot close.

## What is outstanding

**Verify the paper's remaining eight findings** before it goes to anyone — see
the section above. Everything else below is administrative.

**Three administrative items**, none technical, all named on slide 20 of the deck:

1. **Plagiarism report** — not generated. Needs the institute's tool.
2. **Publication status** — the paper is written and compiles but has not been
   submitted. Before it goes anywhere: the guide reads it, a venue is chosen,
   and that venue's LLM-usage disclosure policy is checked.
3. **Panel feedback tracker** for Reviews 0–2 — to be compiled from the review
   records. The Final Review checklist asks for it explicitly.

**One optional edit.** `docs/thesis/acknowledgements.md` names nobody beyond the
supervisor, deliberately — inventing a specific debt is worse than a general
thanks honestly given. If a particular person earned a line, add them.

**Research gaps**, stated in ch. 10 as future work and none of them a defect:
more first-bad-step labels (only 108 exist, 40 in test), more evaluation data
for §8.6, ARES-style conditioning under a budget, a fourth generator outside the
7–8B band.

---

## The thing to understand before changing anything

This project's failure mode is **numbers that drift between documents**. Roughly
75 numerical errors have been found in it by recomputation, and three full
adversarial review passes ran on 2026-09-29/30 — 84 findings claimed across the
thesis, 56 confirmed after verification, all fixed.

The recurring shapes, all of which have bitten more than once:

- **A rate against the wrong denominator.** The thesis deliberately reports
  several similar-looking quantities over different populations: `C1_all`
  (0.6982) vs `C1_checkable` (0.7848); marker notation vs `notation="any"`;
  base risk 0.1578 over all test steps vs no-gate selective risk 0.1554 vs
  split-conformal-at-α=0.30 0.1538; recovery over 14,573 *eligible* solutions,
  not 25,971 total. Before "fixing" an inconsistency, check it is not two
  different quantities.
- **A selection-split number quoted as a test number.** The probe's AUROC is
  0.8748 on selection and 0.6968 on test. The gap is the winner's curse.
- **An interval spanning zero described as an effect.** Four labels are used and
  kept distinct: `refuted`, `unsupported`, `null`, `measured`.

**Run the guards before committing anything:**

```bash
python -m pytest                    # 360 tests, no GPU, no network
python scripts/check_crossrefs.py   # every §, chapter and figure ref resolves
python scripts/check_citations.py --all
python scripts/check_paper.py       # every number in the paper traces to the thesis
python scripts/check_latex.py       # static LaTeX validation
```

`check_prose.py --baseline` / `--compare` fingerprints every numeric literal, so
an editing pass that silently moves one is visible. Use it for any prose-only
edit.

---

## Two traps that are documented but easy to re-trigger

**Chapter ranges survive a renumber wrong.** The 2026-09-29 split inserted
chapter 3 and shifted everything after it. A script moved ~160 references, but
"Chapters 4 through 7" became "Chapters 5 through 7" — only the number following
the keyword moves, and both ends still resolve, so no checker sees it.
`check_crossrefs.py` now lists every range for confirmation by hand.

**The Math-Shepherd file is sorted into contiguous blocks by label.** Any prefix
read is close to single-class — the first 80 MB is 100% wrong-answer GSM8K.
`download_data.py` fetches 48 strided range requests and asserts the class
balance; a test asserts a prefix read is more skewed than a strided one. It has
returned twice.

---

## Facts worth having in hand

Front matter is complete in `docs/thesis/metadata.yaml`, and
`build_latex.py` **refuses to build** while any `FILL:` marker remains:

```
P. Somnath Reddy · 25167023
M.Tech, Computer Engineering
School of Computer Engineering, KIIT, Bhubaneswar
Guide: Ponsuresh Manoharan, Subject Matter Expert — Cyber Security,
       L&T EduTech, Chennai  (signs the certificate alone)
September 2026 · academic year 2026-27
```

The three corpora: Math-Shepherd (Mistral-7B-SFT, 93,129 steps), and two
generated for this work — Qwen2.5-7B-Instruct (2,573 steps) and Llama 3.1 8B
Instruct (1,938 steps). Both generated corpora are committed under `runs/`
because they cost Kaggle GPU sessions and cannot be regenerated on CPU.

The headline result, stated the way the thesis states it: selective verification
of LLM reasoning is bottlenecked at the **signal** and at the **budget**. The
verifier's net-negative result is downstream of the score rather than a third
independent defect — an earlier draft claimed three independent failure points
and the oracle experiment refuted it. If you find the three-point version
anywhere, it is stale.
