# Session handoff

*State as of 2026-10-09. Read this, then `docs/thesis/README.md`.*

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

## The paper's review is now closed

The thesis went through three adversarial review passes **with** verification:
84 findings claimed, 56 confirmed, all fixed. The paper's first pass died on a
session limit with its verification stage unrun, leaving 20 findings filed by
two finder agents (`numbers-in-context` and `dropped-hedges`, overlapping) in
`.claude/.../subagents/workflows/wf_4165863f-acc/journal.jsonl`.

Four were confirmed by hand and fixed on 2026-09-30: the oracle table's recall
column mixing first-bad with global recall; the abstract's mismatched
baselines; "a third of the calls" where the comparison is a fifth; and sweep
endpoints spanning two methods (now 0.1491 → 0.1538 and 4.6% → 21.8%).

The remaining **14 were verified on 2026-10-10** — one verifier re-reading both
documents per finding, then two refuters each (one re-deriving from sources,
one judging whether 6 pages may compress the caveat). Run
`wf_9d4bc15c-b75/journal.jsonl`, 40 agents. Outcome: **10 fixed, 4 dropped.**

The four dropped matter more than the ten fixed, because three of them were
*wrong*, and two would have put a new error into the paper:

- **`abstract-call-ratio` was a false alarm.** 0.154 is split conformal's
  0.1538, not no-gate's 0.1554, and 0.79 is 0.7912. Both arrows run from the
  same row, so 0.37 / 1.09 = "a third" is correctly denominated. The abstract
  is tighter than the thesis here. Do not "fix" it.
- **`c1-denominator`** claimed 78.5% should be 69.8%. It should not: ch1 §1.4,
  the front matter, ch10 and `THESIS.md` all use the unqualified phrasing with
  78.5%, and §2.2 of the paper stipulates `C1_checkable` as the reported
  estimator. The finder's own arithmetic did not reproduce either.
- **`teacher-forcing`** wanted the AUROC block labelled as teacher-forced
  features. Semantic divergence *cannot* be teacher-forced — it is defined over
  independently sampled continuations, `gpu_score_uncertainty.py:18-22` says so
  and writes 0.0 as a placeholder, and the paper already says "resampled" three
  lines above. The fix would have contradicted the paper against itself.
- **`recovery-monotone`**: the paper prints counts over visibly different
  denominators (0/14,573, 12/141, 6/83), never the rates, and ch1 of the thesis
  compresses it identically.

Note that `check_paper.py` passed on all 20 throughout. Every numeric literal
does appear in a source — the defects were numbers attached to the wrong
comparison, which is exactly the gap a numeric checker cannot close.

## What is outstanding

**Three administrative items**, none technical, all named on slide 20 of the deck:

1. **Plagiarism report** — requested 2026-10-09 from the KIIT library
   (lakshminarayan.ghosh@kiit.ac.in) from `25167023@kiit.ac.in`, with the paper
   PDF attached. Awaiting reply on the procedure and whether a Word file is
   needed. The thesis was deliberately NOT sent — the paper only.
   **The attached copy is now stale**: it predates the 2026-10-10 review fixes.
   Resend `outputs/Somnath_Reddy_25167023_Paper.pdf` if he has not run the
   check yet, or let it stand — the edits are hedges and denominators, so a
   similarity score will not move.
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
