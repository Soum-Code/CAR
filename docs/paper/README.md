# Paper

[paper.md](paper.md) is the source. Everything else is built from it.

A workshop-length treatment of the thesis, leading with Chapter 7's three-point
failure analysis and the oracle that separates cause from consequence. C1 is
motivation rather than headline, and CAR appears as the measurement instrument
rather than as a method — every method claim it had is scooped or refuted, and
a reviewer would find that.

## Build

```bash
python scripts/build_paper_latex.py
cd paper-latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

IEEE conference format (`\documentclass[conference]{IEEEtran}`), two columns,
numbered citations off the thesis's own `references.bib`. Current output: 6
pages, 0 overfull boxes, 0 undefined citations, 14 references.

To target a different venue, replace the `PREAMBLE` string in
`scripts/build_paper_latex.py` with that venue's class and style file. The
conversion itself is venue-independent.

Two things the builder handles that the thesis build does not. Tables wider
than one IEEEtran column become `table*` and span both, which is why 5 of the 6
are starred. And the draft's inline `[key]` and `[k1; k2]` citations become
`\cite{}` through a sentinel that survives escaping — inserting a raw `\cite`
before `esc()` runs is the same bug that ate every footnote in the thesis
build.

## Checks

```bash
python scripts/check_paper.py        # every number traces to the thesis
python scripts/check_paper_prose.py  # style habits a restatement picks up
```

`check_paper.py` holds all 239 numeric literals in the draft against the thesis
chapters and findings documents they are drawn from. A hit is not proof a
number is used correctly, only proof it was not invented; anything reported
MISSING is either a typo or a claim the thesis does not support. It has already
caught two.

## Venue

Realistic targets are a NeurIPS/ICLR workshop on LLM evaluation or uncertainty,
or an ACL/EMNLP short paper. The measurement plus the benchmark critique is a
credible short-paper contribution; the three-point failure analysis is the part
most likely to interest a main-conference audience, because it is a negative
result with a mechanism rather than a null.

Both NeurIPS and ICLR publish LLM usage policies that require disclosure.
Check the specific venue's wording before submitting.
