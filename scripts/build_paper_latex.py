"""Build the paper as an IEEE conference submission. [CPU]

docs/paper/paper.md is the source; this emits paper-latex/ ready for pdflatex.

Reuses the thesis converter in build_latex.py rather than reimplementing it.
That converter has 356 tests behind it and a history of bugs that only showed
up under a compiler -- escaped footnotes, emphasis split across line breaks,
unicode smuggled through code blocks -- and none of those are worth
rediscovering in a second implementation.

What is different from the thesis build, and why:

  * Two columns. IEEEtran's text column is roughly half the width of the
    thesis's, so a table that fitted there will not fit here. Tables wider than
    one column are emitted as `table*`, which spans both.
  * Numbered citations. The draft writes [key] and [key1; key2] inline;
    IEEEtran wants \\cite{}. Converted before escaping, via a sentinel that
    survives it.
  * No chapters, no table of contents, and the reference list in the markdown
    is replaced by a real bibliography off references.bib.

    python scripts/build_paper_latex.py
    cd paper-latex && pdflatex main && bibtex main && pdflatex main && pdflatex main
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_latex as B  # noqa: E402

SRC = Path("docs/paper/paper.md")
BIB = Path("docs/thesis/references.bib")
FIGS = Path("docs/thesis/figures")
OUT = Path("paper-latex")

# Characters of \footnotesize text that fit in one IEEEtran column. Measured
# from the compiled output, like the thesis build's own budget.
COLUMN_CHAR_BUDGET = 46

PREAMBLE = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts

\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}
\usepackage{listings}
\usepackage{url}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\lstset{
  basicstyle=\ttfamily\scriptsize,
  breaklines=true,
  frame=single,
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  literate={-}{{-}}1 {>}{{>}}1 {<}{{<}}1
           {μ}{{$\mu$}}1 {α}{{$\alpha$}}1
           {−}{{$-$}}1 {·}{{$\cdot$}}1,
}

\begin{document}

\title{%(title)s}

\author{\IEEEauthorblockN{%(author)s}
\IEEEauthorblockA{\textit{%(department)s} \\
\textit{%(institution)s} \\
%(email)s}
}

\maketitle

\begin{abstract}
%(abstract)s
\end{abstract}

\begin{IEEEkeywords}
%(keywords)s
\end{IEEEkeywords}

"""

TAIL = r"""

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""


def convert_table_2col(rows: list[str]) -> str:
    """Table float, spanning both columns when it cannot fit in one."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]
    ncol = len(header)
    body = [(r + [""] * ncol)[:ncol] for r in body]

    widths = [max(B._visible(r[i]) for r in [header] + body) for i in range(ncol)]
    natural = sum(widths) + 3 * ncol
    wide = natural > COLUMN_CHAR_BUDGET

    star = "*" if wide else ""
    if natural > COLUMN_CHAR_BUDGET * 1.8:
        cut = max(10, sorted(widths)[-1] // 3)
        spec = "".join("L" if w > cut else "l" for w in widths)
        if "L" not in spec:
            spec = "L" * ncol
        env, arg = "tabularx", r"{\textwidth}"
    else:
        spec, env, arg = "l" * ncol, "tabular", ""

    out = [rf"\begin{{table{star}}}[t]", r"\centering", r"\footnotesize",
           rf"\begin{{{env}}}{arg}{{{spec}}}", r"\toprule",
           " & ".join(B.inline(h) for h in header) + r" \\", r"\midrule"]
    for row in body:
        out.append(" & ".join(B.inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", rf"\end{{{env}}}", rf"\end{{table{star}}}"]
    return "\n".join(out)


def protect_citations(md: str) -> tuple[str, list[str]]:
    """[key], [k1; k2] and [key, Prop. 3] become \\cite, via a safe sentinel.

    Done before conversion because a bare [key] is not a markdown link and
    would otherwise print as literal brackets; done via a sentinel because
    esc() would escape the backslash of a \\cite inserted here, which is the
    bug that ate every footnote in the thesis build.
    """
    cites: list[str] = []

    def sub(m: re.Match) -> str:
        inner = m.group(1)
        note = ""
        if "," in inner:
            keys_part, _, note_part = inner.partition(",")
            if not re.fullmatch(r"[\s;a-z0-9]+", note_part):
                inner, note = keys_part, note_part.strip()
        keys = ",".join(k.strip() for k in re.split(r"[;,]", inner) if k.strip())
        cites.append(rf"\cite[{note}]{{{keys}}}" if note else rf"\cite{{{keys}}}")
        return f"ZZCITE{len(cites) - 1}ZZ"

    # [a-z0-9]* on the suffix, not [a-z]*: several keys carry digits after the
    # year -- cobbe2021gsm8k, qwen2024qwen25, dubey2024llama3 -- and a
    # letters-only suffix stops at the first digit, leaving the citation to
    # print as literal brackets.
    md = re.sub(r"\[((?:[a-z]+\d{4}[a-z0-9]*)(?:[;,][^\]]*)?)\]", sub, md)
    return md, cites


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--author")
    ap.add_argument("--institution")
    ap.add_argument("--department")
    ap.add_argument("--email", default="p.somnathreddy26@gmail.com")
    ap.add_argument("--allow-placeholders", action="store_true",
                    help="build even though metadata.yaml has FILL: values")
    args = ap.parse_args()

    # One source of truth. These values also set the thesis title page, and
    # keeping a second copy behind argparse defaults is how the thesis ended up
    # emitting a PDF that said "university name" in italics.
    meta = B.load_metadata(allow_placeholders=True)
    unfilled = [k for k in ("author", "institution", "department")
                if str(meta.get(k, "")).startswith("FILL:")]
    if unfilled and not args.allow_placeholders:
        print(f"unfilled in {B.META}: {', '.join(unfilled)}")
        print("Fill them, or pass --allow-placeholders for a draft.")
        return 1
    args.author = args.author or meta.get("author", "")
    args.institution = args.institution or meta.get("institution", "")
    args.department = args.department or meta.get("department", "")

    if not SRC.exists():
        print(f"missing {SRC}")
        return 1
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    for pdf in sorted(FIGS.glob("*.pdf")):
        shutil.copy2(pdf, out / "figures" / pdf.name)
    shutil.copy2(BIB, out / "references.bib")

    md = SRC.read_text(encoding="utf-8")
    B.check_listing_chars(md, SRC.name)
    md, cites = protect_citations(md)

    lines = md.split("\n")
    title = re.sub(r"^#\s+", "", lines[0]).strip()

    def section_slice(name: str) -> tuple[int, int] | None:
        pat = re.compile(rf"^##\s+(?:\d+\.\s*)?{name}\s*$", re.I)
        start = next((i for i, l in enumerate(lines) if pat.match(l.strip())), None)
        if start is None:
            return None
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].startswith("## ")), len(lines))
        return start, end

    abs_span = section_slice("Abstract")
    if abs_span is None:
        print("no '## Abstract' section in the paper")
        return 1
    abstract_md = "\n".join(lines[abs_span[0] + 1:abs_span[1]]).strip()

    ref_span = section_slice("References")
    body_end = ref_span[0] if ref_span else len(lines)
    body_md = "\n".join(lines[abs_span[1]:body_end])

    # The thesis converter emits \chapter/\section/\subsection for #/##/###.
    # A paper has no chapters, so shift everything up one level.
    body = B.convert("# _\n\n" + body_md, chapter_title="_")
    body = body.split("\n", 1)[1]
    body = re.sub(r"\\section\{", r"\\SECTION{", body)
    body = re.sub(r"\\subsection\{", r"\\SUBSECTION{", body)
    body = body.replace(r"\SECTION{", r"\section{").replace(
        r"\SUBSECTION{", r"\subsection{")
    # Section headings in the draft carry their own numbers; IEEEtran numbers.
    body = re.sub(r"(\\(?:sub)?section\{)(?:\d+(?:\.\d+)?\.?\s+)", r"\1", body)

    # Re-run the tables through the two-column converter.
    def retable(m: re.Match) -> str:
        return m.group(0)
    body = _retable_all(body_md, body)

    abstract = B.inline(" ".join(
        l.strip() for l in abstract_md.split("\n")
        if l.strip() and not l.startswith(("#", "|", "-"))
    ))

    text = (PREAMBLE % {
        "title": B.esc(title), "author": args.author,
        "institution": args.institution, "department": args.department,
        "email": args.email, "abstract": abstract,
        "keywords": "conformal prediction, LLM reasoning, step verification, "
                    "selective risk, uncertainty estimation",
    } + body + TAIL)

    for i, c in enumerate(cites):
        text = text.replace(f"ZZCITE{i}ZZ", c)

    (out / "main.tex").write_text(text, encoding="utf-8")
    n_tab = text.count(r"\begin{table")
    print(f"wrote {out}/main.tex")
    print(f"  {len(body.split()):,} words, {n_tab} tables, {len(cites)} citations")
    print(f"  {len(list((out / 'figures').glob('*')))} figures, "
          f"{len(re.findall(r'^@', (out / 'references.bib').read_text(encoding='utf-8'), re.M))} bib entries")
    return 0


def _retable_all(body_md: str, body_tex: str) -> str:
    """Replace the thesis-shaped table floats with two-column-aware ones."""
    md_tables = re.findall(r"(?:^\|.*\n)+", body_md, re.M)
    tex_tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", body_tex, re.S)
    for md_t, tex_t in zip(md_tables, tex_tables):
        rows = [r for r in md_t.strip().split("\n") if r.startswith("|")]
        if len(rows) < 3:
            continue
        body_tex = body_tex.replace(tex_t, convert_table_2col(rows), 1)
    return body_tex


if __name__ == "__main__":
    sys.exit(main())
