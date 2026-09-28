"""Build a compilable LaTeX project from the markdown thesis. [CPU]

The thesis is written in markdown because that is what it was drafted, revised
and verified in -- every number in it has been recomputed from a committed
artifact, and the markdown is what those checks read. Converting by hand would
fork the source: two copies, one of which silently goes stale.

So this converts instead, and is re-runnable. `docs/thesis/*.md` stays the
single source; `thesis-latex/` is a build product.

    python scripts/build_latex.py          # writes thesis-latex/

WHAT IT HANDLES, AND WHAT IT DELIBERATELY DOES NOT

Handled: headings, emphasis, inline code, fenced code, tables (to booktabs),
block quotes, bullet and numbered lists, images with captions, links, and the
~30 non-ASCII characters the draft actually uses (alpha, mu, arrows, section
marks, minus signs, em dashes).

Not handled, on purpose: nested lists deeper than one level, inline HTML, and
reference-style links. None appear in the source, and a converter that guesses
at constructs it has never seen is a converter that produces silent breakage.
If one appears later this script should be extended rather than worked around
in the .tex, because the .tex is overwritten on every build.

COMPILING

No TeX toolchain is assumed. The output is a self-contained project that
compiles on Overleaf as-is (upload the folder, set main.tex as the root), or
locally with `latexmk -pdf main.tex` if you have one.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

SRC = Path("docs/thesis")
FIGS = Path("docs/thesis/figures")
BIB = Path("docs/thesis/references.bib")
OUT = Path("thesis-latex")

# Characters the draft actually uses. Anything outside this map that is
# non-ASCII raises, rather than silently emitting a character the compiler
# will drop.
UNICODE = {
    "—": "---", "–": "--", "…": r"\dots{}",
    "α": r"$\alpha$", "μ": r"$\mu$", "π": r"$\pi$", "γ": r"$\gamma$",
    "Φ": r"$\Phi$",
    "×": r"$\times$", "−": r"$-$", "·": r"$\cdot$", "→": r"$\rightarrow$",
    "≥": r"$\geq$", "≤": r"$\leq$", "≈": r"$\approx$", "≡": r"$\equiv$",
    "∞": r"$\infty$", "√": r"$\sqrt{\,}$", "⌈": r"$\lceil$", "⌉": r"$\rceil$",
    "⁻": r"$^{-}$", "¹": r"$^{1}$",
    "§": r"\S{}", "†": r"$\dagger$", "è": r"\`e",
    "▁": r"\_", "к": "k", "и": "i", "̂": "", "̃": "",
}

SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
           "_": r"\_", "{": r"\{", "}": r"\}"}


def esc(text: str, *, in_math: bool = False) -> str:
    """Escape LaTeX specials, then map the unicode the draft uses."""
    out = []
    for ch in text:
        if ch in SPECIAL and not in_math:
            out.append(SPECIAL[ch])
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ord(ch) > 127:
            if ch not in UNICODE:
                raise SystemExit(
                    f"unmapped non-ASCII character {ch!r} (U+{ord(ch):04X}). "
                    f"Add it to UNICODE in {__file__} rather than stripping it."
                )
            out.append(UNICODE[ch])
        else:
            out.append(ch)
    return "".join(out)


def inline(text: str) -> str:
    """Inline markup. Code spans are protected from escaping first."""
    spans: list[str] = []

    def stash(m):
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)

    # links: keep the text, footnote the URL (a printed thesis cannot be clicked)
    def link(m):
        label, url = m.group(1), m.group(2)
        if url.startswith("#"):
            return esc(label)
        if url.startswith("http"):
            return esc(label) + r"\footnote{\url{" + url + "}}"
        return esc(label)  # internal repo paths: drop the path, keep the words

    text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = esc(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", text)

    for i, s in enumerate(spans):
        # \texttt needs its own escaping; it is verbatim-ish but not verbatim
        safe = s
        for a, b in (("\\", r"\textbackslash{}"), ("{", r"\{"), ("}", r"\}"),
                     ("_", r"\_"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"),
                     ("$", r"\$"), ("^", r"\textasciicircum{}"),
                     ("~", r"\textasciitilde{}")):
            safe = safe.replace(a, b)
        text = text.replace(f"\x00{i}\x00", r"\texttt{" + safe + "}")
    return text


def convert_table(rows: list[str]) -> str:
    """Markdown pipe table to a booktabs tabular inside a table float."""
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]          # cells[1] is the --- rule
    ncol = len(header)
    # Left-align everything: these are mostly label + number tables, and the
    # numbers carry their own alignment through consistent decimal places.
    spec = "l" * ncol
    out = [r"\begin{table}[htbp]", r"\centering", r"\small",
           r"\begin{tabular}{" + spec + "}", r"\toprule",
           " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for row in body:
        row = (row + [""] * ncol)[:ncol]
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(out)


def convert(md: str, *, chapter_title: str) -> str:
    lines = md.split("\n")
    out: list[str] = []
    i = 0
    in_list = None            # 'itemize' | 'enumerate' | None

    def close_list():
        nonlocal in_list
        if in_list:
            out.append(rf"\end{{{in_list}}}")
            in_list = None

    while i < len(lines):
        line = lines[i]

        # fenced code
        if line.startswith("```"):
            close_list()
            i += 1
            block = []
            while i < len(lines) and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            out += [r"\begin{lstlisting}", *block, r"\end{lstlisting}"]
            continue

        # table
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?$", lines[i + 1]):
            close_list()
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(convert_table(rows))
            continue

        # image  ![alt](path)  -- the caption is the **Figure N.** line beneath
        m = re.match(r"^!\[(.*?)\]\((.+?)\)\s*$", line)
        if m:
            close_list()
            alt, path = m.group(1), m.group(2)
            name = Path(path).stem
            cap = alt
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("**Figure"):
                cap = re.sub(r"^\*\*Figure [0-9.]+\.\*\*\s*", "", lines[j])
                i = j
            out += [r"\begin{figure}[htbp]", r"\centering",
                    r"\includegraphics[width=\linewidth]{figures/" + name + "}",
                    r"\caption{" + inline(cap) + "}",
                    r"\label{fig:" + name + "}",
                    r"\end{figure}"]
            i += 1
            continue

        # headings
        if line.startswith("### "):
            close_list()
            out.append(r"\subsection{" + inline(re.sub(r"^###\s+(?:[0-9.]+\s+)?", "", line)) + "}")
        elif line.startswith("## "):
            close_list()
            out.append(r"\section{" + inline(re.sub(r"^##\s+(?:[0-9.]+\s+)?", "", line)) + "}")
        elif line.startswith("# "):
            pass                                   # chapter title handled by caller
        # block quote
        elif line.startswith(">"):
            close_list()
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip(">").strip())
                i += 1
            out += [r"\begin{quote}", inline(" ".join(q for q in quote if q)), r"\end{quote}"]
            continue
        # lists
        elif re.match(r"^[-*] ", line):
            if in_list != "itemize":
                close_list()
                out.append(r"\begin{itemize}")
                in_list = "itemize"
            out.append(r"\item " + inline(line[2:]))
        elif re.match(r"^\d+\. ", line):
            if in_list != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                in_list = "enumerate"
            out.append(r"\item " + inline(re.sub(r"^\d+\.\s+", "", line)))
        elif not line.strip():
            close_list()
            out.append("")
        else:
            if in_list:
                out[-1] += " " + inline(line.strip())    # continuation line
            else:
                out.append(inline(line))
        i += 1

    close_list()
    body = "\n".join(out)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return rf"\chapter{{{chapter_title}}}" + "\n\n" + body


PREAMBLE_SRC = Path("docs/thesis/latex/preamble.tex")

# listings config the upstream preamble does not carry, appended after it.
LST = r"""
%=============================================================================
% CODE LISTINGS (added for this thesis: it quotes shell and model output)
%=============================================================================
\usepackage{listings}
\lstset{
  basicstyle=\ttfamily\footnotesize,
  breaklines=true,
  breakatwhitespace=false,
  frame=single,
  rulecolor=\color{gray!40},
  backgroundcolor=\color{gray!5},
  columns=fullflexible,
  keepspaces=true,
  showstringspaces=false,
  literate={-}{{-}}1 {>}{{>}}1 {<}{{<}}1,
}
"""

FRONT = r"""
\begin{document}
\frontmatter

\begin{titlepage}
\begin{center}
\vspace*{1cm}

{\Large %(institution)s}\\[0.4cm]
{\large %(department)s}\\[2cm]

{\Huge\bfseries\color{chaptercolor} %(title)s}\\[0.8cm]
{\large\itshape %(subtitle)s}\\[2cm]

{\Large\textit{A thesis submitted in partial fulfilment\\
of the requirements for the degree of}}\\[0.8cm]

{\Large\bfseries %(degree)s}\\[0.3cm]
{\large in}\\[0.3cm]
{\Large\bfseries %(branch)s}\\[1.8cm]

{\large\bfseries %(author)s}\\[0.3cm]
{\large %(roll_number)s}\\[1.5cm]

%(supervisor_block)s

\vfill
{\large %(submission_date)s}
\end{center}
\end{titlepage}

\chapter*{Certificate}
\addcontentsline{toc}{chapter}{Certificate}

This is to certify that the thesis entitled \textbf{``%(title)s''} submitted by
\textbf{%(author)s} (%(roll_number)s) to %(institution)s in partial fulfilment
of the requirements for the award of the degree of %(degree)s in %(branch)s is
a record of bona fide work carried out under my supervision during the academic
year %(academic_year)s.

To the best of my knowledge the content of this thesis has not been submitted
to any other institute or university for the award of any degree or diploma.

\vspace{2.5cm}
\noindent\rule{6cm}{0.4pt}\\
%(supervisor)s\\
%(supervisor_designation)s\\
%(department)s\\
%(institution)s

\chapter*{Declaration}
\addcontentsline{toc}{chapter}{Declaration}

I declare that this thesis is my own work. Where the work of others has been
consulted it is acknowledged, and every quantitative claim is accompanied by
the code and committed artifact that produced it.

I further declare that this work has not been submitted, in whole or in part,
for any other degree or diploma at this or any other institution.

\vspace{2.5cm}
\noindent\rule{6cm}{0.4pt}\\
%(author)s\\
%(roll_number)s\\
%(submission_date)s

%(abstract)s

\chapter*{Acknowledgements}
\addcontentsline{toc}{chapter}{Acknowledgements}

\emph{To be written.}

\tableofcontents
\listoffigures
\listoftables

\mainmatter
"""


META = SRC / "metadata.yaml"


def load_metadata(allow_placeholders: bool) -> dict:
    """Read the front-matter values, and refuse to build on an unfilled one.

    The title page used to take its values from argparse defaults, so a build
    that forgot the flags emitted a PDF reading "university name" in italics --
    and nothing said so. The failure is silent and lands on the title page,
    which is the one page an examiner reads first. So it gates, in the same
    spirit as the annotation-rate and feasibility gates elsewhere in this
    project: a placeholder left in is an error, not a default.
    """
    import yaml

    if not META.exists():
        raise SystemExit(f"missing {META} -- front-matter values live there")
    meta = yaml.safe_load(META.read_text(encoding="utf-8")) or {}

    unfilled = sorted(k for k, v in meta.items()
                      if isinstance(v, str) and v.strip().startswith("FILL:"))
    if unfilled:
        print(f"{len(unfilled)} front-matter value(s) still unfilled in {META}:")
        for k in unfilled:
            print(f"    {k}: {meta[k]}")
        if not allow_placeholders:
            raise SystemExit(
                "\nRefusing to build. Fill these in, or pass --allow-placeholders\n"
                "for a draft build that prints them verbatim on the title page."
            )
        print("  --allow-placeholders given; building a DRAFT anyway.\n")
    return meta


def supervisor_block(meta: dict) -> str:
    """The supervisor lines, with the co-supervisor only when there is one."""
    sup = meta.get("supervisor", "")
    desig = meta.get("supervisor_designation", "")
    lines = [r"{\large\textit{Under the supervision of}}\\[0.6cm]",
             rf"{{\large\bfseries {sup}}}\\[0.2cm]",
             rf"{{\large {desig}}}"]
    co = (meta.get("co_supervisor") or "").strip()
    if co:
        lines += [r"\\[0.8cm]",
                  rf"{{\large\bfseries {co}}}\\[0.2cm]",
                  rf"{{\large {meta.get('co_supervisor_designation', '')}}}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--allow-placeholders", action="store_true",
                    help="build a draft even though metadata.yaml has FILL: values")
    args = ap.parse_args()
    meta = load_metadata(args.allow_placeholders)

    if not SRC.exists():
        print(f"missing {SRC}")
        return 1

    out = args.out
    (out / "chapters").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    # figures: the PDFs, because LaTeX prefers vector and they already exist
    n_fig = 0
    for pdf in sorted(FIGS.glob("*.pdf")):
        shutil.copy2(pdf, out / "figures" / pdf.name)
        n_fig += 1
    if BIB.exists():
        shutil.copy2(BIB, out / "references.bib")

    chapters = []
    abstract = ""
    for md in sorted(SRC.glob("[0-9]*.md")):
        text = md.read_text(encoding="utf-8")
        first = next((ln for ln in text.split("\n") if ln.startswith("# ")), "")
        title = re.sub(r"^#\s+(?:[0-9.]+\.?\s+)?", "", first) or md.stem
        stem = md.stem

        # The references chapter becomes a real bibliography, and the front
        # matter becomes an unnumbered Abstract -- neither is a chapter.
        if "references" in stem:
            continue
        if "front-matter" in stem:
            # Take only "## Abstract" onward, and stop before "## Contents".
            # The header above it -- subtitle, author, draft date -- is the
            # markdown reader's title page, and the LaTeX one is built from
            # metadata.yaml instead; letting it through printed the author
            # block and the word "Abstract" twice inside the Abstract chapter.
            # "## Contents" is a hand-written table \tableofcontents replaces.
            lines = text.split("\n")
            try:
                start = next(i for i, ln in enumerate(lines)
                             if ln.strip().lower() == "## abstract")
            except StopIteration:
                raise SystemExit(f"{md.name}: no '## Abstract' heading to cut at")
            end = next((i for i, ln in enumerate(lines)
                        if i > start and ln.strip().lower() == "## contents"),
                       len(lines))
            inner = "\n".join(lines[start + 1:end]).strip()
            body = convert("# Abstract\n\n" + inner, chapter_title="Abstract")
            body = body.split("\n", 1)[1].lstrip()      # drop the \chapter line
            abstract = (r"\chapter*{Abstract}" "\n"
                        r"\addcontentsline{toc}{chapter}{Abstract}" "\n\n" + body)
            print(f"  {md.name} -> abstract in main.tex "
                  f"({len(inner.split()):,} words, header and contents dropped)")
            continue

        tex = convert(text, chapter_title=title)
        # A label per chapter so \cref works if cross-references are added.
        # Lambda replacement, not a template string: re.sub would read the
        # backslash in \label as an escape and refuse to compile it.
        tex = re.sub(r"(\\chapter\{[^}]*\})",
                     lambda m: m.group(1) + "\n\\label{ch:" + stem + "}",
                     tex, count=1)
        (out / "chapters" / f"{stem}.tex").write_text(tex, encoding="utf-8")
        chapters.append(stem)
        print(f"  {md.name} -> chapters/{stem}.tex  ({len(tex.split()):,} words)")

    preamble = PREAMBLE_SRC.read_text(encoding="utf-8") + LST
    fields = {k: meta.get(k, "") for k in (
        "title", "subtitle", "author", "roll_number", "degree", "branch",
        "institution", "department", "supervisor", "supervisor_designation",
        "submission_date", "academic_year")}
    fields["supervisor_block"] = supervisor_block(meta)
    fields["abstract"] = abstract
    front = FRONT % fields
    body = "\n\n".join(rf"\input{{chapters/{c}}}" for c in chapters)
    main_tex = preamble + front + "\n" + body + r"""

\backmatter

% Nothing in the prose uses \cite -- the draft names its sources inline -- so
% \nocite{*} is what makes the bibliography print at all.
\nocite{*}
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""
    (out / "main.tex").write_text(main_tex, encoding="utf-8")

    (out / "README.md").write_text(
        "# LaTeX build of the thesis\n\n"
        "**Generated. Do not edit by hand** -- `python scripts/build_latex.py`\n"
        "overwrites everything here from `docs/thesis/*.md`, which is the\n"
        "single source. Edits made here are lost on the next build.\n\n"
        "## Compiling\n\n"
        "No TeX toolchain is needed to *produce* this; one is needed to compile.\n\n"
        "**Overleaf (easiest):** upload this whole folder, set `main.tex` as the\n"
        "main document, compile. Nothing else to configure.\n\n"
        "**Locally:**\n\n"
        "```bash\n"
        "latexmk -pdf main.tex\n"
        "```\n\n"
        "## Swapping in a university template\n\n"
        "Only `main.tex` carries the document class and front matter. To use a\n"
        "department template, replace its preamble and keep the\n"
        "`\\input{chapters/...}` lines -- the chapter files are template-agnostic.\n",
        encoding="utf-8")

    print(f"\nwrote {out}/  ({len(chapters)} chapters, {n_fig} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
