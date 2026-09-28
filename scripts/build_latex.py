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
\vspace*{1.5cm}

{\Large %(institution)s}\\[0.4cm]
{\large %(department)s}\\[2.5cm]

{\Huge\bfseries\color{chaptercolor} %(title)s}\\[2cm]

{\Large\textit{A thesis submitted in partial fulfilment\\
of the requirements for the degree of}}\\[0.8cm]

{\Large\bfseries %(degree)s}\\[2cm]

{\large %(author)s}\\[0.4cm]
{\large Supervisor: %(supervisor)s}\\[1.5cm]

{\large \today}

\vfill
\end{center}
\end{titlepage}

\chapter*{Declaration}
\addcontentsline{toc}{chapter}{Declaration}

I declare that this thesis is my own work. Where the work of others has been
consulted it is acknowledged, and every quantitative claim is accompanied by
the code and committed artifact that produced it.

\vspace{1.5cm}
\noindent\rule{6cm}{0.4pt}\\
%(author)s\\
\today

%(abstract)s

\chapter*{Acknowledgements}
\addcontentsline{toc}{chapter}{Acknowledgements}

\emph{To be written.}

\tableofcontents
\listoffigures
\listoftables

\mainmatter
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--title", default="What Step-Level Verification Certifies, "
                                       "and What It Misses")
    ap.add_argument("--author", default="Somnath Reddy")
    ap.add_argument("--supervisor", default="\\emph{supervisor name}")
    ap.add_argument("--institution", default="\\emph{university name}")
    ap.add_argument("--department", default="Department of Computer Science")
    ap.add_argument("--degree", default="Master of Technology")
    ap.add_argument("--date", default=r"\today")
    args = ap.parse_args()

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
            body = convert(text, chapter_title=title)
            body = body.split("\n", 1)[1].lstrip()      # drop the \chapter line
            abstract = (r"\chapter*{Abstract}" "\n"
                        r"\addcontentsline{toc}{chapter}{Abstract}" "\n\n" + body)
            print(f"  {md.name} -> abstract in main.tex")
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
    front = FRONT % {
        "title": args.title, "author": args.author,
        "supervisor": args.supervisor, "institution": args.institution,
        "department": args.department, "degree": args.degree,
        "abstract": abstract,
    }
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
