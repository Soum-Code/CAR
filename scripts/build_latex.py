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


def _mapped(ch: str, *, where: str = "the prose") -> str:
    """The UNICODE lookup, with the error both call sites should raise."""
    if ch not in UNICODE:
        raise SystemExit(
            f"unmapped non-ASCII character {ch!r} (U+{ord(ch):04X}) in {where}. "
            f"Add it to UNICODE in {__file__} rather than stripping it."
        )
    return UNICODE[ch]


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
            out.append(_mapped(ch))
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

    # links: keep the text, footnote the URL (a printed thesis cannot be clicked).
    #
    # The \footnote has to be stashed the same way a code span is. esc() runs
    # AFTER this substitution, so a \footnote inserted here was being escaped
    # into \textbackslash{}footnote\{...\} and printed as visible text -- every
    # footnoted URL in the document, silently, until something finally compiled
    # it. Returning the label unescaped is the other half: esc() escapes it once
    # below, where returning esc(label) escaped it twice.
    raw: list[str] = []

    def link(m):
        label, url = m.group(1), m.group(2)
        if not url.startswith("http"):
            return label        # anchors and repo paths: keep the words only
        raw.append(r"\footnote{\url{" + url + "}}")
        return label + f"\x01{len(raw) - 1}\x01"

    text = re.sub(r"(?<!!)\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = esc(text)
    for i, s in enumerate(raw):
        text = text.replace(f"\x01{i}\x01", s)
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
        # ...and its own unicode mapping. Spans are stashed BEFORE esc() runs,
        # so they never reach the gate there. A `47 × 3 = 131` in the draft was
        # reaching main.tex as a raw U+00D7 inside \texttt, which is exactly
        # the silent pass-through that gate exists to prevent.
        safe = "".join(
            ch if ord(ch) <= 127 else _mapped(ch, where="a code span")
            for ch in safe
        )
        # A long path or command in \texttt is one unbreakable token, and the
        # remaining overfull lines in the compiled PDF were all of this shape:
        # `python scripts/gpu_semantic_scope.py`, `runs/uncertainty_*.jsonl`.
        # Offering a break after each separator lets them wrap where a reader
        # would expect, instead of running into the margin.
        if len(s) > 18:
            for sep in ("/", r"\_", "-", "."):
                safe = safe.replace(sep, sep + r"\allowbreak{}")
        text = text.replace(f"\x00{i}\x00", r"\texttt{" + safe + "}")
    return text


# Roughly how many characters of \small text fit on one line of the text
# block. Measured from the compiled output rather than derived: tables under
# this sit inside the margin, tables over it overflow.
TABLE_CHAR_BUDGET = 95


def _visible(cell: str) -> int:
    """Length as it will print, ignoring markdown that carries no width."""
    return len(re.sub(r"\*\*|\*|`|\[|\]\([^)]*\)", "", cell))


def convert_table(rows: list[str]) -> str:
    """Markdown pipe table to a booktabs table float.

    Narrow tables get a plain `tabular` with `l` columns, which centres well.
    Wide ones get `tabularx` at \\textwidth with the prose-carrying columns as
    wrapping `L`, because `l` columns never break a line: a cell of prose in
    one runs straight off the page edge, which is how chapter 9's refuted-claim
    table ended up 952pt too wide -- more than twice the text block -- in the
    first compiled PDF.
    """
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]          # cells[1] is the --- rule
    ncol = len(header)
    body = [(r + [""] * ncol)[:ncol] for r in body]

    widths = [max(_visible(r[i]) for r in [header] + body) for i in range(ncol)]
    natural = sum(widths) + 3 * ncol

    if natural <= TABLE_CHAR_BUDGET:
        spec = "l" * ncol
        env, open_arg, size = "tabular", "", r"\small"
    else:
        # Wrap the columns wide enough to be prose; keep the rest at natural
        # width so numeric columns do not get stretched into empty space.
        cut = max(12, sorted(widths)[-1] // 3)
        spec = "".join("L" if w > cut else "l" for w in widths)
        if "L" not in spec:                      # all columns similar: wrap all
            spec = "L" * ncol
        env, open_arg = "tabularx", r"{\textwidth}"
        size = r"\footnotesize" if natural > TABLE_CHAR_BUDGET * 1.6 else r"\small"

    out = [r"\begin{table}[htbp]", r"\centering", size,
           rf"\begin{{{env}}}{open_arg}{{{spec}}}", r"\toprule",
           " & ".join(inline(h) for h in header) + r" \\", r"\midrule"]
    for row in body:
        out.append(" & ".join(inline(c) for c in row) + r" \\")
    out += [r"\bottomrule", rf"\end{{{env}}}", r"\end{table}"]
    return "\n".join(out)


def _starts_block(ln: str) -> bool:
    """Does this line begin a new markdown construct rather than continue one?"""
    s = ln.strip()
    return (not s
            or s.startswith(("#", ">", "|", "```", "!["))
            or bool(re.match(r"^[-*] ", s))
            or bool(re.match(r"^\d+\. ", s)))


def _gather(lines: list[str], i: int, first: str) -> tuple[str, int]:
    """Join a paragraph or list item into one string before converting it.

    inline() used to run per line, so any span crossing a line break lost its
    pair: chapter 2's *Reliable and Efficient Agentic Workflow\\nExecution*
    reached the PDF with literal asterisks, because neither half matched the
    italic pattern on its own. Markdown treats a paragraph as one unit, so the
    converter has to as well.
    """
    parts = [first]
    while i + 1 < len(lines) and not _starts_block(lines[i + 1]):
        i += 1
        parts.append(lines[i].strip())
    return " ".join(p for p in parts if p), i


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
            item, i = _gather(lines, i, line[2:])
            out.append(r"\item " + inline(item))
        elif re.match(r"^\d+\. ", line):
            if in_list != "enumerate":
                close_list()
                out.append(r"\begin{enumerate}")
                in_list = "enumerate"
            item, i = _gather(lines, i, re.sub(r"^\d+\.\s+", "", line))
            out.append(r"\item " + inline(item))
        elif re.fullmatch(r"-{3,}|\*{3,}|_{3,}", line.strip()):
            # A markdown horizontal rule. Passed through untouched it reaches
            # TeX as "---", which sets an em dash: the paper printed a stray
            # dash between the acknowledgment and the references, and the
            # thesis abstract carried three. Sectioning already separates these
            # documents, so the rule becomes vertical space rather than a line.
            close_list()
            out.append(r"\medskip")
        elif not line.strip():
            close_list()
            out.append("")
        else:
            if in_list:
                out[-1] += " " + inline(line.strip())    # continuation line
            else:
                para, i = _gather(lines, i, line.strip())
                out.append(inline(para))
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
  extendedchars=true,
  % Code blocks are emitted verbatim, so esc() never sees them and the unicode
  % map above does not apply. The draft quotes formulas inside them --
  % "(mu - alpha) / (1 - alpha)" with real Greek and a real U+2212 -- and
  % inputenc rejects those inside a listing with "Invalid UTF-8 byte sequence".
  % listings has to be told about each one separately. LST_LITERATE below is
  % the same set, and build_latex.py gates on the two staying in sync.
  literate={-}{{-}}1 {>}{{>}}1 {<}{{<}}1
           {μ}{{$\mu$}}1 {α}{{$\alpha$}}1
           {−}{{$-$}}1 {·}{{$\cdot$}}1,
}
"""

# Every non-ASCII character the listings `literate` above can render. A code
# block containing anything else produces an invalid PDF, so the build stops.
LST_LITERATE = {"μ", "α", "−", "·"}


def check_listing_chars(md: str, name: str) -> None:
    """Refuse to emit a code block carrying unicode listings cannot render.

    The prose path gates on this in esc(); the verbatim path had no gate at
    all, which is how a real U+03BC reached a listing and killed the compile
    with an error pointing at \\lst@EC rather than at the character.
    """
    for block in re.findall(r"^```.*?^```", md, re.S | re.M):
        for ch in block:
            if ord(ch) > 127 and ch not in LST_LITERATE:
                raise SystemExit(
                    f"{name}: code block contains {ch!r} (U+{ord(ch):04X}), "
                    f"which the listings literate map cannot render. Add it to "
                    f"LST and LST_LITERATE in {__file__}, or use ASCII in the "
                    f"code block."
                )

FRONT = r"""
\begin{document}
\frontmatter

\begin{titlepage}
\begin{center}
\vspace*{1cm}

{\Large %(institution)s}\\[0.4cm]
{\large %(department)s}\\[2cm]

{\Huge\bfseries\color{chaptercolor} %(title)s}\\[0.7cm]
{\large\itshape %(subtitle)s}\\[1.4cm]

{\Large\textit{A thesis submitted in partial fulfilment\\
of the requirements for the degree of}}\\[0.7cm]

{\Large\bfseries %(degree)s}\\[0.25cm]
{\large in}\\[0.25cm]
{\Large\bfseries %(branch)s}\\[1.2cm]

{\large\bfseries %(author)s}\\[0.25cm]
{\large %(roll_number)s}\\[1cm]

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
a record of bona fide work carried out under %(our_supervision)s during the
academic year %(academic_year)s.

To the best of %(our_knowledge)s the content of this thesis has not been
submitted to any other institute or university for the award of any degree or
diploma.

%(signature_blocks)s

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

%(acknowledgements)s

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


def _person(meta: dict, prefix: str) -> tuple[str, str, str, str] | None:
    """The four lines describing one supervisor, or None if unnamed."""
    name = esc(str(meta.get(prefix, "") or "").strip())
    if not name:
        return None
    return (name,
            esc(str(meta.get(f"{prefix}_designation", "") or "").strip()),
            esc(str(meta.get(f"{prefix}_organisation", "") or "").strip()),
            esc(str(meta.get(f"{prefix}_location", "") or "").strip()))


def signature_blocks(meta: dict) -> str:
    """Certificate signature lines: one per supervisor, side by side.

    Most M.Tech programmes want the internal faculty guide's signature even
    when the work was mentored externally, so both appear, each over their own
    affiliation rather than over the student's.
    """
    people = [p for p in (_person(meta, "supervisor"),
                          _person(meta, "co_supervisor")) if p]
    if not people:
        return ""

    def block(p, width):
        name, desig, org, loc = p
        body = "\\\\\n".join(x for x in (name, desig, org, loc) if x)
        # \raggedright, because these columns are narrow enough that justified
        # text hyphenates an institution name across lines ("Technol-ogy").
        return (rf"\begin{{minipage}}[t]{{{width}\textwidth}}" "\n"
                r"\raggedright\small" "\n"
                r"\rule{5.5cm}{0.4pt}\\" "\n" + body + "\n" + r"\end{minipage}")

    if len(people) == 1:
        return r"\vspace{2.5cm}" "\n" r"\noindent" "\n" + block(people[0], "0.6")
    return (r"\vspace{2.5cm}" "\n" r"\noindent" "\n"
            + block(people[0], "0.46") + "\n" + r"\hfill" + "\n"
            + block(people[1], "0.46"))


def supervisor_block(meta: dict) -> str:
    """Title-page supervision lines, listing whoever is named."""
    people = [p for p in (_person(meta, "supervisor"),
                          _person(meta, "co_supervisor")) if p]
    if not people:
        return ""
    label = ("Under the supervision of" if len(people) == 1
             else "Under the supervision of")
    lines = [rf"{{\large\textit{{{label}}}}}\\[0.5cm]"]
    # The student's own institution is already printed at the top of this page,
    # so repeating it under an internal guide's name costs three lines and
    # pushes the submission date onto a second page. Only an affiliation that
    # DIFFERS from the student's is worth the space.
    home = {esc(str(meta.get("institution", ""))),
            esc(str(meta.get("department", "")))}
    for n, (name, desig, org, loc) in enumerate(people):
        if n:
            lines.append(r"\\[0.5cm]")
        lines.append(rf"{{\large\bfseries {name}}}\\[0.15cm]")
        lines.append(rf"{{\large {desig}}}")
        if org and org not in home:
            where = ", ".join(x for x in (org, loc) if x)
            lines.append(rf"\\[0.15cm]{{\normalsize {where}}}")
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

    # Clear the chapter directory first. The build only ever writes, so a
    # renumbering left both 03-framework.tex and 04-framework.tex on disk --
    # harmless while main.tex \inputs only the current set, but exactly the
    # state in which a stale chapter gets picked up later and nobody notices.
    for stale in (out / "chapters").glob("*.tex"):
        stale.unlink()

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
        check_listing_chars(text, md.name)
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
            # Star the headings inside it. The abstract is an unnumbered
            # chapter, so a numbered \section under it comes out as "0.1
            # Declaration on what is measured" in the table of contents.
            body = body.replace(r"\section{", r"\section*{").replace(
                r"\subsection{", r"\subsection*{")
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
        "supervisor_organisation", "supervisor_location",
        "submission_date", "academic_year")}
    # These land in LaTeX verbatim, and real-world values carry specials:
    # "L&T EduTech" alone would break the compile on an unescaped ampersand.
    fields = {k: esc(str(v)) for k, v in fields.items()}
    fields["supervisor_block"] = supervisor_block(meta)
    fields["signature_blocks"] = signature_blocks(meta)
    # "under my supervision" reads wrong over two signature lines.
    n_sup = len([p for p in (_person(meta, "supervisor"),
                             _person(meta, "co_supervisor")) if p])
    fields["our_supervision"] = ("our joint supervision" if n_sup > 1
                                 else "my supervision")
    fields["our_knowledge"] = "our knowledge" if n_sup > 1 else "my knowledge"

    # Acknowledgements live in their own markdown file, like every other piece
    # of prose here. The build warns while the scaffold's bracketed slots are
    # still in it -- they are placeholders for people only the author can name,
    # and printing "[Family. Yours to write]" in a submitted thesis would be
    # worse than the "To be written." this replaced.
    ack_md = SRC / "acknowledgements.md"
    if ack_md.exists():
        raw = ack_md.read_text(encoding="utf-8")
        raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
        body = convert(raw, chapter_title="Acknowledgements")
        fields["acknowledgements"] = body.split("\n", 1)[1].lstrip()
        if re.search(r"\[[^\]]{25,}\]", raw):
            print("  NOTE: acknowledgements.md still has unfilled [...] slots")
    else:
        fields["acknowledgements"] = r"\emph{To be written.}"
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
