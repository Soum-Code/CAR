"""Static checks on the generated LaTeX, since no TeX toolchain is installed.

`build_latex.py` emits a project that cannot be compiled on this machine, so
the usual feedback loop -- run it and read the errors -- is unavailable. These
are the failures that would otherwise only surface on Overleaf: unbalanced
environments, unbalanced braces, unescaped specials outside verbatim blocks,
figures referencing files that were not copied, and a bibliography that would
print empty because nothing cites it.

    python scripts/check_latex.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("thesis-latex")


def strip_verbatim(text: str) -> str:
    return re.sub(r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text,
                  flags=re.S)


def strip_comments(text: str) -> str:
    """Drop everything from an unescaped % to end of line, as TeX does.

    Without this the preamble's `%=====` banner comments read as 92 stray
    percent signs. What the check is actually for is a % that arrived from
    converted prose -- "50%" written without a backslash -- which silently
    eats the rest of the line.
    """
    return re.sub(r"(?<!\\)%.*$", "", text, flags=re.M)


def main() -> int:
    if not ROOT.exists():
        print(f"missing {ROOT} -- run scripts/build_latex.py first")
        return 1

    issues = 0
    tex_files = sorted(ROOT.rglob("*.tex"))

    for f in tex_files:
        t = f.read_text(encoding="utf-8")
        body = strip_comments(strip_verbatim(t))

        begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", t))
        ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", t))
        for env in sorted(set(begins) | set(ends)):
            if begins[env] != ends[env]:
                print(f"  {f.name}: {env} begin={begins[env]} end={ends[env]}")
                issues += 1

        if t.count("{") != t.count("}"):
            print(f"  {f.name}: braces {t.count('{')} vs {t.count('}')}")
            issues += 1

        # a bare % comments out the rest of the line -- silent data loss
        bare = len(re.findall(r"(?<!\\)%", body))
        if bare:
            print(f"  {f.name}: {bare} unescaped %")
            issues += 1
        # Escaped specials come out FIRST. Stripping math before removing \$
        # makes the $...$ matcher pair a real delimiter with an escaped dollar
        # and swallow the text between -- which reported `80\%-of-\$40` as an
        # error when it is correct.
        stripped = re.sub(r"\\[&$#%_{}]", "", body)
        stripped = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", "",
                          stripped, flags=re.S)
        stripped = re.sub(r"\$[^$]*\$", "", stripped)
        for ch in ("&", "$", "#"):
            n = len(re.findall(re.escape(ch), stripped))
            if n:
                print(f"  {f.name}: {n} unescaped {ch} outside tabular/math")
                issues += 1

    # every \includegraphics resolves
    for f in tex_files:
        for name in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}",
                               f.read_text(encoding="utf-8")):
            hits = list((ROOT / Path(name).parent).glob(Path(name).name + ".*"))
            if not hits:
                print(f"  {f.name}: missing figure {name}")
                issues += 1

    # a bibliography with no citations prints nothing
    main_tex = (ROOT / "main.tex").read_text(encoding="utf-8")
    cites = sum(len(re.findall(r"\\cite\w*\{", f.read_text(encoding="utf-8")))
                for f in tex_files)
    if r"\bibliography{" in main_tex and cites == 0 and r"\nocite{*}" not in main_tex:
        print("  main.tex: bibliography is included but nothing cites it and "
              "there is no \\nocite{*} -- it would print empty")
        issues += 1

    bib = ROOT / "references.bib"
    n_bib = len(re.findall(r"^@", bib.read_text(encoding="utf-8"), re.M)) if bib.exists() else 0

    print()
    print(f"{len(tex_files)} .tex files, {n_bib} bibliography entries, "
          f"{len(list((ROOT / 'figures').glob('*')))} figures")
    print(f"issues: {issues}")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
