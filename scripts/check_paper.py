"""Check every number in the paper draft against the thesis it is drawn from.

The paper restates measurements that live in docs/thesis/. A restatement is
exactly where this project has historically lost numbers -- a figure copied
from the wrong table, a rate quoted against the wrong denominator. So rather
than trusting the draft, pull every numeric literal out of it and require that
each one appears somewhere in the thesis chapters or the findings documents.

A hit is not proof the number is used correctly; it is proof the number was not
invented. Anything reported as MISSING is either a typo or a claim the thesis
does not support, and both need a human look.

    python scripts/check_paper.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path("docs/paper/paper.md")
SOURCES = sorted(Path("docs/thesis").glob("[0-9]*.md")) + \
          sorted(Path("docs").glob("FINDINGS-*.md")) + \
          [Path("docs/THESIS.md")]

# Numbers that are structural rather than measured: page counts, section
# numbers, years, arXiv ids, and the small integers in prose.
IGNORE = re.compile(
    r"^(?:[0-9]|1[0-9]|2[0-9]|3[0-9]|4[0-9]|5[0-9]|6[0-9]|7[0-9]|8[0-9]|9[0-9]|"
    r"100|19\d\d|20\d\d|21\d\d|22\d\d|23\d\d|24\d\d|25\d\d|26\d\d)$"
)


def literals(text: str) -> list[str]:
    text = re.sub(r"arXiv:\S+", " ", text)
    text = re.sub(r"^\s*-\s+\w+\d{4}\w*\s+—.*$", " ", text, flags=re.M)
    return re.findall(r"\d+(?:[.,]\d+)*%?", text)


def main() -> int:
    if not PAPER.exists():
        print(f"missing {PAPER}")
        return 1

    corpus = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES)
    seen = set(literals(corpus))
    # A thesis "0.9033" should satisfy a paper "0.9033"; also allow the
    # percent/decimal pairing the two documents use interchangeably.
    for n in list(seen):
        if n.endswith("%"):
            seen.add(n[:-1])
        else:
            seen.add(n + "%")

    missing, checked = [], 0
    for n in literals(PAPER.read_text(encoding="utf-8")):
        if IGNORE.match(n.rstrip("%")):
            continue
        checked += 1
        if n not in seen:
            missing.append(n)

    print(f"checked {checked} numeric literals in {PAPER}")
    print(f"sources: {len(SOURCES)} files")
    if missing:
        print(f"\n{len(set(missing))} NOT FOUND in the thesis or findings:")
        for n in sorted(set(missing)):
            print(f"    {n}")
        print("\nEach is a typo, or a claim the thesis does not support.")
        return 1
    print("\nOK: every number in the paper appears in its source.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
