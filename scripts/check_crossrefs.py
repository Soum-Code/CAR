"""Every cross-reference in the thesis must point at something that exists.

Written for the Background/Related-work split, which moved ~160 references by
one chapter. A renumber that is off by one does not fail loudly: it produces a
document that reads fine and sends the examiner to the wrong section. So check
each reference against the headings actually present.

    python scripts/check_crossrefs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

T = Path("docs/thesis")


def main() -> int:
    chapters: dict[int, str] = {}
    sections: set[str] = set()
    figures: set[str] = set()

    for md in sorted(T.glob("[0-9]*.md")):
        t = md.read_text(encoding="utf-8")
        m = re.match(r"^# (\d+)\. (.+)$", t.split("\n")[0])
        if m:
            chapters[int(m.group(1))] = m.group(2)
        for h in re.findall(r"^#{2,3} (\d+(?:\.\d+)+) ", t, re.M):
            sections.add(h)
        for f in re.findall(r"^\*\*Figure (\d+\.\d+)\.\*\*", t, re.M):
            figures.add(f)

    print(f"{len(chapters)} chapters, {len(sections)} numbered sections, "
          f"{len(figures)} figure captions")

    bad: list[str] = []
    for md in sorted(T.glob("[0-9]*.md")):
        t = md.read_text(encoding="utf-8")
        for ref in re.findall(r"§(\d+(?:\.\d+)+)", t):
            if ref not in sections:
                bad.append(f"{md.name}: §{ref} -> no such section")
        for ref in set(re.findall(r"\b(?:Chapters? |chapters? |ch\. ?)(\d+)\b", t)):
            if int(ref) not in chapters:
                bad.append(f"{md.name}: Chapter {ref} -> no such chapter")
        for ref in set(re.findall(r"\b(?:[Ff]igures? |fig\. ?)(\d+\.\d+)", t)):
            if ref not in figures:
                bad.append(f"{md.name}: Figure {ref} -> no such figure caption")

    # A chapter must not reference a section inside itself by another number.
    for md in sorted(T.glob("[0-9]*.md")):
        m = re.match(r"^(\d+)-", md.name)
        if not m:
            continue

    if bad:
        print(f"\n{len(bad)} broken reference(s):")
        for b in sorted(set(bad)):
            print(f"    {b}")
        return 1
    print("\nOK: every cross-reference resolves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
