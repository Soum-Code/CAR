"""Style diagnostics for the paper draft.

check_prose.py does this for the thesis chapters; the paper is a separate
document with its own file, so it needs its own pass. Reports the habits a
reader notices and an author does not: em dashes used as a rhythm crutch,
sentences that all run the same length, openings reused across paragraphs, and
bold applied to things that are not labels.

    python scripts/check_paper_prose.py
"""

from __future__ import annotations

import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PAPER = Path("docs/paper/paper.md")


def sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^\|.*$", " ", text, flags=re.M)
    text = re.sub(r"^[#>*-].*$", " ", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
            if len(s.strip()) > 15]


def main() -> int:
    if not PAPER.exists():
        print(f"missing {PAPER}")
        return 1
    t = PAPER.read_text(encoding="utf-8")
    sents = sentences(t)
    lens = [len(s.split()) for s in sents]

    print(f"{len(sents)} sentences, mean {statistics.mean(lens):.1f} words, "
          f"sd {statistics.pstdev(lens):.1f}")
    short = sum(1 for n in lens if n <= 8) / len(lens)
    long_ = sum(1 for n in lens if n >= 30) / len(lens)
    print(f"  short (<=8 words): {short:.1%}   long (>=30): {long_:.1%}")
    if short < 0.08:
        print("  -- too few short sentences; rhythm will read uniform")

    em = t.count("—")
    print(f"\nem dashes: {em}  ({em / max(1, len(sents)):.2f} per sentence)")
    if em > len(sents) * 0.15:
        print("  -- reads as a tic; prefer commas, parentheses or a full stop")

    print("\nsentence openings used 3+ times:")
    op = Counter(" ".join(s.split()[:2]).lower().strip(",") for s in sents)
    for k, v in op.most_common(10):
        if v >= 3:
            print(f"  {v:>3}x  {k}")

    bold = re.findall(r"\*\*([^*]+)\*\*", t)
    print(f"\nbold spans: {len(bold)}")
    wordy = [b for b in bold if len(b.split()) > 6]
    if wordy:
        print(f"  {len(wordy)} are long enough to be emphasis rather than a label:")
        for b in wordy[:6]:
            print(f"    {b[:70]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
