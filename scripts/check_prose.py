"""Diagnose prose habits in the draft, and guard an edit pass. [CPU]

Two jobs.

DIAGNOSE (default): find the tics a careful reader notices and the author does
not -- sentence openings reused across paragraphs, stock constructions, a
monotone sentence-length rhythm, words leaned on.

GUARD (--baseline / --compare): an editing pass must not move a single number
or flip a hedge. This thesis has had roughly seventy-five errors found in it by
recomputation; an edit that silently changed 0.6968 to 0.697, or "spans zero"
to "is worse", would undo that work and nothing else would catch it. So take a
fingerprint before editing and diff it after.

    python scripts/check_prose.py                    # diagnose
    python scripts/check_prose.py --baseline /tmp/b.json
    ...edit...
    python scripts/check_prose.py --compare /tmp/b.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

SRC = Path("docs/thesis")

# Constructions worth counting. Not banned -- a thesis may legitimately say
# "that is the finding" once. Counted so repetition becomes visible.
TICS = {
    "and that is the X": r"\b(?:and )?that is (?:the|what|why)\b",
    "which is the X": r"\bwhich is (?:the|what|why|exactly)\b",
    "It is not X, it is Y": r"\bis not\b[^.]{0,60}\bit is\b",
    "worth Xing": r"\bworth (?:\w+ing)\b",
    "the point is": r"\bthe (?:whole )?point\b",
    "exactly the": r"\bexactly (?:the|what|why|how)\b",
    "precisely": r"\bprecisely\b",
    "in fact": r"\bin fact\b",
    "It is worth noting": r"\b[Ii]t is worth (?:noting|being)\b",
    "and it is": r"\band it is\b",
    "rather than": r"\brather than\b",
    "not X but Y": r"\bnot\b[^.]{0,40}\bbut\b",
}

HEDGES = ["spans zero", "excludes zero", "not significant", "suggestive",
          "unsupported", "refuted", "null", "measured", "modelled",
          "simulated", "projected", "estimate", "floor", "ceiling"]


def sentences(text: str) -> list[str]:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"^\|.*$", " ", text, flags=re.M)
    text = re.sub(r"^[#>].*$", " ", text, flags=re.M)
    text = re.sub(r"\s+", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]


def numbers(text: str) -> Counter:
    """Every numeric literal, so an edit that moves one is visible."""
    return Counter(re.findall(r"\d+(?:[.,]\d+)*%?", text))


def fingerprint() -> dict:
    fp = {}
    for md in sorted(SRC.glob("[0-9]*.md")):
        t = md.read_text(encoding="utf-8")
        fp[md.name] = {
            "numbers": dict(numbers(t)),
            "hedges": {h: t.lower().count(h) for h in HEDGES},
        }
    return fp


def diagnose() -> int:
    all_sents, per_file = [], {}
    for md in sorted(SRC.glob("[0-9]*.md")):
        t = md.read_text(encoding="utf-8")
        s = sentences(t)
        per_file[md.name] = s
        all_sents += s

    print("=" * 76)
    print("PROSE DIAGNOSIS")
    print("=" * 76)

    lens = [len(s.split()) for s in all_sents]
    print(f"{len(all_sents)} sentences, mean {sum(lens)/len(lens):.1f} words")
    buckets = Counter(min(l // 10 * 10, 50) for l in lens)
    for b in sorted(buckets):
        bar = "#" * round(buckets[b] / len(lens) * 60)
        label = f"{b}-{b+9}" if b < 50 else "50+"
        print(f"  {label:>6} words  {buckets[b]:>4}  {bar}")
    short = sum(1 for l in lens if l <= 8) / len(lens)
    print(f"  short sentences (<=8 words): {short:.1%}"
          f"   {'-- little variation, prose will feel uniform' if short < 0.10 else ''}")

    print("\nSENTENCE OPENINGS used 5+ times (first two words):")
    op = Counter(" ".join(s.split()[:2]).lower().strip(",") for s in all_sents)
    for k, v in op.most_common(14):
        if v >= 5:
            print(f"  {v:>3}x  {k}")

    print("\nSTOCK CONSTRUCTIONS:")
    joined = " ".join(all_sents)
    for name, pat in sorted(TICS.items(),
                            key=lambda kv: -len(re.findall(kv[1], joined, re.I))):
        n = len(re.findall(pat, joined, re.I))
        if n:
            worst = max(per_file, key=lambda f: len(re.findall(pat, " ".join(per_file[f]), re.I)))
            print(f"  {n:>3}x  {name:<24} (most in {worst})")

    print("\nCONTENT WORDS used most (excluding the obvious):")
    stop = set("""the a an and or but is are was were be been it its this that
        these those of to in on at for with as by from not no if then than so
        which what when where who whom whose all any both each few more most
        other some such only own same too very can will just should now they
        them their there here we our us you your he she his her i do does did
        done have has had having would could may might must shall one two
        three first second last also into over under about""".split())
    words = Counter(w for w in re.findall(r"[a-z']+", joined.lower())
                    if w not in stop and len(w) > 3)
    print("  " + "   ".join(f"{w}({n})" for w, n in words.most_common(16)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--compare", type=Path)
    args = ap.parse_args()

    if args.baseline:
        args.baseline.write_text(json.dumps(fingerprint(), indent=1), encoding="utf-8")
        n = sum(len(v["numbers"]) for v in fingerprint().values())
        print(f"baseline written to {args.baseline} ({n} distinct numeric literals)")
        return 0

    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        after = fingerprint()
        bad = 0
        for fname in sorted(set(before) | set(after)):
            b = before.get(fname, {"numbers": {}, "hedges": {}})
            a = after.get(fname, {"numbers": {}, "hedges": {}})
            for num in sorted(set(b["numbers"]) | set(a["numbers"])):
                nb, na = b["numbers"].get(num, 0), a["numbers"].get(num, 0)
                if nb != na:
                    print(f"  {fname}: number {num!r} {nb} -> {na}")
                    bad += 1
            for h in HEDGES:
                hb, ha = b["hedges"].get(h, 0), a["hedges"].get(h, 0)
                if hb != ha:
                    print(f"  {fname}: hedge {h!r} {hb} -> {ha}")
                    bad += 1
        print(f"\nchanges to numbers or hedges: {bad}")
        if bad:
            print("Every one must be deliberate. An edit pass should move NEITHER.")
        return 1 if bad else 0

    return diagnose()


if __name__ == "__main__":
    raise SystemExit(main())
