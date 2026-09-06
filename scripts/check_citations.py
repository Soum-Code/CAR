"""Every arXiv ID cited in the thesis must have a bibliography entry. [CPU]

    python scripts/check_citations.py

A bibliography drifts silently: a chapter gains a citation, the .bib does not,
and nobody notices until a submission deadline. This is the cheapest possible
guard -- scan the prose for arXiv IDs, scan the .bib for `eprint` fields,
compare the two sets.

Also reports which entries are still marked UNVERIFIED, because an entry whose
author list has not been checked against the source is not ready to submit even
though it is present.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

THESIS = Path("docs/thesis")
BIB = THESIS / "references.bib"
ALSO_SCAN = [Path("README.md"), Path("docs/THESIS.md"), Path("docs/POSITIONING.md")]

# 2110.14168 / 2208.02814v2 / arXiv:2312.08935 -- all forms used in the prose.
_ARXIV = re.compile(r"(?:arxiv\.org/(?:abs|html|pdf)/|arXiv:)(\d{4}\.\d{4,5})")
_EPRINT = re.compile(r"eprint\s*=\s*\{(\d{4}\.\d{4,5})\}")
_ENTRY = re.compile(r"@\w+\{([^,]+),")


def cited(paths) -> dict[str, set[str]]:
    """arXiv id -> the files citing it."""
    out: dict[str, set[str]] = {}
    for p in paths:
        if not p.exists():
            continue
        for m in _ARXIV.finditer(p.read_text(encoding="utf-8")):
            out.setdefault(m.group(1), set()).add(p.name)
    return out


_FIELD = re.compile(r"(\w+)\s*=\s*\{(.*?)\}(?:,\s*\n|\s*\n\})", re.S)


def parse_bib(text: str) -> list[dict]:
    """Crude but sufficient: entries here are hand-written and uniform."""
    out = []
    for chunk in re.split(r"\n@", "\n" + text)[1:]:
        head, _, body = chunk.partition("{")
        key = body.split(",", 1)[0].strip()
        fields = {k.lower(): " ".join(v.split()) for k, v in _FIELD.findall(chunk)}
        fields.update(kind=head.strip().lower(), key=key)
        out.append(fields)
    return out


_ACCENTS = {
    r"\`e": "è", r"\'e": "é", r"\"o": "ö", r"\"u": "ü", r"\'a": "á",
    r"\`a": "à", r"\'o": "ó", r"\~n": "ñ", r"\c c": "ç",
}


def _detex(s: str) -> str:
    """BibTeX accent escapes -> the characters they stand for.

    The .bib keeps `Cand{\\`e}s` because that is what a LaTeX build needs; the
    markdown render must not show the backslash.
    """
    s = s.replace("{", "").replace("}", "")
    for tex, ch in _ACCENTS.items():
        s = s.replace(tex, ch)
    return " ".join(s.split())


def render_markdown(entries, section_of) -> str:
    """The bibliography as prose, generated from the .bib.

    Written rather than hand-maintained so the markdown draft and the .bib
    cannot disagree -- which they would, on the first citation added to one and
    not the other.
    """
    lines = [
        "# References",
        "",
        "Generated from [`references.bib`](references.bib) by",
        "`python scripts/check_citations.py --markdown`. Do not edit by hand.",
        "",
        "Entries marked **[unverified]** have an arXiv identifier and a title",
        "recorded during the literature review, but no author list confirmed",
        "against the source. They must be checked before submission. No author",
        "list is guessed: a missing one is recoverable, an invented one is not.",
        "",
    ]
    for section, keys in section_of.items():
        picked = [e for e in entries if e["key"] in keys]
        if not picked:
            continue
        lines += [f"## {section}", ""]
        for e in picked:
            author = _detex(e.get("author", "")) or "—"
            bits = [f"**{e.get('title', e['key'])}**"]
            venue = e.get("booktitle") or e.get("journal") or e.get("publisher")
            meta = ", ".join(x for x in (author, venue, e.get("year")) if x)
            bits.append(meta)
            if e.get("url"):
                bits.append(f"[{e.get('eprint', 'link')}]({e['url']})")
            flag = " **[unverified]**" if "UNVERIFIED" in e.get("note", "") else ""
            note = re.sub(r"^UNVERIFIED[^.]*\.\s*", "", e.get("note", ""))
            lines.append(f"- `{e['key']}`{flag} — " + " · ".join(bits))
            if note:
                lines.append(f"  <br>{note}")
        lines.append("")
    return "\n".join(lines) + "\n"


SECTIONS = {
    "Conformal prediction and risk control": {
        "angelopoulos2024crc", "vovk2005alrw", "gibbs2021aci", "barber2023beyond",
        "khosravi2026csa", "kotte2026certify"},
    "Benchmarks and step-labelled data": {
        "cobbe2021gsm8k", "geva2021strategyqa", "wang2024mathshepherd"},
    "Process supervision and step-level scoring": {
        "lightman2023verify", "uheads2025", "stepuncertainty2026",
        "farquhar2024semantic"},
    "Verification, self-correction and propagation": {
        "huang2024selfcorrect", "singh2026snowball", "sherlock2025", "ares2025"},
    "Selective labels": {"lakkaraju2017selective"},
    "Models used in the measurements": {
        "jiang2023mistral", "qwen2024qwen25", "dubey2024llama3"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true",
                    help="also scan the README and the findings documents")
    ap.add_argument("--markdown", action="store_true",
                    help="regenerate docs/thesis/10-references.md from the .bib")
    args = ap.parse_args()

    if args.markdown:
        entries = parse_bib(BIB.read_text(encoding="utf-8"))
        known = {k for ks in SECTIONS.values() for k in ks}
        stray = [e["key"] for e in entries if e["key"] not in known]
        if stray:
            print(f"entries missing from SECTIONS: {stray}")
            return 1
        out = THESIS / "10-references.md"
        out.write_text(render_markdown(entries, SECTIONS), encoding="utf-8")
        print(f"wrote {out} ({len(entries)} entries)")
        return 0

    if not BIB.exists():
        print(f"missing {BIB}")
        return 1

    paths = sorted(THESIS.glob("*.md"))
    if args.all:
        paths += ALSO_SCAN

    bib = BIB.read_text(encoding="utf-8")
    have = set(_EPRINT.findall(bib))
    keys = _ENTRY.findall(bib)
    used = cited(paths)

    missing = sorted(set(used) - have)
    unused = sorted(have - set(used))

    print(f"{len(keys)} bibliography entries, {len(have)} with an arXiv id")
    print(f"{len(used)} distinct arXiv ids cited across {len(paths)} files")
    print()

    if missing:
        print("CITED BUT NOT IN THE BIBLIOGRAPHY:")
        for a in missing:
            print(f"  arXiv:{a}   cited in {', '.join(sorted(used[a]))}")
        print()

    if unused:
        # Not an error: the bibliography deliberately carries background works
        # the chapters discuss without an inline arXiv link.
        print("In the bibliography but not cited by arXiv id (fine if named in prose):")
        for a in unused:
            print(f"  arXiv:{a}")
        print()

    unverified = [k for k, body in
                  ((k, b) for k, b in zip(keys, re.split(r"@\w+\{", bib)[1:], strict=True))
                  if "UNVERIFIED" in body]
    if unverified:
        print(f"{len(unverified)} entries need their metadata checked before submission:")
        for k in unverified:
            print(f"  {k}")
        print()

    if missing:
        print("FAIL: a citation has no bibliography entry.")
        return 1
    print("OK: every cited arXiv id has an entry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
