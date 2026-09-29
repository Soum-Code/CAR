"""The front-matter gate, and the abstract it stopped duplicating.

Two things worth pinning.

The title page used to take its values from argparse defaults, so a build that
forgot the flags produced a PDF reading "university name" in italics, on the
one page an examiner reads first, with nothing saying so. It now reads
`docs/thesis/metadata.yaml` and refuses to build while any value there still
carries a `FILL:` marker. That gate is only useful if it actually blocks, so
this asserts it does -- and that `--allow-placeholders` still lets a draft
through, because otherwise nobody can build until submission day.

The second is a bug this file was written alongside. `build_latex.py` folded
the whole of `00-front-matter.md` into the Abstract chapter, header and all, so
the PDF printed the subtitle, the author block and the word "Abstract" twice
inside the abstract. The fix slices from `## Abstract` to `## Contents`, and
that slice has to keep the two substantive sections that sit below it.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_latex  # noqa: E402

META = ROOT / "docs" / "thesis" / "metadata.yaml"
FRONT_MD = ROOT / "docs" / "thesis" / "00-front-matter.md"


def test_metadata_file_exists():
    assert META.exists(), "the title page has no values without it"


def test_gate_blocks_on_unfilled_values(monkeypatch, tmp_path):
    """A FILL: marker must stop the build, not default quietly."""
    stub = tmp_path / "metadata.yaml"
    stub.write_text(
        "title: T\nauthor: A\ninstitution: 'FILL: your university'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_latex, "META", stub)
    with pytest.raises(SystemExit):
        build_latex.load_metadata(allow_placeholders=False)


def test_allow_placeholders_lets_a_draft_through(monkeypatch, tmp_path):
    stub = tmp_path / "metadata.yaml"
    stub.write_text(
        "title: T\nauthor: A\ninstitution: 'FILL: your university'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(build_latex, "META", stub)
    meta = build_latex.load_metadata(allow_placeholders=True)
    assert meta["title"] == "T"


def test_filled_metadata_passes(monkeypatch, tmp_path):
    stub = tmp_path / "metadata.yaml"
    stub.write_text("title: T\nauthor: A\ninstitution: Real University\n",
                    encoding="utf-8")
    monkeypatch.setattr(build_latex, "META", stub)
    meta = build_latex.load_metadata(allow_placeholders=False)
    assert meta["institution"] == "Real University"


def test_co_supervisor_omitted_when_blank():
    """An empty co_supervisor must not leave a dangling rule and blank name."""
    one = build_latex.supervisor_block(
        {"supervisor": "Dr A", "supervisor_designation": "Professor",
         "co_supervisor": "", "co_supervisor_designation": ""})
    assert "Dr A" in one and one.count(r"\bfseries") == 1

    two = build_latex.supervisor_block(
        {"supervisor": "Dr A", "supervisor_designation": "Professor",
         "co_supervisor": "Dr B", "co_supervisor_designation": "Reader"})
    assert "Dr B" in two and two.count(r"\bfseries") == 2


def test_front_matter_has_the_headings_the_build_slices_on():
    """build_latex.py cuts between these two; renaming one breaks the build."""
    lines = [ln.strip().lower() for ln in
             FRONT_MD.read_text(encoding="utf-8").split("\n")]
    assert "## abstract" in lines
    assert "## contents" in lines
    assert lines.index("## abstract") < lines.index("## contents")


def test_code_spans_go_through_the_unicode_gate():
    """A code span must not smuggle raw non-ASCII into \\texttt.

    Spans are stashed before esc() runs, so for a while they bypassed the
    unmapped-character check entirely: the draft's `47 x 3 = 131` (with a real
    U+00D7) reached main.tex as a raw byte inside \\texttt. That is precisely
    the silent pass-through the gate exists to stop, and it only surfaced when
    a compiler was finally pointed at the output.
    """
    out = build_latex.inline("a `47 × 3 = 131` span")
    assert "×" not in out
    assert r"$\times$" in out


def test_unmapped_character_in_a_code_span_still_raises():
    with pytest.raises(SystemExit):
        build_latex.inline("a `☃ snowman` span")


def test_code_blocks_gate_on_unrenderable_unicode():
    """Fenced blocks are emitted verbatim, so they need their own gate.

    Third instance of one bug: prose was gated, inline spans were not, and
    fenced blocks were not either. A real U+03BC inside a formula block killed
    the compile with "Invalid UTF-8 byte sequence" pointing at \\lst@EC, which
    names neither the file nor the character.
    """
    ok = "text\n\n```\n(μ − α) / (1 − α)\n```\n"
    build_latex.check_listing_chars(ok, "ok.md")          # in the literate map

    with pytest.raises(SystemExit):
        build_latex.check_listing_chars("```\n☃\n```\n", "bad.md")


def test_footnoted_url_survives_escaping():
    """A link's \\footnote must not be escaped by the esc() that follows it.

    inline() substitutes links, then calls esc() on the whole string. The
    \\footnote{\\url{...}} the substitution had just inserted was therefore
    escaped into \\textbackslash{}footnote\\{...\\} and printed as visible text.
    Every footnoted URL in the thesis rendered that way, silently, until the
    document was compiled for the first time.
    """
    out = build_latex.inline("see [arXiv:2310.01798](https://arxiv.org/abs/2310.01798) for")
    assert r"\footnote{\url{https://arxiv.org/abs/2310.01798}}" in out
    assert "textbackslash" not in out


def test_internal_link_is_escaped_once_not_twice():
    out = build_latex.inline("see [FINDINGS_PROBE](../FINDINGS_PROBE.md) here")
    assert r"\_" in out
    assert r"\textbackslash" not in out


def test_emphasis_spanning_a_line_break_still_converts():
    """Paragraphs are one unit; inline() used to run per line and split spans."""
    md = "**Sherlock** (Ro et al., *Reliable and Efficient Agentic Workflow\nExecution*) is"
    out = build_latex.convert(md, chapter_title="T")
    assert r"\emph{Reliable and Efficient Agentic Workflow Execution}" in out
    assert "*" not in out.replace(r"\*", "")


def test_listings_literate_covers_every_declared_char():
    """LST_LITERATE and the \\lstset literate= list must not drift apart."""
    for ch in build_latex.LST_LITERATE:
        assert ch in build_latex.LST, f"{ch!r} allowed but not in \\lstset"


def test_substantive_sections_survive_the_slice():
    """The measured-vs-modelled declaration and reproducibility must remain.

    They sit between Abstract and Contents, so the slice keeps them. If either
    moved below Contents it would silently vanish from the PDF.
    """
    text = FRONT_MD.read_text(encoding="utf-8")
    lines = text.split("\n")
    start = next(i for i, ln in enumerate(lines)
                 if ln.strip().lower() == "## abstract")
    end = next(i for i, ln in enumerate(lines)
               if i > start and ln.strip().lower() == "## contents")
    kept = "\n".join(lines[start:end]).lower()
    assert "what is measured and what is modelled" in kept
    assert "reproducibility" in kept
