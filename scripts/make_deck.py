"""Final Review defence deck, built from the thesis. [CPU]

Structured against the Final Review checklist in MTech_ProjectGuidelines_2026.pdf
rather than as a free-form talk: the panel marks fifteen specific items, and a
deck that answers them in order is easier to score than one that does not.

Numbers come from the thesis chapters. Nothing here is rounded differently from
the chapter it came from, and nothing is stated at greater strength.

    python scripts/make_deck.py
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

OUT = Path("docs/CAR-final-review.pptx")

# Content-informed: terracotta marks refuted claims, teal marks measured ones.
# The thesis is largely a record of things that did not work, so the two carry
# meaning rather than decoration.
SLATE = RGBColor(0x2B, 0x3A, 0x42)
PAPER = RGBColor(0xF4, 0xF4, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TERRA = RGBColor(0xB8, 0x50, 0x42)
TEAL = RGBColor(0x3E, 0x6E, 0x6C)
MUTED = RGBColor(0x6E, 0x7B, 0x82)
INK = RGBColor(0x1A, 0x22, 0x28)

HEAD = "Cambria"
BODY = "Calibri"

W, H = Inches(13.333), Inches(7.5)


def deck() -> Presentation:
    p = Presentation()
    p.slide_width, p.slide_height = W, H
    return p


def blank(prs, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = SLATE if dark else PAPER
    return s


def text(s, txt, x, y, w, h, size=16, bold=False, color=INK, font=BODY,
         align=PP_ALIGN.LEFT, space_after=6, italic=False):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    lines = txt.split("\n") if isinstance(txt, str) else txt
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        para.space_after = Pt(space_after)
        r = para.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return tb


def bullets(s, items, x, y, w, h, size=15, color=INK, gap=10):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    for i, it in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(gap)
        if isinstance(it, tuple):
            lead, rest = it
            r = para.add_run(); r.text = lead
            r.font.size = Pt(size); r.font.bold = True
            r.font.color.rgb = color; r.font.name = BODY
            r2 = para.add_run(); r2.text = rest
            r2.font.size = Pt(size); r2.font.color.rgb = color; r2.font.name = BODY
        else:
            r = para.add_run(); r.text = it
            r.font.size = Pt(size); r.font.color.rgb = color; r.font.name = BODY
    return tb


def title_of(s, txt, sub=None):
    """Title, with the subtitle pushed down when the title wraps.

    A 32pt title wrapping to two lines is 1.17in tall in a 0.9in box, so a
    fixed subtitle position at 1.32in collided with it. Measure instead: shrink
    slightly past one line, and move the subtitle to clear whatever results.
    """
    size = 32
    avail = 12.0
    # ~0.50 of point size per character for this font at mixed case.
    per_line = max(1, int(avail / (size * 0.50 / 72)))
    if len(txt) > per_line:
        size = 27
        per_line = max(1, int(avail / (size * 0.50 / 72)))
    lines = max(1, -(-len(txt) // per_line))
    h = lines * size * 1.22 / 72
    text(s, txt, 0.7, 0.45, avail, h, size=size, bold=True, color=SLATE,
         font=HEAD, space_after=0)
    if sub:
        text(s, sub, 0.7, 0.45 + h + 0.10, avail, 0.55, size=14, color=MUTED,
             italic=True)


def table(s, rows, x, y, w, col_w=None, size=12, head_size=12, h_row=0.34,
          emph_col=None, emph_rows=()):
    nr, nc = len(rows), len(rows[0])
    shape = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w),
                               Inches(h_row * nr))
    t = shape.table
    if col_w:
        total = sum(col_w)
        for i, cw in enumerate(col_w):
            t.columns[i].width = Inches(w * cw / total)
    for ri, row in enumerate(rows):
        t.rows[ri].height = Inches(h_row)
        for ci, val in enumerate(row):
            cell = t.cell(ri, ci)
            cell.text = str(val)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.08)
            cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = 0
            f = cell.fill
            f.solid()
            if ri == 0:
                f.fore_color.rgb = SLATE
            elif ri in emph_rows:
                f.fore_color.rgb = RGBColor(0xE8, 0xE4, 0xE0)
            else:
                f.fore_color.rgb = WHITE if ri % 2 else RGBColor(0xFA, 0xFA, 0xF8)
            for para in cell.text_frame.paragraphs:
                para.alignment = PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER
                for r in para.runs:
                    r.font.size = Pt(head_size if ri == 0 else size)
                    r.font.name = BODY
                    r.font.bold = ri == 0 or ri in emph_rows
                    if ri == 0:
                        r.font.color.rgb = WHITE
                    elif emph_col is not None and ci == emph_col:
                        r.font.color.rgb = TERRA
                    else:
                        r.font.color.rgb = INK
    return shape


def chip(s, label, x, y, w=2.2, h=0.42, fill=TEAL, size=12):
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.margin_top = tf.margin_bottom = 0
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    r = p0.add_run(); r.text = label
    r.font.size = Pt(size); r.font.bold = True
    r.font.color.rgb = WHITE; r.font.name = BODY
    box.fill.solid(); box.fill.fore_color.rgb = fill
    box.line.fill.background()
    return box


def stat(s, value, label, x, y, w=3.0, vsize=44, color=SLATE):
    # Value box height matches where the label starts, so the two abut rather
    # than overlapping by a tenth of an inch.
    vh = 0.72
    text(s, value, x, y, w, vh, size=vsize, bold=True, color=color, font=HEAD)
    text(s, label, x, y + vh, w, 0.7, size=12, color=MUTED)


def note(s, txt):
    s.notes_slide.notes_text_frame.text = txt


def build() -> None:
    prs = deck()

    # ---------------------------------------------------------------- 1 title
    s = blank(prs, dark=True)
    text(s, "What Step-Level Verification Certifies,\nand What It Misses",
         0.9, 2.0, 11.5, 1.8, size=40, bold=True, color=WHITE, font=HEAD,
         space_after=2)
    text(s, "A measurement study of selective verification in multi-step LLM reasoning",
         0.9, 3.75, 11.5, 0.5, size=17, color=RGBColor(0xC8, 0xD2, 0xD6), italic=True)
    text(s, "P. Somnath Reddy   ·   25167023", 0.9, 4.85, 6.0, 0.4, size=15,
         color=WHITE)
    text(s, "M.Tech, Computer Engineering\nSchool of Computer Engineering, KIIT",
         0.9, 5.35, 6.0, 0.8, size=13, color=RGBColor(0xA8, 0xB6, 0xBC))
    text(s, "Guide: Ponsuresh Manoharan\nSubject Matter Expert — Cyber Security, L&T EduTech",
         7.6, 5.35, 5.0, 0.8, size=13, color=RGBColor(0xA8, 0xB6, 0xBC))
    chip(s, "FINAL REVIEW", 0.9, 1.25, w=2.1, h=0.4, fill=TERRA)
    note(s, "Final Review. The deck follows the fifteen checklist items in the "
            "guidelines, in order. I will flag the three that are not yet "
            "complete rather than leave the panel to find them.")

    # ---------------------------------------------------------------- 2 problem
    s = blank(prs)
    title_of(s, "The problem", "Checklist: problem statement, motivation")
    text(s, "A language model answering a multi-step question writes a chain of "
            "intermediate claims. A wrong claim early is not a local defect — "
            "everything after it is generated conditioned on it.",
         0.7, 1.95, 11.9, 1.0, size=18)
    text(s, "Where should a limited verification budget go?", 0.7, 3.0, 11.9, 0.5,
         size=20, bold=True, color=TERRA, font=HEAD)
    text(s, "The obvious design, and the one this project set out to build:",
         0.7, 3.85, 11.9, 0.4, size=15, color=MUTED)
    for i, (n, lab, desc) in enumerate([
            ("1", "A score", "token entropy, surprisal,\nsampling disagreement"),
            ("2", "A calibration", "conformal prediction turns it\ninto a threshold with a guarantee"),
            ("3", "A gate", "verify the steps above it,\nsubject to a budget")]):
        x = 0.7 + i * 4.1
        chip(s, f"{n}.  {lab}", x, 4.4, w=3.7, h=0.44, fill=SLATE)
        text(s, desc, x, 5.0, 3.7, 0.9, size=13, color=INK)
    text(s, "Each part has substantial literature behind it. Assembled, it does not work. "
            "This thesis measures why.",
         0.7, 6.25, 11.9, 0.6, size=16, bold=True, color=SLATE)
    note(s, "The motivation is the propagation property: one bad step poisons "
            "every step after it, so where you spend the budget matters. Every "
            "component of the obvious design is published and works in isolation.")

    # ---------------------------------------------------------------- 3 pivot
    s = blank(prs)
    title_of(s, "What was proposed, and what happened to it",
             "Checklist: scope changes flagged and justified · contribution distinguishable")
    table(s, [
        ["Original claim", "Outcome"],
        ["Adaptive conformal under censored feedback is novel", "SCOOPED — CSA Thm E.1, stronger guarantee"],
        ["Composite token-level uncertainty is the key signal", "REFUTED — AUROC 0.5589"],
        ["Semantic entropy at step level is the key signal", "REFUTED — 0.5740; combining buys 0.0002"],
        ["Influence-weighted allocation beats uniform", "REFUTED — loses on 5 DAG families, 2 corpora"],
        ["\"Verify early beats verify late\"", "REFUTED — worst shape at every scope > 0"],
        ["StrategyQA as the primary benchmark", "WRONG CHOICE — 72.9% of graphs one hop deep"],
        ["α = 0.10 as a working risk target", "INFEASIBLE — 32.3% entry fee before any method"],
    ], 0.7, 2.0, 11.9, col_w=[5, 6], size=12.5, h_row=0.42, emph_col=1)
    text(s, "The project began as a method paper. Two rounds of literature checking and "
            "stress-testing removed every method claim. What survived is a set of measurements — "
            "and they compose into a stronger claim about the design space than the system would have been.",
         0.7, 5.75, 11.9, 1.0, size=14.5, color=SLATE)
    note(s, "This is the honest pivot. The scoop was found by us, in week one, and "
            "is cited rather than hidden — a reviewer who finds it independently "
            "discounts everything else. Every refutation here has a regression test.")

    # ---------------------------------------------------------------- 4 the distinction
    s = blank(prs)
    title_of(s, "The distinction everything rests on",
             "Checklist: methodology · contribution to knowledge")
    text(s, "global_correct(t)  =  local_valid(t)  AND  NOT premise_corrupt(t)",
         0.7, 2.05, 11.9, 0.6, size=22, bold=True, color=SLATE, font="Courier New")
    for i, (h, d, c) in enumerate([
            ("Local invalidity",
             "The step does not follow from its own premises.\n"
             "47 × 3 = 131 is wrong in any context.\nA calculator catches it every time.", TEAL),
            ("Inherited corruption",
             "The step follows perfectly, and a premise is false.\n"
             "Impeccable reasoning to a false conclusion.\n"
             "No amount of checking the step reveals anything.", TERRA)]):
        x = 0.7 + i * 6.1
        chip(s, h, x, 3.0, w=5.6, h=0.44, fill=c)
        text(s, d, x, 3.62, 5.6, 1.4, size=14)
    text(s, "A verifier reports the first. Conformal machinery calibrates whatever the "
            "verifier reports — so it calibrates local validity.",
         0.7, 5.3, 11.9, 0.7, size=16, bold=True, color=SLATE)
    text(s, "The question this thesis asks first: how much of the failure does that leave out?",
         0.7, 6.15, 11.9, 0.5, size=16, italic=True, color=TERRA)
    note(s, "This is the whole thesis in one line. Both halves are separately "
            "observable on GSM8K: the inline calculator annotations give local "
            "validity deterministically, and Math-Shepherd's Monte-Carlo labels "
            "give global correctness. Having both on the same steps is what makes "
            "the measurement possible at all.")

    # ---------------------------------------------------------------- 5 data
    s = blank(prs)
    title_of(s, "Datasets and corpora",
             "Checklist: dataset finalized — source, size, split strategy")
    table(s, [
        ["Corpus", "Generator", "GSM8K acc", "Solutions", "Steps", "Labels"],
        ["Math-Shepherd", "Mistral-7B-SFT", "~45%", "25,971", "93,129", "published +/−"],
        ["Generated here", "Qwen2.5-7B-Instruct", "80.0%", "500", "2,573", "K=4 rollouts"],
        ["Generated here", "Llama 3.1 8B Instruct", "68.4%", "500", "1,938", "K=4 rollouts"],
    ], 0.7, 2.0, 11.9, col_w=[3, 4, 2, 2, 2, 3], size=13, h_row=0.42)
    text(s, "Splits by SHA-256 of example id, not by shuffled index — an example cannot migrate "
            "between splits as the corpus grows. The composite scaler is fitted on dev only; "
            "fitting it on calibration or test would void the conformal guarantee.",
         0.7, 4.0, 11.9, 0.9, size=14)
    for i, (v, l) in enumerate([("175", "dev"), ("143", "calibration"), ("182", "test questions"),
                                ("925", "test steps")]):
        stat(s, v, l, 0.7 + i * 3.0, 5.0, w=2.8, vsize=34)
    text(s, "Both generated corpora were written in Math-Shepherd's own format, so one analysis "
            "path reads all three. The comparison is between corpora, not between implementations.",
         0.7, 6.45, 11.9, 0.6, size=13, italic=True, color=MUTED)
    note(s, "The third generator, Llama, was licence-gated on Kaggle until late "
            "and was added on 28 September. Settings were fixed by the Qwen run: "
            "500 problems, K=4, temperature 0.7, 4-shot.")

    # ---------------------------------------------------------------- 6 result 1
    s = blank(prs)
    title_of(s, "Result 1 — the certified quantity is not the quantity of interest",
             "Within wrong-answer solutions, one definition applied to all three corpora")
    table(s, [
        ["", "Mistral-7B-SFT", "Llama 3.1 8B", "Qwen2.5-7B"],
        ["GSM8K accuracy", "~45%", "68.4%", "80.0%"],
        ["local error", "0.1708", "0.1034", "0.0813"],
        ["global error", "0.7106", "0.7442", "0.5772"],
        ["C1 (checkable)", "0.7848", "0.8738", "0.9040"],
        ["n globally-wrong checkable steps", "46,555", "309", "125"],
        ["95% Wilson CI", "[0.781, 0.789]", "[0.832, 0.906]", "[0.840, 0.944]"],
    ], 0.7, 2.0, 11.9, col_w=[5, 3, 3, 3], size=13, h_row=0.42, emph_rows={4})
    text(s, "78.5% of globally-wrong steps are arithmetically perfect — wrong only because a premise was.",
         0.7, 5.15, 11.9, 0.5, size=17, bold=True, color=TERRA)
    text(s, "C1 is monotone in generator accuracy. Mistral's interval is disjoint from both others; "
            "Llama's and Qwen's overlap each other, so the ordering between those two is not resolved here.",
         0.7, 5.75, 11.9, 0.7, size=14)
    text(s, "Controlling local selective risk at level α bounds nothing about the answer.",
         0.7, 6.5, 11.9, 0.5, size=15, bold=True, color=SLATE)
    note(s, "The mechanism: the stronger model halves its arithmetic slips without "
            "halving its inherited corruption, so a larger share of what remains is "
            "the kind no calculator can see. Global error is NOT monotone — Llama's "
            "is highest — so C1 rises because the denominator shrinks, not because "
            "the numerator grows. A deterministic verifier gets LESS useful as "
            "generators improve.")

    # ---------------------------------------------------------------- 7 reach
    s = blank(prs)
    title_of(s, "Result 2 — verifier reach is the controlling variable, and it is semantic",
             "Measured on the population deterministic checking provably cannot see")
    table(s, [
        ["Verifier", "Independent?", "Task-trained?", "Scope", "False alarm"],
        ["arithmetic, step-local", "yes", "—", "0.0000", "—"],
        ["arithmetic, unbounded lookback", "yes", "—", "0.1999", "—"],
        ["same-model critic (the generator)", "NO", "no", "0.0000", "0.0000"],
        ["independent judge (Qwen2.5-7B)", "yes", "no", "0.2283", "0.0200"],
        ["task PRM (Math-Shepherd-7B)", "yes", "YES", "0.9033", "0.0987"],
    ], 0.7, 2.0, 11.9, col_w=[5, 2.5, 2.5, 2, 2.5], size=13, h_row=0.42,
          emph_rows={5})
    text(s, "Reach is not how far back you look. It requires independence from the generator "
            "AND task specialisation — neither alone suffices.",
         0.7, 4.85, 11.9, 0.7, size=16, bold=True, color=SLATE)
    text(s, "Widening an arithmetic window saturates at 0.1999 because 80.0% of inherited "
            "corruption has no upstream arithmetic error at all. The same-model critic detects "
            "zero errors — it is the model that wrote these solutions, asked whether they are sound.",
         0.7, 5.65, 11.9, 0.9, size=14)
    text(s, "The number that matters most later is not the 0.9033. It is the 0.0987 next to it.",
         0.7, 6.6, 11.9, 0.5, size=15, bold=True, color=TERRA)
    note(s, "Sample sizes differ by arm and this matters: the PRM was run on 1,500 "
            "arithmetic-blind steps plus 750 controls; both judge arms on 600 plus "
            "300. So the judges' scopes rest on 600 positives, not the PRM's larger "
            "sample. Three tokenizer faults produced the opposite result before a "
            "validation gate caught them — see the methodology slide.")

    # ---------------------------------------------------------------- 8 allocation
    s = blank(prs)
    title_of(s, "Result 3 — \"verify early\" is false",
             "Final-answer error by allocation policy, GSM8K graphs broken out by depth")
    table(s, [
        ["Depth", "Graphs", "uniform", "front", "back", "influence", "depth", "cut", "spread"],
        ["2", "2,988", "0.2049", "0.2049", "0.2042", "0.2033", "0.2048", "0.2016", "0.0033"],
        ["3", "2,431", "0.2632", "0.2777", "0.2382", "0.2744", "0.2290", "0.2423", "0.0488"],
        ["4", "1,010", "0.2992", "0.3269", "0.2577", "0.3241", "0.2455", "0.2746", "0.0814"],
        ["5", "308", "0.3221", "0.3618", "0.2632", "0.3600", "0.2533", "0.2983", "0.1084"],
        ["6", "52", "0.3578", "0.4059", "0.2872", "0.4013", "0.2779", "0.3373", "0.1279"],
    ], 0.7, 2.0, 11.9, col_w=[1.6, 2, 2, 2, 2, 2.4, 2, 2, 2], size=12, h_row=0.40)
    text(s, "front and influence are the two worst policies on both benchmarks. The spread grows "
            "monotonically with depth — 0.0033 to 0.1279 — and they diverge from the field in exactly "
            "the regime the thesis is about. At depth 6, front-loading costs 12.8 points.",
         0.7, 4.75, 11.9, 0.9, size=14)
    text(s, "The last defence — that early steps are intrinsically harder — is closed by measurement: "
            "corr(position, local error) = +0.950, error doubling from 11% at step 1 to 22% at step 8.",
         0.7, 5.7, 11.9, 0.8, size=14)
    text(s, "The useful structural signal is ancestor count, not descendant count. "
            "That is the opposite of the proposed rule.",
         0.7, 6.5, 11.9, 0.5, size=15, bold=True, color=TERRA)
    note(s, "Five separate tests, two of which were confounded and had to be "
            "discarded. The first ran on linear chains, where descendant count is "
            "exactly T−t — so the influence schedule IS the front schedule, "
            "correlation −1.000. The test could not distinguish the two "
            "hypotheses. Catching that is the single most useful methodological "
            "correction in the project.")

    # ---------------------------------------------------------------- 9 the gate
    s = blank(prs)
    title_of(s, "Result 4 — the assembled gate does not control risk",
             "500 Qwen solutions, 2,573 steps, split 175 dev / 143 calibration / 182 test")
    text(s, "The signal does not rank the risk", 0.7, 1.95, 5.6, 0.4, size=16,
         bold=True, color=SLATE, font=HEAD)
    table(s, [
        ["Feature set", "AUROC (test)"],
        ["token-level only", "0.5589"],
        ["semantic divergence only", "0.5740"],
        ["both", "0.5742"],
        ["synthetic control (1 s.d.)", "0.8668"],
        ["pure noise control", "0.4828"],
    ], 0.7, 2.45, 5.6, col_w=[4, 2], size=12.5, h_row=0.38)
    text(s, "Coverage holds; risk does not", 6.9, 1.95, 5.7, 0.4, size=16,
         bold=True, color=SLATE, font=HEAD)
    table(s, [
        ["α", "binds?", "risk", "verify %"],
        ["0.05", "yes", "0.1491", "4.6%"],
        ["0.10", "yes", "0.1494", "9.2%"],
        ["0.15", "yes", "0.1516", "14.1%"],
        ["0.20", "no", "0.1502", "17.0%"],
        ["0.30", "no", "0.1538", "21.8%"],
    ], 6.9, 2.45, 5.7, col_w=[2, 2, 2.5, 2.5], size=12.5, h_row=0.38,
          emph_rows={1})
    text(s, "Combining the two signals buys 0.0002. At α = 0.05 the measured selective risk is "
            "three times the target, and it barely moves across the whole sweep while verification "
            "climbs from 4.6% to 21.8%.",
         0.7, 5.0, 11.9, 0.8, size=14)
    text(s, "A conformal guarantee is a statement about the acceptance rule, not about the risk of "
            "what it accepts. With an uninformative score the two come apart completely — and nothing "
            "in the procedure reports that.",
         0.7, 5.85, 11.9, 0.9, size=15, bold=True, color=TERRA)
    note(s, "Coverage is guaranteed by construction, which is precisely the point: "
            "the guarantee holds, is reported as holding, and tells you nothing "
            "about the quantity you care about. The synthetic and noise controls "
            "are what license reading 0.5589 as a property of the signal rather "
            "than of the harness.")

    # ---------------------------------------------------------------- 10 oracle
    s = blank(prs)
    title_of(s, "Which component is at fault? The oracle answers it",
             "Checklist: ablation studies — same calibrator, budget and verifier; only the score changes")
    table(s, [
        ["Score", "calls/q", "recall", "selective risk", "PROJ accuracy"],
        ["no gate", "0.00", "—", "0.1554", "0.8022"],
        ["real, split conformal", "1.09", "0.1761", "0.1538", "0.7912"],
        ["real, always verify", "1.89", "0.2606", "—", "0.7637"],
        ["probe (AUROC 0.6968)", "0.96", "0.2465", "0.1394", "0.7802"],
        ["ORACLE", "0.37", "0.4366", "0.0885", "0.9780"],
    ], 0.7, 2.05, 11.9, col_w=[4, 2, 2, 3, 3], size=13, h_row=0.42, emph_rows={5})
    text(s, "The same verifier, at the same measured 9.87% false-alarm rate, moves projected accuracy "
            "from 0.7637 to 0.9780 — 17.6 points above the no-gate baseline — on a fifth of the calls.",
         0.7, 4.7, 11.9, 0.8, size=15, bold=True, color=SLATE)
    text(s, "The verifier was never the defect. It is a good verifier being aimed badly.",
         0.7, 5.5, 11.9, 0.45, size=17, bold=True, color=TEAL)
    text(s, "But the oracle still misses α = 0.05 by 1.8×. At two calls per question over a mean of "
            "5.15 steps, most wrong steps go unverified however perfectly they are ranked. That residual "
            "is the budget, not the score.",
         0.7, 6.05, 11.9, 0.9, size=14)
    note(s, "This is the single most important experiment in the thesis. It "
            "collapses what an earlier draft treated as three independent failure "
            "points into two. The figure of merit is not scope minus false alarm — "
            "it is conditioned on what the gate selects, and the score sets that "
            "conditioning. All accuracy figures here are PROJECTED under a stated "
            "repair model; they cannot be measured on a fixed corpus.")

    # ---------------------------------------------------------------- 11 sweep
    s = blank(prs)
    title_of(s, "How good does the score need to be?",
             "Score quality as a dial: AUROC = Φ(d/√2), everything else held fixed. 13 points × 64 seeds")
    table(s, [
        ["AUROC", "verify %", "sel. risk", "1st-bad recall", "PROJ acc", "95% CI", ""],
        ["0.550", "22.5%", "0.1536", "0.2853", "0.7782", "[0.7725, 0.7840]", "net −"],
        ["0.625", "23.1%", "0.1463", "0.3786", "0.7940", "[0.7880, 0.8001]", "net −"],
        ["0.650", "23.3%", "0.1441", "0.4111", "0.8010", "[0.7953, 0.8067]", "~same"],
        ["0.700", "23.6%", "0.1405", "0.4635", "0.8119", "[0.8057, 0.8181]", "net +"],
        ["0.800", "24.1%", "0.1335", "0.5753", "0.8341", "[0.8282, 0.8400]", "net +"],
        ["0.990", "24.6%", "0.1258", "0.7736", "0.8712", "[0.8650, 0.8775]", "net +"],
    ], 0.7, 2.05, 11.9, col_w=[2, 2, 2, 2.8, 2.2, 3.4, 1.8], size=12, h_row=0.38,
          emph_rows={4})
    text(s, "The crossing is at AUROC ≈ 0.65 — below the probe that already exists. "
            "0.6968 sits inside the [0.625, 0.700] band the sweep cannot resolve, so the probe is at "
            "the boundary, not past it.",
         0.7, 5.0, 11.9, 0.8, size=14.5)
    text(s, "The threshold is made entirely of false alarms.", 0.7, 5.85, 5.8, 0.4,
         size=16, bold=True, color=TERRA)
    text(s, "At FA = 0 the same verifier is worth having at every score quality tested, down to 0.55. "
            "Halving a verifier's false-alarm rate lowers the score quality you need more than raising "
            "its detection rate would — though detection was never varied, so that comparison is an "
            "argument, not a result.",
         0.7, 6.25, 11.9, 0.9, size=13.5)
    note(s, "And no score quality holds a binding alpha: the best any row manages "
            "at alpha = 0.05 is 0.0956, at AUROC 0.99 — still 1.9 times the "
            "target. That rules out reading the oracle's failure as an artifact of "
            "its degenerate binary score distribution. The budget is confirmed as "
            "the separate bottleneck across the whole range.")

    # ---------------------------------------------------------------- 12 auroc
    s = blank(prs)
    title_of(s, "AUROC is the wrong figure of merit for a step score",
             "Verification pinned at 19.1% so only the ranking differs")
    table(s, [
        ["", "AUROC", "calls/q", "recall", "1st-bad recall", "PROJ acc"],
        ["synthetic score", "0.6974", "0.79", "0.2416", "0.3413", "0.8206"],
        ["probe, layer 25", "0.6968", "0.96", "0.2465", "0.3077", "0.7802"],
    ], 0.7, 2.05, 11.9, col_w=[3.5, 2, 2, 2, 3, 2.5], size=13.5, h_row=0.44)
    text(s, "Two scores of near-identical AUROC, differing by 0.04 projected accuracy. "
            "The probe sits at the 0th percentile of 64 synthetic draws at the same AUROC.",
         0.7, 3.65, 11.9, 0.7, size=15)
    text(s, "corr(step position, probe score)      +0.1813\n"
            "corr(step position, composite score)  −0.2766",
         0.7, 4.45, 6.0, 0.9, size=15, font="Courier New", color=SLATE)
    text(s, "The probe flags late steps. Local error rises with position, so a score that chases "
            "positional difficulty is rewarded on AUROC — but the projection only pays for the FIRST "
            "bad step, because everything downstream inherits corruption that repairing a later step "
            "does not undo.",
         0.7, 5.5, 11.9, 0.9, size=14)
    text(s, "Rank by AUROC to compare with the literature. Select on first-bad-step recall.",
         0.7, 6.45, 11.9, 0.5, size=16, bold=True, color=TERRA)
    note(s, "This refutes the metric the chapter itself is built on, which is why "
            "it is reported. The honest limit: only 40 of 182 test trajectories "
            "contain a globally-wrong step, so first-bad recall has a denominator "
            "near 39. The claim rests on the monotone trend across 13 grid points "
            "times 64 seeds, not on this single pair of rows.")

    # ---------------------------------------------------------------- 13 rigor
    s = blank(prs)
    title_of(s, "Statistical rigor", "Checklist: multiple runs/seeds, confidence intervals, significance")
    for i, (v, l, c) in enumerate([
            ("64", "seeds per grid point\nin the score-quality sweep", SLATE),
            ("13 × 64", "grid points × seeds\n832 gate runs", SLATE),
            ("Wilson", "intervals on every\nproportion reported", TEAL),
            ("clustered", "bootstrap — steps within\na solution are not independent", TEAL)]):
        stat(s, v, l, 0.7 + i * 3.1, 2.0, w=2.9, vsize=34, color=c)
    text(s, "What the intervals changed", 0.7, 3.7, 11.9, 0.4, size=17, bold=True,
         color=SLATE, font=HEAD)
    table(s, [
        ["Effect", "Estimate", "95% CI", "Verdict"],
        ["Probe: doubling training data", "+0.0105", "[−0.035, +0.057]", "spans zero"],
        ["Probe: non-linear head at matched layer", "+0.0051", "[−0.031, +0.042]", "spans zero"],
        ["Entailment vs numeric equivalence", "−0.0114", "[−0.087, +0.059]", "not a ranking"],
        ["First-bad target vs pooled probe", "+0.2216", "[+0.051, +0.390]", "excludes zero"],
        ["First-bad target vs published probe", "+0.1412", "[−0.065, +0.333]", "spans zero"],
    ], 0.7, 4.2, 11.9, col_w=[5, 2.2, 3, 2.5], size=12.5, h_row=0.40, emph_col=3)
    text(s, "Every one of these was reported as an effect in an earlier draft. The intervals are why "
            "they are not reported that way now.",
         0.7, 6.72, 11.9, 0.6, size=13.5, italic=True, color=MUTED)
    note(s, "The last two rows are the same effect against two reasonable "
            "comparators, and they disagree on significance. The thesis reports "
            "both rather than the flattering one. That is the discipline the panel "
            "should probe: ask me for any number and I can name its interval and "
            "the script that produced it.")

    # ---------------------------------------------------------------- 14 method discipline
    s = blank(prs)
    title_of(s, "Measurement discipline — three faults that each produced a confident wrong answer",
             "Checklist: error analysis and failure cases, not just success metrics")
    table(s, [
        ["Fault", "Symptom", "Cause"],
        ["Wrong output logits",
         "PRM flagged 94.9% of controls — steps carrying its own training labels",
         "tok.encode gave ▁+ = 648 locally, + = 28806 on Kaggle"],
        ["Wrong input positions",
         "every score came back NaN",
         "the ки step-tag id is an input id; Kaggle encoded it differently"],
        ["Right padding in batched generation",
         "judge result 0.1933 → 0.2283 once fixed",
         "pad tokens inserted between prompt and continuation"],
    ], 0.7, 2.0, 11.9, col_w=[3.5, 4.5, 5], size=12, h_row=0.62)
    text(s, "The first completed run reported the OPPOSITE conclusion — negative net scope — and it was "
            "entirely an artifact.",
         0.7, 4.5, 11.9, 0.5, size=15, bold=True, color=TERRA)
    text(s, "A validation gate now scores 400 steps with known labels before any real measurement and "
            "aborts below 0.15 separation. It caught three successive broken configurations before the "
            "fourth passed at 0.5788.",
         0.7, 5.1, 11.9, 0.8, size=14)
    text(s, "On a borrowed model, the harness must prove it can reproduce that model's known behaviour "
            "before any novel number from it is believed.",
         0.7, 5.95, 11.9, 0.7, size=15.5, bold=True, color=SLATE, font=HEAD)
    text(s, "Roughly 75 numerical errors were found in this project by recomputation. Every one is fixed; "
            "the guards that catch each class are in the repo.",
         0.7, 6.7, 11.9, 0.5, size=13, italic=True, color=MUTED)
    note(s, "Without the validation gate this project would have published a "
            "confident negative result produced entirely by a tokenizer mismatch. "
            "The same-model critic's failure is the one case where a gate failure "
            "is the FINDING rather than a bug — it genuinely cannot separate its "
            "own errors, which is what Huang et al. predict.")

    # ---------------------------------------------------------------- 15 negatives
    s = blank(prs)
    title_of(s, "What was refuted — including this thesis's own claims",
             "Checklist: limitations honestly discussed, no overclaiming")
    table(s, [
        ["Claim", "Source", "Outcome"],
        ["A calibrated gate controls risk", "the whole premise", "REFUTED — misses α by 3×"],
        ["The three failures are independent", "ch. 8 first draft", "PARTLY WRONG — verifier is downstream of the score"],
        ["Corruption is near-absorbing", "this thesis, ch. 5", "GENERATOR-SPECIFIC, and monotone"],
        ["net = scope − FA is the figure of merit", "this thesis, ch. 6", "CORRECTED — base-rate error"],
        ["0.6968 is a floor, pending more data", "this thesis, ch. 8.6", "UNSUPPORTED — curve was on the selection split"],
        ["A non-linear probe is the right instrument", "ch. 8.6 and ch. 10", "NULL — +0.0051, sign flips by layer"],
        ["AUROC is the figure of merit", "implicit throughout", "REFUTED — first-bad recall is"],
    ], 0.7, 2.0, 11.9, col_w=[5, 3, 5.5], size=12, h_row=0.42, emph_col=2)
    text(s, "This thesis refuted more claims than it established, including most of its own. "
            "Each refutation that rests on a measurement carries a regression test, so it cannot "
            "quietly stop reproducing.",
         0.7, 5.35, 11.9, 0.8, size=15, bold=True, color=SLATE)
    text(s, "Four labels are used and kept distinct: REFUTED (a measurement contradicts it), "
            "UNSUPPORTED (the evidence was invalid — neither established nor ruled out), "
            "NULL (measured, indistinguishable from zero), and MEASURED.",
         0.7, 6.2, 11.9, 0.8, size=13.5, color=MUTED)
    note(s, "The refuted-versus-unsupported distinction is worth defending if "
            "asked. C9c's supporting learning curve had been scored on the "
            "selection split — the same data the layer and regularisation were "
            "chosen on. A clean test gives +0.0105 with an interval spanning zero. "
            "Invalid evidence is not the same as a false claim, and the thesis "
            "marks the difference.")

    # ---------------------------------------------------------------- 16 contribution
    s = blank(prs)
    title_of(s, "Contribution to knowledge",
             "Checklist: what is new, what is better, by how much")
    items = [
        ("The certified quantity is not the quantity of interest. ",
         "78.5% of globally-wrong steps are arithmetically perfect, on 93,129 labelled steps."),
        ("The gap widens as generators improve. ",
         "0.7848 → 0.8738 → 0.9040 across three models, monotone in accuracy, Mistral's interval "
         "disjoint from both others. A deterministic verifier gets LESS useful as models improve."),
        ("Reach requires independence AND task specialisation. ",
         "0.0000 / 0.2283 / 0.9033 across four verifier classes; neither property alone suffices."),
        ("\"Verify early\" is false, and the useful signal is ancestor count. ",
         "Front-loading is worst at every scope > 0; corr(position, error) = +0.950."),
        ("The failure is attributable, not merely observed. ",
         "An oracle takes risk 0.154 → 0.089 and accuracy 0.79 → 0.98 on a third of the calls, "
         "showing the verifier is downstream of the score. Two bottlenecks survive: signal and budget."),
        ("AUROC is not a sufficient figure of merit. ",
         "Equal-AUROC scores differ by 0.04 projected accuracy, because only the first bad step "
         "can be repaired."),
    ]
    bullets(s, items, 0.7, 2.0, 11.9, 4.2, size=14, gap=13)
    text(s, "A claim about the design space rather than about one system — and falsifiable in the only "
            "way that matters: a signal that ranks step error would break it.",
         0.7, 6.4, 11.9, 0.6, size=15, bold=True, color=TEAL)
    note(s, "Six contributions, each with a measured number and a regression "
            "test. The fifth is the one I would lead with at a conference: a "
            "negative result with a mechanism transfers, where a working gate on "
            "GSM8K would be one more entry in a crowded area.")

    # ---------------------------------------------------------------- 17 limitations
    s = blank(prs)
    title_of(s, "Limitations", "Checklist: limitations honestly discussed — stated before being asked")
    left = [
        ("Sample size. ", "182 test questions, 925 test steps. The headline gap is far "
         "outside sampling noise; between-condition differences are not."),
        ("108 first-bad steps exist in the whole corpus, ", "40 in test. That is the binding "
         "constraint on §8.6's only constructive result."),
        ("A \"step\" is not model-invariant. ", "Math-Shepherd's step is one calculator "
         "operation; Qwen writes 5.15 per solution, 59% of them narration. Per-step rates "
         "across generators are rates over different units."),
    ]
    right = [
        ("Projected, not measured, accuracy. ", "The corpus is fixed, so gating cannot change "
         "what the model wrote. Final-answer accuracy is projected under an explicit repair "
         "model and labelled PROJECTED throughout."),
        ("Derived graphs carry 5.7% edge error, ", "hand-validated — and that is a lower bound, "
         "since dependencies routed through unannotated lines are invisible to any "
         "operand-matching scheme."),
        ("Three generators, one band. ", "All 7–8B instruction-tuned models on GSM8K. Nothing "
         "here establishes the relationship continues to a 70B model or to a different task."),
    ]
    bullets(s, left, 0.7, 2.05, 5.8, 4.2, size=13.5, gap=14)
    bullets(s, right, 6.9, 2.05, 5.7, 4.2, size=13.5, gap=14)
    text(s, "Every number in this thesis is labelled MEASURED or PROJECTED, and the two are never blended.",
         0.7, 6.5, 11.9, 0.5, size=15, bold=True, color=SLATE)
    note(s, "I would rather state these than have the panel find them. The one I "
            "would flag hardest is the corpus size: section 8.6 varies three "
            "levers and every effect sits inside its own interval, which is why "
            "the conclusion is that the corpus, not the probe, is the binding "
            "constraint.")

    # ---------------------------------------------------------------- 18 repro
    s = blank(prs)
    title_of(s, "Reproducibility", "Checklist: code repository finalized, documented, reproducible")
    for i, (v, l) in enumerate([("360", "tests — no GPU,\nno network required"),
                                ("89", "pages, 10 chapters,\ncompiles from source"),
                                ("21", "references, every\narXiv id verified"),
                                ("15", "figures, regenerated\nby two scripts")]):
        stat(s, v, l, 0.7 + i * 3.1, 2.0, w=2.9, vsize=38)
    text(s, "Guards that stop numbers drifting", 0.7, 3.75, 11.9, 0.4, size=17,
         bold=True, color=SLATE, font=HEAD)
    table(s, [
        ["Guard", "What it prevents"],
        ["regression tests on every refuted claim", "a refutation quietly stopping reproducing"],
        ["check_citations.py", "a cited arXiv id with no bibliography entry"],
        ["check_crossrefs.py", "a reference resolving to the wrong chapter after a renumber"],
        ["check_prose.py --baseline/--compare", "an editing pass silently moving a number"],
        ["annotation-rate and solve-rate gates", "a rate measured on a biased subset of steps"],
        ["feasibility check at config time", "a run whose budget sits below its own Kotte floor"],
        ["select_and_fit cannot take test indices", "the winner's curse becoming the headline"],
    ], 0.7, 4.25, 11.9, col_w=[5, 7], size=12, h_row=0.35)
    note(s, "Public repository, dual-licensed MIT or Apache 2.0. pip install -e "
            "'.[dev]' then pytest reproduces every CPU result from a fresh clone. "
            "GPU results are committed as artifacts rather than regenerated, "
            "because they cost Kaggle sessions.")

    # ---------------------------------------------------------------- 19 publication
    s = blank(prs)
    title_of(s, "Publication status and future work",
             "Checklist: publication status · future work identifies genuine next steps")
    chip(s, "PAPER WRITTEN — NOT YET SUBMITTED", 0.7, 2.0, w=4.6, h=0.42, fill=TERRA)
    text(s, "6 pages, IEEE conference format, compiled and checked. Every numeric literal in it is held "
            "against the thesis by a script. Outstanding before submission: guide's review, venue "
            "selection, and the venue's LLM-usage disclosure policy.",
         0.7, 2.6, 11.9, 0.9, size=14)
    text(s, "Future work — four genuine next steps", 0.7, 3.6, 11.9, 0.4, size=17,
         bold=True, color=SLATE, font=HEAD)
    table(s, [
        ["Direction", "Why it is the next step", "Cost"],
        ["ARES-style conditioning under a budget",
         "Scoring against VERIFIED premises reaches 90.3% F1 on propagated errors — but assumes a "
         "verified prefix, which a budgeted gate cannot supply. Which fraction to verify is open.",
         "a paper"],
        ["More first-bad-step labels",
         "Only 108 exist in the whole corpus, 40 in test. This is the binding constraint on the one "
         "constructive result in ch. 8.",
         "new corpus"],
        ["A fourth generator outside 7–8B",
         "The monotone relationship is measured across a narrow band. Whether it continues to a 70B "
         "model is untested.",
         "1 GPU-day"],
        ["Risk control under gate-induced dependence",
         "The theoretical problem is open. Barber et al. relax exchangeability for EXOGENOUS drift; "
         "the violation here is endogenous — the gate's own decisions move the distribution.",
         "open"],
    ], 0.7, 4.1, 11.9, col_w=[3.5, 7, 1.7], size=11.5, h_row=0.62)
    note(s, "The first is the one I would actually pursue. It also reframes "
            "chapter 7: allocation would stop being about catching errors and "
            "become about building a trustworthy prefix to condition later checks "
            "on — and ancestor count is a far more natural signal for that "
            "objective than for the one tested here.")

    # ---------------------------------------------------------------- 20 checklist
    s = blank(prs)
    title_of(s, "Final Review checklist — status",
             "Stated plainly, including what is not done")
    done = [
        "Full system implemented and demonstrated end to end",
        "Final results with statistical rigor — 64 seeds, Wilson intervals, clustered bootstrap",
        "Comparison tables against multiple baselines — 4 verifier classes, 6 allocation policies, 8 gate conditions",
        "Ablation studies — FA = 0, oracle score, three probe levers, three equivalence relations",
        "Limitations honestly discussed",
        "Contribution restated with magnitudes",
        "Complete thesis, 10 chapters, institute format",
        "Code repository documented and reproducible — 360 tests",
        "Future work identifies genuine next steps",
        "Viva readiness — every design choice has a recorded reason",
    ]
    todo = [
        "Plagiarism report — not yet generated",
        "Publication status — paper written, not yet submitted",
        "Panel feedback tracker from Reviews 0–2 — to be compiled",
    ]
    chip(s, f"COMPLETE  ({len(done)})", 0.7, 1.95, w=2.6, h=0.4, fill=TEAL)
    bullets(s, [f"·  {d}" for d in done], 0.7, 2.5, 7.6, 3.9, size=12.5, gap=7)
    chip(s, f"OUTSTANDING  ({len(todo)})", 8.7, 1.95, w=3.0, h=0.4, fill=TERRA)
    bullets(s, [f"·  {d}" for d in todo], 8.7, 2.5, 3.9, 2.0, size=12.5, gap=9)
    text(s, "These three are administrative rather than technical, and none blocks the "
            "work itself. I would rather name them than have the panel find them.",
         8.7, 4.7, 3.9, 1.4, size=12.5, italic=True, color=MUTED)
    note(s, "Being straight about the three gaps costs nothing and buys "
            "credibility for the ten that are done. The plagiarism report needs "
            "the institute's tool; the paper needs the guide's review before it "
            "goes anywhere; the feedback tracker I can assemble from the review "
            "records.")

    # ---------------------------------------------------------------- 21 close
    s = blank(prs, dark=True)
    text(s, "The project is more useful for having failed.", 0.9, 2.1, 11.5, 0.8,
         size=34, bold=True, color=WHITE, font=HEAD)
    text(s, "A working gate on GSM8K would have been one more entry in a crowded area. A measurement "
            "of WHY the obvious design cannot work — with the failure points separated, attributed and "
            "quantified — is a result that transfers.",
         0.9, 3.15, 11.5, 1.1, size=17, color=RGBColor(0xD4, 0xDE, 0xE2))
    text(s, "Selective verification of LLM reasoning is bottlenecked at the SIGNAL and at the BUDGET. "
            "The verifier's net-negative result is downstream of the score rather than a third, "
            "independent defect, and the calibration cannot certify the quantity of interest while the "
            "score does not rank it.",
         0.9, 4.5, 11.5, 1.3, size=15, color=RGBColor(0xC8, 0xD2, 0xD6), italic=True)
    text(s, "Thank you.", 0.9, 6.1, 5.0, 0.5, size=20, bold=True, color=WHITE, font=HEAD)
    note(s, "Close on the framing, not on a summary. If there is one question I "
            "expect, it is why a negative result is a thesis — and the answer is "
            "the attribution: we can say WHICH component fails and by how much, "
            "which is what makes it transfer to anyone else building this.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    print(f"wrote {OUT}  ({len(prs.slides.__iter__.__self__._sldIdLst)} slides)")


if __name__ == "__main__":
    build()
