"""Generate the thesis figures. [CPU]

    python scripts/make_figures.py

Writes PNG + PDF to docs/thesis/figures/.

Figures 6-8 are computed from the committed corpus, so they cannot drift from
the numbers in Chapter 7. The rest are measured constants, each carrying the
script that produced it -- hardcoded rather than recomputed because the sources
range from a 900-second CPU sweep to a 7h40m GPU run, and a figure script that
takes an afternoon does not get re-run.

PALETTE

Categorical slots 1-4 from a validated palette (blue / orange / aqua / yellow),
in fixed order, never cycled. The order is the colour-blindness safety
mechanism, not decoration: worst adjacent CVD dE 9.1, worst adjacent
normal-vision dE 22.9, both clear of their floors. Aqua and yellow sit below
3:1 contrast on this surface, so every chart using them ships visible direct
labels -- the documented relief for that warning.

Text is always ink, never a series colour. A coloured mark beside a label
carries identity; the label itself does not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

OUT = Path("docs/thesis/figures")

# ---- design tokens --------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#e8e7e3"

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # blue orange aqua yellow
CRITICAL = "#d03b3b"
GOOD = "#0ca30c"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
})


def style(ax, *, grid_axis="y"):
    """Recessive axes: two spines, a faint grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return ax


def note(fig, text, y=-0.06):
    """Source line, below everything.

    Placed at negative figure coords on purpose: `bbox_inches="tight"` grows
    the canvas to include it, so it cannot collide with the x-axis label the
    way a y=0.005 placement did.
    """
    fig.text(0.0, y, text, fontsize=7, color=INK_3, ha="left", va="top")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png")


# ---- measured constants ---------------------------------------------------
# Each block names the script that produced it.

# scripts/exp_measure_error_rate.py, scripts/exp_generator_transfer.py
GAP = {
    "Mistral-7B-SFT\n~45% GSM8K": {"local": 0.1708, "global": 0.7106, "c1": 0.7848,
                                   "ci": (0.781, 0.789)},
    "Qwen2.5-7B-Instruct\n80% GSM8K": {"local": 0.0813, "global": 0.5772, "c1": 0.9040,
                                       "ci": (0.840, 0.944)},
}

# scripts/exp_verifier_scope.py
LOOKBACK = [(0, 0.0000), (1, 0.1197), (2, 0.1690), (3, 0.1880), (5, 0.1983), (6, 0.1999)]
LOOKBACK_XTICKS = ["0", "1", "2", "3", "5", "∞"]

# runs/semantic_scope_*.json, via scripts/gpu_semantic_scope.py
REACH = [
    ("arithmetic, step-local", 0.0000, 0.0000),
    ("same-model critic", 0.0000, 0.0000),
    ("arithmetic, unbounded lookback", 0.1999, 0.0000),
    ("independent judge (Qwen2.5-7B)", 0.2283, 0.0200),
    ("task PRM (Math-Shepherd-7B)", 0.9033, 0.0987),
]

# scripts/exp_benchmark_compare.py
DEPTH = {
    "depth": [1, 2, 3, 4, 5, 6],
    "n": [171, 2988, 2431, 1010, 308, 52],
    "front": [0.1503, 0.2049, 0.2777, 0.3269, 0.3618, 0.4059],
    "uniform": [0.1518, 0.2049, 0.2632, 0.2992, 0.3221, 0.3578],
    "back": [0.1495, 0.2042, 0.2382, 0.2577, 0.2632, 0.2872],
    "depth_policy": [0.1518, 0.2048, 0.2290, 0.2455, 0.2533, 0.2779],
}

# scripts/exp_measure_error_rate.py
POSITION = {
    "pos": [1, 2, 3, 4, 5, 6, 7, 8],
    "local": [0.1095, 0.1098, 0.1319, 0.1662, 0.1981, 0.2031, 0.1939, 0.2182],
    "glob": [0.3077, 0.5159, 0.6475, 0.7250, 0.7963, 0.8569, 0.8730, 0.9124],
}

# scripts/exp_gate_pipeline.py -- projected, see ch.7 on the repair model
VERIFIER_VALUE = [
    ("arithmetic\nstep-local", 0.0000, 0.0000, 0.8022),
    ("independent\njudge", 0.2283, 0.0200, 0.8022),
    ("task PRM\nas measured", 0.9033, 0.0987, 0.7637),
    ("task PRM\nablation FA=0", 0.9033, 0.0000, 0.9231),
    # Same verifier, same measured false-alarm rate, oracle score. This is the
    # point of the figure: the verifier is downstream of the score.
    ("task PRM\nORACLE score", 0.9033, 0.0987, 0.9780),
]
BASELINE_ACC = 0.8022

# scripts/gpu_probe_states.py -- the probe result
PROBE = Path("runs/probe_qwen25_7b.json")
# scripts/exp_score_quality_threshold.py
SCORE_QUALITY = Path("runs/score_quality_threshold.json")
# scripts/exp_probe_variants.py -- the held-out learning curve
PROBE_VARIANTS = Path("runs/probe_variants.json")

# scripts/exp_gate_pipeline.py --probe : selective risk by score quality
SCORE_VS_RISK = [
    ("token + semantic", 0.5742, 0.1491, 0.1494, 0.1538),
    ("probe, layer 25", 0.6968, 0.1432, 0.1363, 0.1394),
    ("oracle", 1.0000, 0.0885, 0.0885, 0.0885),
]
NO_GATE_RISK = 0.1554

CORPUS = Path("runs/generated_qwen25_7b.jsonl")
FEATURES = Path("runs/uncertainty_qwen25_7b_sem.jsonl")


# ---- figures --------------------------------------------------------------


def fig1_gap():
    """C1: global error decomposes, and the inherited part grows with model strength."""
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    style(ax, grid_axis="x")

    labels = list(GAP)
    y = np.arange(len(labels))[::-1]
    for yi, lab in zip(y, labels, strict=True):
        d = GAP[lab]
        loc_part = d["global"] * (1 - d["c1"])
        inh_part = d["global"] * d["c1"]
        # 2px surface gap between segments, per the mark spec
        ax.barh(yi, loc_part, height=0.5, color=S2, edgecolor=SURFACE, linewidth=2,
                zorder=3)
        ax.barh(yi, inh_part, height=0.5, left=loc_part, color=S1,
                edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(loc_part / 2, yi, f"{loc_part:.2f}", ha="center", va="center",
                color="white", fontsize=8, fontweight="bold", zorder=4)
        ax.text(loc_part + inh_part / 2, yi, f"{inh_part:.2f}", ha="center",
                va="center", color="white", fontsize=8.5, fontweight="bold", zorder=4)
        ax.text(d["global"] + 0.015, yi, f"{d['c1']:.0%} inherited",
                va="center", color=INK, fontsize=8.5, fontweight="bold")

    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("global error rate, within wrong-answer solutions")
    ax.set_title("Most globally-wrong steps are arithmetically perfect —\n"
                 "and more so on the stronger generator", loc="left", pad=10)
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color=S2, label="locally invalid — a verifier can see it"),
        Line2D([], [], marker="s", ls="", color=S1, label="inherited corruption — no verifier can"),
    ], loc="upper right", bbox_to_anchor=(1.0, -0.22), ncol=2)
    note(fig, "Math-Shepherd 93,129 steps; Qwen corpus 2,573 steps. "
              "C1 measured on checkable steps.", y=-0.30)
    save(fig, "fig1-the-gap")


def fig2_reach():
    """C3: reach spans the unit interval and needs independence AND specialisation."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    style(ax, grid_axis="x")

    labels = [r[0] for r in REACH]
    y = np.arange(len(REACH))[::-1]
    h = 0.34
    for yi, (_, scope, fa) in zip(y, REACH, strict=True):
        ax.barh(yi + h / 2 + 0.01, scope, height=h, color=S1, zorder=3)
        ax.barh(yi - h / 2 - 0.01, fa, height=h, color=S2, zorder=3)
        ax.text(scope + 0.012, yi + h / 2 + 0.01, f"{scope:.4f}", va="center",
                color=INK, fontsize=8)
        if fa > 0:
            ax.text(fa + 0.012, yi - h / 2 - 0.01, f"{fa:.4f}", va="center",
                    color=INK_2, fontsize=8)

    ax.set_yticks(y, labels, fontsize=8.5)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("rate on the arithmetic-blind population")
    ax.set_title("Verifier reach spans the unit interval", loc="left", pad=10)
    # Upper right: the two zero-scope rows leave that quadrant empty. Anywhere
    # lower and the legend lands on the PRM bar.
    ax.legend(handles=[
        Line2D([], [], marker="s", ls="", color=S1, label="scope — P(detect | globally wrong)"),
        Line2D([], [], marker="s", ls="", color=S2, label="false alarm — P(flag | correct)"),
    ], loc="upper right", bbox_to_anchor=(1.0, 1.02))
    note(fig, "1,500 arithmetic-blind steps + 750 controls. Only the independent, "
              "task-trained verifier closes the gap — and ch.7 shows its false-alarm rate undoes it.")
    save(fig, "fig2-verifier-reach")


def fig3_lookback():
    """C3: widening the window saturates, because 80% has no upstream arithmetic error."""
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    style(ax)

    x = np.arange(len(LOOKBACK))
    v = [s for _, s in LOOKBACK]
    ax.plot(x, v, color=S1, marker="o", zorder=3)
    ax.axhline(0.1999, color=INK_3, ls=(0, (4, 4)), lw=1.2, zorder=2)
    ax.text(0.15, 0.207, "saturates at 0.1999", color=INK_2, fontsize=8)
    ax.annotate("80.0% of inherited corruption has\nno upstream arithmetic error at all",
                xy=(5, 0.1999), xytext=(1.6, 0.085), fontsize=8, color=INK_2,
                arrowprops=dict(arrowstyle="->", color=INK_3, lw=1))

    ax.set_xticks(x, LOOKBACK_XTICKS)
    ax.set_ylim(0, 0.26)
    ax.set_xlabel("arithmetic verifier lookback window, k")
    ax.set_ylabel("scope")
    ax.set_title("Reach is not window size", loc="left", pad=10)
    note(fig, "35,535 locally-valid, globally-wrong Math-Shepherd steps.")
    save(fig, "fig3-lookback-saturation")


def fig4_allocation():
    """C4: the policy gap opens up exactly where propagation has room."""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    style(ax)

    d = DEPTH["depth"]
    series = [("front", DEPTH["front"], S2), ("uniform", DEPTH["uniform"], INK_3),
              ("back", DEPTH["back"], S3), ("depth (ancestors)", DEPTH["depth_policy"], S1)]
    nudge = {"back": 0.006, "depth (ancestors)": -0.006}
    for name, vals, c in series:
        ax.plot(d, vals, color=c, marker="o", zorder=3)
        ax.text(d[-1] + 0.08, vals[-1] + nudge.get(name, 0.0), name, color=INK,
                fontsize=8.5, va="center")

    ax.set_xlim(0.8, 8.4)
    ax.set_xticks(d)
    ax.set_xlabel("reasoning-graph depth (longest directed path)")
    ax.set_ylabel("final-answer error")
    ax.set_title("Front-loading is worst, and the gap grows with depth",
                 loc="left", pad=10)
    ax.set_ylim(0.125, 0.425)
    for xi, n in zip(d, DEPTH["n"], strict=True):
        ax.text(xi, 0.131, f"n={n:,}", ha="center", color=INK_3, fontsize=7)
    note(fig, "6,974 derived GSM8K graphs, budget 37.5%. Spread 0.0033 at depth 2 → 0.1279 at depth 6.")
    save(fig, "fig4-allocation-by-depth")


def fig5_position():
    """C4b: later steps are harder, and the two curves diverge as corruption accrues."""
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    style(ax)

    p = POSITION["pos"]
    ax.plot(p, POSITION["glob"], color=S1, marker="o", zorder=3)
    ax.plot(p, POSITION["local"], color=S2, marker="o", zorder=3)
    ax.fill_between(p, POSITION["local"], POSITION["glob"], color=S1, alpha=0.07,
                    zorder=1)
    ax.text(8.15, POSITION["glob"][-1], "global error", color=INK, fontsize=8.5,
            va="center")
    ax.text(8.15, POSITION["local"][-1], "local error", color=INK, fontsize=8.5,
            va="center")
    ax.text(5.0, 0.55, "the shaded gap is\ninherited corruption", color=INK_2,
            fontsize=8.5, ha="center")

    ax.set_xlim(0.7, 10.2)
    ax.set_xticks(p)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("step position in the solution")
    ax.set_ylabel("error rate")
    ax.set_title("Later steps are harder — which argues against verifying early",
                 loc="left", pad=10)
    note(fig, "Math-Shepherd. corr(position, local error) = +0.950; +0.866 on the Qwen corpus.")
    save(fig, "fig5-position-gradient")


# ---- figures computed from the corpus -------------------------------------


def _pipeline_data():
    from car.data.splits import make_splits
    from car.generation.replay import load_corpus
    from car.uncertainty.composite import CompositeScorer

    corpus = load_corpus(CORPUS, FEATURES, prefix="qwen")
    by_id = {ex.example_id: ex for ex in corpus}
    splits = make_splits([ex.to_example() for ex in corpus], dev_frac=0.3,
                         cal_frac=0.3, salt="car-v1")
    dev = [s.features for ex in splits.dev for s in by_id[ex.example_id].steps]

    def scores(weights, split):
        sc = CompositeScorer(weights).fit(dev)
        s, y = [], []
        for ex in split:
            for st in by_id[ex.example_id].steps:
                s.append(sc.score(st.features))
                y.append(st.global_ok)
        return np.asarray(s), np.asarray(y)

    return by_id, splits, scores


def fig6_roc():
    """C8: neither signal ranks the risk, and the harness proves it can detect one."""
    from sklearn.metrics import roc_auc_score, roc_curve

    _, splits, scores = _pipeline_data()
    TOKEN = {"token_entropy": 1.0, "max_surprisal": 0.5, "mean_logprob": 1.0}
    SEM = {"semantic_divergence": 1.0}

    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    style(ax, grid_axis="both")

    for name, w, c in (("token-level", TOKEN, S2),
                       ("semantic divergence", SEM, S3),
                       ("both", {**TOKEN, **SEM}, S1)):
        s, y = scores(w, splits.test)
        fpr, tpr, _ = roc_curve(~y, s)
        auc = roc_auc_score(~y, s)
        ax.plot(fpr, tpr, color=c, zorder=3, label=f"{name} — AUROC {auc:.4f}")

    # Control: three synthetic features at 1 sigma, combined through the same
    # scorer -- exactly what `--synthetic-signal 1.0` does in ch. 7. A single
    # feature gives 0.77 and would disagree with the 0.8668 in the text; the
    # combination is what averages the noise down.
    from car.types import UncertaintyFeatures
    from car.uncertainty.composite import CompositeScorer

    _, y = scores(TOKEN, splits.test)
    rng = np.random.default_rng(0)
    feats = [
        UncertaintyFeatures(
            token_entropy=float(rng.normal(0.0 if ok else 1.0, 1.0)),
            max_surprisal=float(rng.normal(0.0 if ok else 1.0, 1.0)),
            mean_logprob=float(rng.normal(0.0 if ok else -1.0, 1.0)),
        )
        for ok in y
    ]
    synth = CompositeScorer(TOKEN).fit(feats).score_many(feats)
    fpr, tpr, _ = roc_curve(~y, synth)
    ax.plot(fpr, tpr, color=INK_3, ls=(0, (5, 3)), lw=1.5, zorder=2,
            label=f"synthetic 1σ signal — {roc_auc_score(~y, synth):.4f}")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.5, zorder=1, label="chance — 0.5000")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Generator uncertainty does not rank\nglobal step error", loc="left",
                 pad=10)
    ax.legend(loc="lower right")
    note(fig, "925 held-out steps. The dashed control is the same construction as "
              "--synthetic-signal 1.0 in ch. 7, which scores 0.8668 on its own draw; "
              "the harness finds a signal when one is there.")
    save(fig, "fig6-roc")


def fig7_alpha_sweep():
    """C8b: the measured risk ignores the target, at every alpha that binds."""
    from car.agent.loop import CARAgent
    from car.conformal import AdaptiveCalibrator, SplitConformalCalibrator
    from car.generation.replay import (
        ReplayStepGenerator, answer_is_correct, score_final_answer,
    )
    from car.types import Decision
    from car.uncertainty.composite import CompositeScorer
    from car.verification.scoped import ScopedVerifier

    by_id, splits, scores = _pipeline_data()
    W = {"token_entropy": 1.0, "max_surprisal": 0.5, "mean_logprob": 1.0,
         "semantic_divergence": 1.0}
    dev = [s.features for ex in splits.dev for s in by_id[ex.example_id].steps]
    scorer = CompositeScorer(W).fit(dev)
    cal_s, cal_y = scores(W, splits.calibration)
    _, test_y = scores(W, splits.test)
    mu = float(np.mean(~test_y))

    labels = {(by_id[ex.example_id].question, i): st.global_ok
              for ex in splits.test
              for i, st in enumerate(by_id[ex.example_id].steps)}

    alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
    out = {"split": [], "car": []}
    for a in alphas:
        for key, cal in (("split", SplitConformalCalibrator(alpha=a).fit(cal_s, cal_y)),
                         ("car", AdaptiveCalibrator(alpha=a, gamma=0.005, epsilon=0.2,
                                                    update_mode="ipw", seed=0
                                                    ).fit(cal_s, cal_y))):
            agent = CARAgent(
                generator=ReplayStepGenerator(list(by_id.values())), scorer=scorer,
                calibrator=cal,
                verifier=ScopedVerifier(labels, scope=0.9033, false_alarm=0.0, seed=0),
                budget_per_question=2, max_steps=16,
                finalise=score_final_answer, score_answer=answer_is_correct,
            )
            acc = [r.label for t in agent.run_all(splits.test) for r in t.steps
                   if r.decision == Decision.CONTINUE and r.label is not None]
            out[key].append(float(np.mean([not x for x in acc])))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    style(ax, grid_axis="both")

    ax.plot([0, 0.32], [0, 0.32], color=GRID, lw=1.5, zorder=1)
    ax.text(0.232, 0.243, "target met", color=INK_3, fontsize=8, rotation=40,
            ha="center", va="bottom")
    ax.axvspan(0, mu, color=S4, alpha=0.10, zorder=0)
    ax.text(mu / 2, 0.028, "α binds here", ha="center", color=INK_2, fontsize=8.5)
    ax.text(mu + 0.006, 0.028, "α vacuous —\nmet by verifying nothing", color=INK_3,
            fontsize=8)
    ax.axvline(mu, color=INK_3, ls=(0, (3, 3)), lw=1.2, zorder=2)

    ax.plot(alphas, out["split"], color=S1, marker="o", zorder=4)
    ax.plot(alphas, out["car"], color=S2, marker="o", zorder=4)
    ax.text(0.305, out["split"][-1] - 0.012, "split conformal", color=INK, fontsize=8.5)
    ax.text(0.305, out["car"][-1] + 0.008, "CAR (adaptive)", color=INK, fontsize=8.5)

    ax.set_xlim(0, 0.40)
    ax.set_ylim(0, 0.32)
    ax.set_xlabel("target risk α")
    ax.set_ylabel("measured selective risk on the accepted set")
    ax.set_title("The gate does not hold its target", loc="left", pad=10)
    note(fig, f"Base risk μ = {mu:.4f}. At α = 0.05 the measured risk is ~3× the target, "
              "and it barely moves across the sweep.")
    save(fig, "fig7-alpha-sweep")
    return out, alphas, mu


def fig8_pareto(sweep, alphas, mu):
    """The accuracy-cost trade-off with the impossibility floor as a reference."""
    from car.conformal.feasibility import min_verification_rate

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    style(ax, grid_axis="both")

    v = np.linspace(0.0, 0.6, 300)
    # Kotte Prop. 3 solved for alpha: rate = (mu - a)/(1 - a)  ->  a = (mu - r)/(1 - r).
    # Any (rate, alpha) pair below this curve is unreachable by any
    # distribution-free method, whatever the score or calibrator.
    # Clipped at r = mu: past that the floor is zero and drawing it along the
    # x-axis reads as a data series rather than as a boundary.
    v = v[v <= mu]
    floor_alpha = np.array([(mu - r) / (1 - r) for r in v])
    ax.fill_between(v, 0, floor_alpha, color=CRITICAL, alpha=0.08, zorder=0)
    ax.plot(v, floor_alpha, color=CRITICAL, lw=1.5, ls=(0, (5, 3)), zorder=2)
    ax.text(0.185, 0.045, "unreachable by ANY distribution-free\n"
                          "method at this base risk  (Kotte Prop. 3)",
            color=CRITICAL, fontsize=8, va="center")
    ax.annotate("", xy=(0.075, 0.030), xytext=(0.180, 0.045),
                arrowprops=dict(arrowstyle="->", color=CRITICAL, lw=1))

    rates = {0.05: 0.046, 0.10: 0.092, 0.15: 0.141, 0.20: 0.170, 0.30: 0.218}
    xs = [rates[a] for a in alphas]
    ax.plot(xs, sweep["split"], color=S1, marker="o", zorder=4)
    # Stagger: the points are close together and level labels overlapped.
    for i, (a, x, y) in enumerate(zip(alphas, xs, sweep["split"], strict=True)):
        dy = -16 if i % 2 else 10
        ax.annotate(f"α={a:g}", (x, y), textcoords="offset points", xytext=(0, dy),
                    ha="center", color=INK_2, fontsize=7.5)
    ax.axhline(mu, color=INK_3, ls=(0, (3, 3)), lw=1.2, zorder=2)
    ax.text(0.515, mu + 0.004, f"no gate at all: μ = {mu:.4f}", color=INK_2,
            fontsize=8, ha="right")

    ax.set_xlim(0, 0.52)
    ax.set_ylim(0, 0.23)
    ax.set_xlabel("verification rate (fraction of steps checked)")
    ax.set_ylabel("measured selective risk")
    ax.set_title("Spending budget buys almost nothing", loc="left", pad=10)
    note(fig, "Split conformal on the Qwen corpus. The curve is flat: 4.6% → 21.8% "
              "verification moves risk by 0.005.")
    save(fig, "fig8-pareto-with-floor")


def fig9_verifier_value():
    """C8c: the base-rate effect that makes the best verifier net-negative."""
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    style(ax)

    names = [v[0] for v in VERIFIER_VALUE]
    acc = [v[3] for v in VERIFIER_VALUE]
    x = np.arange(len(names))
    colors = [INK_3, INK_3, CRITICAL, S3, GOOD]
    ax.bar(x, acc, width=0.55, color=colors, zorder=3)
    for xi, a in zip(x, acc, strict=True):
        ax.text(xi, a + 0.006, f"{a:.4f}", ha="center", color=INK, fontsize=8.5,
                fontweight="bold")

    ax.axhline(BASELINE_ACC, color=INK_2, ls=(0, (4, 3)), lw=1.4, zorder=4)
    ax.text(4.42, BASELINE_ACC + 0.004, "no gate\n0.8022", color=INK_2, fontsize=8)

    ax.set_xticks(x, names, fontsize=8.5)
    ax.set_ylim(0.65, 1.01)
    ax.set_ylabel("projected final-answer accuracy  (MODELLED)")
    ax.set_title("The verifier is not the problem, the score is.\n"
                 "Same verifier, same 9.87% false-alarm rate, three scores.",
                 loc="left", pad=14, x=-0.06)
    note(fig, "PROJECTED, not measured: replay cannot regenerate text, so repair "
              "is modelled. A false alarm can only fire on a step the gate chose "
              "to verify, and a good score rarely chooses a correct one.")
    save(fig, "fig9-verifier-value")


def fig10_probe_layers():
    """A real depth profile, and what more training data is actually worth.

    The right panel has been wrong twice. It first plotted the round-one curve,
    scored on the SELECTION split -- the same data the layer and C were chosen
    on, so biased and measured on the wrong population. The replacement walked
    an UNSHUFFLED pool, so its composition drifted with its size and it read as
    flat. This plots the held-out curve over a shuffled pool: it rises, and at
    fixed layer and C the doubling is worth +0.0105 with the interval spanning
    zero.
    """
    import json

    if not PROBE.exists():
        print("  (skipping fig10: probe result not present)")
        return
    d = json.loads(PROBE.read_text(encoding="utf-8"))
    per = {int(k): v for k, v in d["per_layer_select"].items()}
    curve = None
    if PROBE_VARIANTS.exists():
        curve = json.loads(PROBE_VARIANTS.read_text(encoding="utf-8"))["learning_curve_pooled"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.2),
                                  gridspec_kw={"width_ratios": [2.1, 1]})
    style(ax)
    style(ax2)

    xs = sorted(per)
    ax.plot(xs, [per[x] for x in xs], color=S1, marker="o", markersize=4, zorder=3)
    ax.scatter([d["layer"]], [per[d["layer"]]], s=90, color=S2, zorder=4)
    ax.annotate(f"layer {d['layer']}, selected", (d["layer"], per[d["layer"]]),
                textcoords="offset points", xytext=(-8, -22), ha="right",
                color=INK, fontsize=8.5)
    ax.axhline(0.5742, color=INK_3, ls=(0, (4, 4)), lw=1.2, zorder=2)
    ax.text(0.4, 0.585, "ch. 7 best signal, 0.5742", color=INK_2, fontsize=8)
    ax.set_xlabel("hidden layer (0 = embeddings, 28 = final)")
    ax.set_ylabel("AUROC on the selection split")
    ax.set_ylim(0.55, 0.92)
    ax.set_title("Internal states encode step soundness", loc="left", pad=10)

    if curve:
        ns = [row[0] for row in curve]
        ax2.plot(ns, [row[1] for row in curve], color=S3, marker="o", zorder=3)
        ax2.axhline(0.6968, color=INK_3, ls=(0, (4, 4)), lw=1.2, zorder=2)
        ax2.text(ns[0], 0.699, "as published, 0.6968", color=INK_2, fontsize=7.5,
                 va="bottom")
        ax2.set_ylim(0.54, 0.74)
        ax2.set_title("and more data helps a little", loc="left", pad=10)
        src = ("Right: held-out AUROC over a SHUFFLED pooled training set, so "
               "size is the only thing changing. At fixed layer and C, doubling\n"
               "the data is worth +0.0105 (CI [-0.035, +0.057]). Two earlier "
               "versions of this panel were wrong: the first plotted the\n"
               "selection-split curve, the second an unshuffled pool whose "
               "composition drifted with its size.")
    else:
        ns = [n for n, _ in d["learning_curve"]]
        ax2.plot(ns, [a for _, a in d["learning_curve"]], color=S3, marker="o",
                 zorder=3)
        ax2.set_title("selection-split curve (biased)", loc="left", pad=10)
        src = ("Right: the round-one curve, scored on the selection split -- run "
               "scripts/exp_probe_variants.py for the held-out one.")
    ax2.set_xlabel("probe training steps")
    ax2.set_ylabel("AUROC on TEST" if curve else "AUROC on selection")
    note(fig, "Left: every layer tried, not just the winner. " + src, y=-0.12)
    save(fig, "fig10-probe-layers")


def fig11_score_vs_risk():
    """The payoff and the limit in one picture."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    style(ax, grid_axis="both")

    aurocs = [a for _, a, *_ in SCORE_VS_RISK]
    # The two alpha settings give nearly the same curve, which is itself the
    # finding: the measured risk barely responds to the target. Labelled at the
    # right end, where the points are not crowded.
    for j, (alpha, col, lab) in enumerate(
            [(0.05, S1, "gate set to α = 0.05"), (0.10, S2, "gate set to α = 0.10")]):
        risks = [r[2 + j] for r in SCORE_VS_RISK]
        ax.plot(aurocs, risks, color=col, marker="o", zorder=3)
        ax.axhline(alpha, color=col, ls=(0, (3, 3)), lw=1.2, zorder=2)
        ax.text(0.515, alpha + 0.0015, f"its target, {alpha:g}", color=col,
                fontsize=8, va="bottom")
        ax.text(1.015, risks[-1] + (0.004 if j == 0 else -0.004), lab,
                color=INK, fontsize=8.5, va="center")

    ax.axhline(NO_GATE_RISK, color=INK_3, ls=(0, (5, 3)), lw=1.2, zorder=2)
    ax.text(0.515, NO_GATE_RISK + 0.0015, "no gate at all", color=INK_2,
            fontsize=8, va="bottom")

    ax.set_xlim(0.50, 1.32)
    # Names on the axis rather than floating in the plot: the three
    # labels collided with each other and with the y ticks.
    ax.set_xticks([0.5742, 0.6968, 1.0],
                  ["token + semantic\n0.5742",
                   "probe, layer 25\n0.6968",
                   "oracle\n1.0000"])
    ax.set_ylim(0.03, 0.175)
    ax.set_xlabel("score AUROC")
    ax.set_ylabel("measured selective risk")
    ax.set_title("A better score helps, and does not reach the target",
                 loc="left", pad=10)
    note(fig, "Same corpus, calibrator and budget; only the score differs. Even a "
              "perfect score misses alpha = 0.05 by 1.8x -- past that point the "
              "budget binds, not the ranking.")
    save(fig, "fig11-score-vs-risk")


def fig12_score_quality():
    """Where the verifier stops being a liability, and why AUROC misses it."""
    import json

    blob = json.loads(SCORE_QUALITY.read_text(encoding="utf-8"))
    base = blob["baseline_projected_accuracy"]
    lo, hi = blob["crossing_ci_last_negative"], blob["crossing_ci_first_positive"]

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))

    # ---- left: projected accuracy against score quality --------------------
    ax = style(axes[0], grid_axis="both")
    for key, col, lab in [("sweep_b", S2, "false alarms off"),
                          ("sweep_a", S1, "measured 9.87% false alarm")]:
        pts = blob[key]
        x = [p["auroc_target"] for p in pts]
        y = [p["projected_accuracy"] for p in pts]
        ax.fill_between(x, [p["projected_accuracy_lo"] for p in pts],
                        [p["projected_accuracy_hi"] for p in pts],
                        color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, y, color=col, lw=1.8, zorder=3)
        ax.text(0.995, y[-1] + 0.004, lab, color=col, fontsize=8, ha="right")

    # The band the sweep cannot resolve. Drawing it is the point: a single
    # crossing number would be quoting seed noise to three decimals.
    ax.axvspan(lo, hi, color=INK_3, alpha=0.10, lw=0, zorder=1)
    ax.axhline(base, color=INK_3, ls=(0, (5, 3)), lw=1.2, zorder=2)
    ax.text(0.553, base + 0.004, f"no gate at all, {base:.4f}", color=INK_2,
            fontsize=8, va="bottom")
    ax.text((lo + hi) / 2, 0.957, f"crossing\n{lo:g}–{hi:g}", color=INK_2,
            fontsize=8, ha="center", va="top")

    # Both labels go to the RIGHT of their marker: token+sem sits at x = 0.5742,
    # close enough to the axis that a left-placed label overruns the y ticks.
    for auroc, acc, lab, dy in [(0.5742, 0.7912, "token+sem", 5),
                                (0.6968, 0.7802, "probe", -12)]:
        ax.plot([auroc], [acc], marker="D", ms=5.5, color=INK, zorder=5)
        ax.annotate(lab, (auroc, acc), textcoords="offset points",
                    xytext=(7, dy), fontsize=8, color=INK, ha="left")

    ax.set_xlim(0.54, 1.0)
    ax.set_ylim(0.755, 0.97)
    ax.set_xlabel("score AUROC")
    ax.set_ylabel("projected accuracy (MODELLED)")
    ax.set_title("The threshold is made of false alarms", loc="left", pad=10)

    # ---- right: first-bad-step recall, where the real scores come apart ----
    ax = style(axes[1], grid_axis="both")
    pts = blob["sweep_a"]
    x = [p["auroc_target"] for p in pts]
    ax.plot(x, [p["first_bad_recall"] for p in pts], color=S1, lw=1.8,
            zorder=3, label="synthetic score")
    ax.plot(x, [p["recall"] for p in pts], color=INK_3, lw=1.4,
            ls=(0, (4, 3)), zorder=3, label="  (any bad step)")

    for auroc, fbr, lab, dx in [(0.5742, 0.3590, "token+sem", 5),
                                (0.6968, 0.3077, "probe", 5)]:
        ax.plot([auroc], [fbr], marker="D", ms=5.5, color=INK, zorder=5)
        ax.annotate(lab, (auroc, fbr), textcoords="offset points",
                    xytext=(dx, -11), fontsize=8, color=INK)

    ax.set_xlim(0.54, 1.0)
    ax.set_xlabel("score AUROC")
    ax.set_ylabel("first-bad-step recall")
    ax.set_title("Equal AUROC, unequal value", loc="left", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    # Wrapped by hand: `note` does not wrap, and bbox_inches="tight" grows the
    # canvas to fit a single long line, which flattened this figure to 6:1.
    note(fig,
         "Left: 64 seeds per point, band is a 95% CI on the mean; the shaded column is "
         "where the sweep cannot separate the gate from doing nothing.\n"
         "Right: the probe sits BELOW the curve for its own AUROC and the composite "
         "above it. The probe's score correlates +0.18 with step position and the\n"
         "composite's -0.28, while the projection only pays for the FIRST bad step. "
         "Calls/q differ across the three (1.18 synthetic, 1.09 token+sem, 0.96\n"
         "probe); sweep D pins the rate and the ordering holds.",
         y=-0.10)
    save(fig, "fig12-score-quality")


def main():
    print("writing figures to", OUT)
    fig1_gap()
    fig2_reach()
    fig3_lookback()
    fig4_allocation()
    fig5_position()

    if not (CORPUS.exists() and FEATURES.exists()):
        print("  (skipping figs 6-8: corpus/features not present)")
        return 0
    fig6_roc()
    sweep, alphas, mu = fig7_alpha_sweep()
    fig8_pareto(sweep, alphas, mu)
    fig9_verifier_value()
    fig10_probe_layers()
    fig11_score_vs_risk()
    if SCORE_QUALITY.exists():
        fig12_score_quality()
    else:
        print("  (skipping fig 12: run scripts/exp_score_quality_threshold.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
