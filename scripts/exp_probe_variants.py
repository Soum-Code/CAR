"""What should a step-level probe be trained on, and is it starved? [CPU]

Two gaps chapter 7 leaves open, both answerable from the saved hidden states
without touching a GPU again.

GAP 1 -- IS THE PROBE STARVED? (ch. 7.6)
    The probe reaches AUROC 0.6968 on 670 training steps with a learning curve
    that climbs 0.7374 -> 0.8748 and never flattens, so 0.6968 was reported as
    a floor on the signal class rather than a ceiling. Pooling dev with the
    calibration split takes training from 670 to ~1,150 steps at no cost. If
    the curve is still rising there, "starved" is confirmed and the number to
    quote is a floor; if it flattens, 0.70 is close to what this signal gives.

    The pooled probe is NOT deployable in the gate. It has seen the calibration
    split, and a score selected on the data the conformal threshold is fitted
    to leaks into the guarantee. It is reported as a probe-quality result only,
    and `--gate-safe` is the variant chapter 7 may quote.

GAP 2 -- IS AUROC THE RIGHT TRAINING TARGET? (ch. 7.4)
    7.4 measured that two scores of equal AUROC are worth different amounts,
    because the propagation model only pays for the FIRST globally-wrong step
    in a solution, and the probe's score correlates +0.18 with step position
    while the composite's correlates -0.28. So the probe spends its ranking
    power on late steps that no repair can rescue.

    Three ways at it, in increasing order of how much data they need:

      position-residual  train on the global label, but project step position
                         out of the states first. Uses all 460 positives.
      select-on-firstbad train on the global label, choose layer and C by
                         first-bad recall instead of AUROC. All 460 positives,
                         but a thin selection signal.
      firstbad-target    train directly on "is this the first bad step".
                         Only 108 positives exist in the whole corpus.

    THE HONEST CONSTRAINT: 108 first-bad steps, 40 of them in test. The direct
    target is underpowered and its interval is reported rather than hidden. The
    first two variants are the ones this corpus can actually answer.

    python scripts/gpu_probe_states.py --save-states runs/probe_states.npz  # GPU, once
    python scripts/exp_probe_variants.py                                    # CPU, minutes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car.uncertainty.probe import DEFAULT_CS, auroc, fit_one, select_and_fit  # noqa: E402

STATES = Path("runs/probe_states.npz")
OUT = Path("runs/probe_variants.json")

# ch. 7 reference points, so every row below is comparable to something.
REFERENCE = {"token + semantic": 0.5742, "probe as published": 0.6968,
             "ch.5 task PRM scope": 0.9033}


def first_bad_recall(scores, first_bad, rate):
    """Share of first-bad steps caught in the top `rate` of the score.

    The quantity 7.4 shows the projection actually pays for. `rate` is fixed so
    the comparison is at matched verification budget rather than at whatever
    operating point each score happens to prefer.
    """
    first_bad = np.asarray(first_bad, dtype=bool)
    if not first_bad.any():
        return float("nan")
    k = max(1, int(round(len(scores) * rate)))
    flagged = np.argsort(-np.asarray(scores))[:k]
    return float(first_bad[flagged].sum() / first_bad.sum())


def residualise(X, position):
    """Project step position out of the states.

    Least-squares removal of the component of every hidden dimension that is
    linearly predictable from normalised position. If the probe's advantage is
    really positional bookkeeping rather than error detection, this is what
    takes it away -- and what is left is the part that cannot be.
    """
    p = np.asarray(position, dtype=float).reshape(-1, 1)
    A = np.hstack([p, np.ones_like(p)])
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return X - A @ coef


def select_by(states_by_layer, y, train_idx, select_idx, score_fn, Cs=DEFAULT_CS):
    """select_and_fit, but the selection criterion is a parameter.

    Same discipline: `score_fn` is evaluated on `select_idx` only, and test
    indices are not an argument here either.
    """
    best = None
    per_layer = {}
    for layer, X in sorted(states_by_layer.items()):
        X = np.asarray(X)
        for C in Cs:
            model, scaler = fit_one(X[train_idx], y[train_idx], C)
            s = model.decision_function(scaler.transform(X[select_idx]))
            v = score_fn(s, select_idx)
            if v != v:
                continue
            if layer not in per_layer or v > per_layer[layer]:
                per_layer[layer] = v
            if best is None or v > best[0]:
                best = (v, layer, C, model, scaler)
    if best is None:
        raise ValueError("no layer produced a usable probe")
    v, layer, C, model, scaler = best
    return {"layer": layer, "C": C, "criterion_select": v,
            "model": model, "scaler": scaler, "per_layer": per_layer}


def bootstrap_auroc(scores, y, sol_id, seed=0, n=2000):
    """Solution-clustered CI. Steps inside a solution are not independent, and
    a step-level bootstrap would report an interval several times too narrow."""
    rng = np.random.default_rng(seed)
    sols = np.unique(sol_id)
    idx_by_sol = [np.where(sol_id == s)[0] for s in sols]
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(sols), len(sols))
        idx = np.concatenate([idx_by_sol[p] for p in pick])
        yy = y[idx]
        if yy.all() or not yy.any():
            continue
        order = np.argsort(scores[idx])
        ranks = np.empty(len(idx), dtype=float)
        ranks[order] = np.arange(1, len(idx) + 1)
        n_pos, n_neg = yy.sum(), (~yy).sum()
        vals.append((ranks[yy].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def plain_auroc(scores, y):
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--states", type=Path, default=STATES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--select-frac", type=float, default=0.3)
    ap.add_argument("--budget-rate", type=float, default=0.191,
                    help="verification rate at which first-bad recall is "
                         "compared; the probe's measured rate in ch. 7.4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.states.exists():
        print(f"missing {args.states}\n"
              f"run: python scripts/gpu_probe_states.py --save-states {args.states}")
        return 1

    z = np.load(args.states, allow_pickle=False)
    S = z["states"].astype(np.float32)
    y = z["global_wrong"].astype(bool)
    fb = z["first_bad"].astype(bool)
    role = z["role"].astype(str)
    sol_id = z["solution_id"]
    position = z["position"]
    n_in_sol = z["n_steps_in_solution"]
    norm_pos = position / np.maximum(1, n_in_sol - 1)

    n_layers = S.shape[1]
    by_layer = {L: S[:, L, :] for L in range(n_layers)}
    by_layer_resid = None  # built lazily; residualising 29 layers is not free

    rng = np.random.default_rng(args.seed)
    dev = rng.permutation(np.where(role == "dev")[0])
    cal = np.where(role == "cal")[0]
    test_idx = np.where(role == "test")[0]

    n_sel = max(1, int(len(dev) * args.select_frac))
    sel_dev, train_dev = dev[:n_sel], dev[n_sel:]
    # Pooled: same selection slice, so only the TRAINING size changes between
    # the two. Anything else would confound size with selection.
    train_pool = np.concatenate([train_dev, cal])

    print("=" * 96)
    print("PROBE VARIANTS: is it starved, and is AUROC the right target?")
    print("=" * 96)
    print(f"states {S.shape}, {n_layers - 1} transformer layers + embeddings")
    print(f"steps {len(y)}   globally wrong {y.sum()} ({y.mean():.2%})   "
          f"first-bad {fb.sum()} ({fb.mean():.2%})")
    print(f"train(dev) {len(train_dev)} | train(dev+cal) {len(train_pool)} | "
          f"select {len(sel_dev)} | test {len(test_idx)}")
    print(f"test first-bad positives: {int(fb[test_idx].sum())}  "
          f"<- the ceiling on how sharp gap 2 can get here")
    print()

    rows = []

    def evaluate(name, res, gate_safe, note):
        X = res["_X"]
        s_test = res["model"].decision_function(res["scaler"].transform(X[test_idx]))
        a = plain_auroc(s_test, y[test_idx])
        lo, hi = bootstrap_auroc(s_test, y[test_idx], sol_id[test_idx], seed=args.seed)
        fbr = first_bad_recall(s_test, fb[test_idx], args.budget_rate)
        a_fb = plain_auroc(s_test, fb[test_idx])
        row = {"variant": name, "layer": int(res["layer"]), "C": float(res["C"]),
               "n_train": int(len(res["_train"])), "auroc_test": a,
               "auroc_ci": [lo, hi], "auroc_vs_firstbad": a_fb,
               "first_bad_recall": fbr, "gate_safe": gate_safe, "note": note,
               "score_position_corr": float(
                   np.corrcoef(s_test, norm_pos[test_idx])[0, 1])}
        rows.append(row)
        print(f"  {name:<26}{a:>8.4f}  [{lo:.3f},{hi:.3f}]"
              f"{a_fb:>9.4f}{fbr:>11.4f}{row['score_position_corr']:>+9.3f}"
              f"   {'yes' if gate_safe else 'NO':>3}")
        return row

    print(f"  {'variant':<26}{'AUROC':>8}  {'95% CI':^13}{'AUROCfb':>9}"
          f"{'1st-bad':>11}{'pos.corr':>9}   gate")
    print("  " + "-" * 92)

    # --- A. as published: global label, dev only. Reproduces 0.6968. --------
    a_res = select_and_fit(by_layer, y, train_dev, sel_dev)
    evaluate("A global, dev", {"layer": a_res.layer, "C": a_res.C,
                               "model": a_res.model, "scaler": a_res.scaler,
                               "_X": by_layer[a_res.layer], "_train": train_dev},
             True, "reproduces ch. 7.6")

    # --- B. GAP 1: same everything, more training data. --------------------
    b_res = select_and_fit(by_layer, y, train_pool, sel_dev)
    evaluate("B global, dev+cal", {"layer": b_res.layer, "C": b_res.C,
                                   "model": b_res.model, "scaler": b_res.scaler,
                                   "_X": by_layer[b_res.layer], "_train": train_pool},
             False, "saw the calibration split")

    # --- C. GAP 2, direct: train on the first-bad label. -------------------
    c_res = select_and_fit(by_layer, fb, train_pool, sel_dev)
    evaluate("C first-bad target", {"layer": c_res.layer, "C": c_res.C,
                                    "model": c_res.model, "scaler": c_res.scaler,
                                    "_X": by_layer[c_res.layer], "_train": train_pool},
             False, f"only {int(fb[train_pool].sum())} positives to train on")

    # --- D. GAP 2, mechanism: take step position out of the states. --------
    by_layer_resid = {L: residualise(X, norm_pos) for L, X in by_layer.items()}
    d_res = select_and_fit(by_layer_resid, y, train_pool, sel_dev)
    evaluate("D position-residualised", {"layer": d_res.layer, "C": d_res.C,
                                         "model": d_res.model, "scaler": d_res.scaler,
                                         "_X": by_layer_resid[d_res.layer],
                                         "_train": train_pool},
             False, "position projected out before fitting")

    # --- E. GAP 2, selection: same fit, chosen on the objective that pays. --
    def fb_criterion(scores, idx):
        return first_bad_recall(scores, fb[idx], args.budget_rate)

    e_res = select_by(by_layer, y, train_pool, sel_dev, fb_criterion)
    e_res["_X"] = by_layer[e_res["layer"]]
    e_res["_train"] = train_pool
    evaluate("E selected on 1st-bad", e_res, False,
             "layer and C chosen by first-bad recall, not AUROC")

    # --- learning curve on the pooled set, the direct answer to gap 1 ------
    print()
    print("  learning curve, global label, pooled training set:")
    curve = []
    X = by_layer[b_res.layer]
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        k = max(20, int(len(train_pool) * frac))
        sub = train_pool[:k]
        model, scaler = fit_one(X[sub], y[sub], b_res.C)
        a = auroc(model, scaler, X[test_idx], y[test_idx])
        curve.append((int(k), float(a)))
        print(f"    n={k:>5}   test AUROC {a:.4f}")
    slope = (curve[-1][1] - curve[-2][1]) / max(1, curve[-1][0] - curve[-2][0]) * 1000
    print(f"    slope over the last segment: {slope:+.4f} AUROC per 1000 steps")
    print(f"    -> {'STILL CLIMBING: 0.6968 is a floor' if slope > 0.005 else 'FLATTENING: near what this signal gives'}")

    print()
    print("  reference points:")
    for k, v in REFERENCE.items():
        print(f"    {k:<24}{v:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_steps": int(len(y)), "n_first_bad": int(fb.sum()),
        "n_first_bad_test": int(fb[test_idx].sum()),
        "budget_rate": args.budget_rate,
        "learning_curve_pooled": curve,
        "reference": REFERENCE,
        "variants": rows,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
