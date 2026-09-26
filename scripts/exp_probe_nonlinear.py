"""Is 0.6968 a property of the states, or of logistic regression? [CPU]

Chapter 7.6 settles the two variables it could. Training data is a small lever
(+0.0105 for a doubling, CI spanning zero), and the training target matters a
lot (first-bad recall 0.30 -> 0.45). What it never varied is the **instrument**:
every probe in this thesis is a logistic regression on one layer. ReProbe's are
not linear, and the prediction that a probe here would reach ~0.9033 was made
about their setting.

So this asks the last question in the series: does a non-linear head on the
same frozen states do better than a linear one?

DESIGN, AND WHY IT IS NOT A FRESH 145-CONFIGURATION SEARCH

The layer is chosen ONCE, by the linear probe on the selection split, and both
instruments are then compared on the layers it liked. Re-running a full layer
sweep for the MLP would make the comparison "best of 145 linear vs best of many
more non-linear", and the winner's curse would do the work. §7.6 already paid
0.18 AUROC for that lesson once.

  stage 1   linear probe per layer, selection split      -> a layer ranking
  stage 2   top-k layers x a small MLP grid, selection   -> one non-linear probe
  stage 3   both scored on test, paired bootstrap on the difference

Test indices are not an argument to any selection function here, same as
`select_and_fit`. Both instruments are run at both training sizes, because §7.6
showed size is worth about +0.01 and confounding it with the instrument is
exactly the mistake that produced the first wrong headline in that section.

    python scripts/exp_probe_nonlinear.py          # needs runs/probe_states.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from car.uncertainty.probe import DEFAULT_CS, fit_one  # noqa: E402
from exp_probe_variants import (  # noqa: E402
    first_bad_recall,
    paired_bootstrap,
    plain_auroc,
)

STATES = Path("runs/probe_states.npz")
OUT = Path("runs/probe_nonlinear.json")

# Small on purpose. 670 training rows against 3584 input dimensions is a regime
# where capacity buys overfitting, so the grid explores regularisation more
# than width.
HIDDEN = ((16,), (32,), (64,), (32, 16))
ALPHAS = (1e-2, 1e-1, 1.0, 10.0)


def fit_mlp(X_train, y_train, hidden, alpha, seed=0):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=hidden, alpha=alpha, max_iter=600,
        early_stopping=True, n_iter_no_change=20, validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(scaler.transform(X_train), y_train)
    return model, scaler


def score_of(model, scaler, X):
    """Decision-function-like score, whichever the estimator offers."""
    Z = scaler.transform(X)
    if hasattr(model, "decision_function"):
        return model.decision_function(Z)
    return model.predict_proba(Z)[:, 1]


def _best_linear_at(X, y, train_idx, sel_idx):
    """Best C at a FIXED layer, chosen on the selection split."""
    best = None
    for C in DEFAULT_CS:
        m, s = fit_one(X[train_idx], y[train_idx], C)
        a = plain_auroc(score_of(m, s, X[sel_idx]), y[sel_idx])
        if a != a:
            continue
        if best is None or a > best[0]:
            best = (a, m, s)
    return (best[1], best[2]) if best else (None, None)


def _best_mlp_at(X, y, train_idx, sel_idx, seed=0):
    """Best (hidden, alpha) at a FIXED layer, chosen on the selection split."""
    best = None
    for hidden in HIDDEN:
        for alpha in ALPHAS:
            try:
                m, s = fit_mlp(X[train_idx], y[train_idx], hidden, alpha, seed=seed)
            except Exception:
                continue
            a = plain_auroc(score_of(m, s, X[sel_idx]), y[sel_idx])
            if a != a:
                continue
            if best is None or a > best[0]:
                best = (a, m, s)
    return (best[1], best[2]) if best else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--states", type=Path, default=STATES)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--top-layers", type=int, default=5,
                    help="how many of the linear probe's best layers the MLP "
                         "may search over; kept small to bound the curse")
    ap.add_argument("--select-frac", type=float, default=0.3)
    ap.add_argument("--budget-rate", type=float, default=0.191)
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
    by_layer = {L: S[:, L, :] for L in range(S.shape[1])}

    rng = np.random.default_rng(args.seed)
    dev = rng.permutation(np.where(role == "dev")[0])
    cal = np.where(role == "cal")[0]
    test = np.where(role == "test")[0]
    n_sel = max(1, int(len(dev) * args.select_frac))
    sel, train_dev = dev[:n_sel], dev[n_sel:]
    train_pool = rng.permutation(np.concatenate([train_dev, cal]))

    print("=" * 92)
    print("NON-LINEAR PROBE: is 0.6968 the states, or the logistic regression?")
    print("=" * 92)
    print(f"states {S.shape}   train(dev) {len(train_dev)} | "
          f"train(dev+cal) {len(train_pool)} | select {len(sel)} | test {len(test)}")

    results = {}
    for tag, train_idx in (("dev", train_dev), ("pooled", train_pool)):
        print()
        print(f"--- training set: {tag} ({len(train_idx)} steps) "
              f"{'[gate-safe]' if tag == 'dev' else '[saw calibration]'}")

        # stage 1 -- the linear baseline, and the layer ranking it implies
        lin_best, per_layer = None, {}
        for L, X in sorted(by_layer.items()):
            for C in DEFAULT_CS:
                m, s = fit_one(X[train_idx], y[train_idx], C)
                a = plain_auroc(score_of(m, s, X[sel]), y[sel])
                if a != a:
                    continue
                if L not in per_layer or a > per_layer[L]:
                    per_layer[L] = a
                if lin_best is None or a > lin_best[0]:
                    lin_best = (a, L, C, m, s)
        _, lin_L, lin_C, lin_m, lin_s = lin_best
        lin_score = score_of(lin_m, lin_s, by_layer[lin_L])
        lin_test = plain_auroc(lin_score[test], y[test])
        print(f"  linear     layer {lin_L:>2} C={lin_C:<7} "
              f"select {lin_best[0]:.4f}  test {lin_test:.4f}")

        # stage 2 -- MLP, restricted to the layers the linear probe ranked top
        top = [L for L, _ in sorted(per_layer.items(), key=lambda kv: -kv[1])
               ][: args.top_layers]
        mlp_best = None
        for L in top:
            X = by_layer[L]
            for hidden in HIDDEN:
                for alpha in ALPHAS:
                    try:
                        m, s = fit_mlp(X[train_idx], y[train_idx], hidden, alpha,
                                       seed=args.seed)
                    except Exception:
                        continue
                    a = plain_auroc(score_of(m, s, X[sel]), y[sel])
                    if a != a:
                        continue
                    if mlp_best is None or a > mlp_best[0]:
                        mlp_best = (a, L, hidden, alpha, m, s)
        if mlp_best is None:
            print("  no MLP converged")
            continue
        _, mL, mh, ma, mm, ms = mlp_best
        mlp_score = score_of(mm, ms, by_layer[mL])
        mlp_test = plain_auroc(mlp_score[test], y[test])
        print(f"  non-linear layer {mL:>2} hidden={str(mh):<10} alpha={ma:<6} "
              f"select {mlp_best[0]:.4f}  test {mlp_test:.4f}")

        # stage 3a -- MATCHED-LAYER comparison, which is the only one that
        # isolates the instrument. An earlier version of this script compared
        # each arm's own argmax, so its "instrument effect" was instrument plus
        # layer re-selection -- the identical confound `fixed_config_doubling`
        # exists to prevent for the data lever, applied there and forgotten
        # here. The sign flips between layers, which is the finding.
        matched = {}
        for where, L in (("linear_layer", lin_L), ("mlp_layer", mL)):
            lin_m2, lin_s2 = _best_linear_at(by_layer[L], y, train_idx, sel)
            mlp_m2, mlp_s2 = _best_mlp_at(by_layer[L], y, train_idx, sel,
                                          seed=args.seed)
            if mlp_m2 is None:
                continue
            a = score_of(lin_m2, lin_s2, by_layer[L])
            b = score_of(mlp_m2, mlp_s2, by_layer[L])
            d = paired_bootstrap(plain_auroc, a[test], b[test], y[test],
                                 sol_id[test], seed=args.seed)
            matched[where] = {
                "layer": int(L),
                "auroc_linear": plain_auroc(a[test], y[test]),
                "auroc_nonlinear": plain_auroc(b[test], y[test]),
                "delta": d,
            }
            print(f"  MATCHED at layer {L:>2} ({where}): linear "
                  f"{matched[where]['auroc_linear']:.4f}  MLP "
                  f"{matched[where]['auroc_nonlinear']:.4f}  "
                  f"{d['mean']:+.4f} CI [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]")

        # stage 3b -- what the selection split can actually resolve. If a
        # 0.001 selection margin moves test AUROC by 0.07, no instrument
        # effect of this size is identifiable and the layer argmax is noise.
        layer_rows = []
        for L in top:
            m2, s2 = _best_linear_at(by_layer[L], y, train_idx, sel)
            sc = score_of(m2, s2, by_layer[L])
            layer_rows.append((int(L), plain_auroc(sc[sel], y[sel]),
                               plain_auroc(sc[test], y[test])))
        layer_rows.sort(key=lambda r: -r[1])
        sel_spread = layer_rows[0][1] - layer_rows[-1][1]
        test_spread = max(r[2] for r in layer_rows) - min(r[2] for r in layer_rows)
        print(f"  layer resolution over the top {len(layer_rows)}: selection spread "
              f"{sel_spread:.4f} vs TEST spread {test_spread:.4f}")

        # stage 3c -- the published comparison, kept so the confound is visible
        delta = paired_bootstrap(plain_auroc, lin_score[test], mlp_score[test],
                                 y[test], sol_id[test], seed=args.seed)
        print(f"  non-linear MINUS linear: {delta['mean']:+.4f}  95% CI "
              f"[{delta['ci'][0]:+.4f},{delta['ci'][1]:+.4f}]  "
              f"P={delta['p_better']:.2f}  "
              f"-- {'excludes zero' if delta['ci'][0] > 0 else 'SPANS ZERO'}")
        # The gap each instrument pays for being chosen on the selection split.
        print(f"  winner's curse  linear {lin_best[0] - lin_test:+.4f}   "
              f"non-linear {mlp_best[0] - mlp_test:+.4f}")

        results[tag] = {
            "n_train": int(len(train_idx)),
            "linear": {"layer": int(lin_L), "C": float(lin_C),
                       "auroc_select": float(lin_best[0]),
                       "auroc_test": float(lin_test),
                       "first_bad_recall": first_bad_recall(
                           lin_score[test], fb[test], args.budget_rate)},
            "nonlinear": {"layer": int(mL), "hidden": list(mh),
                          "alpha": float(ma),
                          "auroc_select": float(mlp_best[0]),
                          "auroc_test": float(mlp_test),
                          "first_bad_recall": first_bad_recall(
                              mlp_score[test], fb[test], args.budget_rate)},
            "delta_auroc": delta,
            "layers_searched": [int(L) for L in top],
            "n_configurations": len(top) * len(HIDDEN) * len(ALPHAS),
            "matched_layer": matched,
            "layer_resolution": {
                "rows": layer_rows,
                "selection_spread": float(sel_spread),
                "test_spread": float(test_spread),
            },
        }

    print()
    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    gs = results.get("dev")
    if gs:
        # Keyed off the MATCHED comparison. The argmax-vs-argmax delta is kept
        # in the artifact but is not the instrument effect: it carries layer
        # re-selection, and reading a verdict off it is how this script's first
        # version produced a wrong headline.
        signs = []
        for tag, v in results.items():
            for where, m in v.get("matched_layer", {}).items():
                signs.append((tag, where, m["layer"], m["delta"]))
        pos = [s for s in signs if s[3]["ci"][0] > 0]
        neg = [s for s in signs if s[3]["ci"][1] < 0]
        print("Matched-layer instrument effects, every layer tested:")
        for tag, where, L, d in signs:
            verdict = ("favours MLP" if d["ci"][0] > 0
                       else "favours LINEAR" if d["ci"][1] < 0 else "spans zero")
            print(f"  {tag:<7} layer {L:>2}  {d['mean']:+.4f} "
                  f"CI [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}]  {verdict}")

        means = [s[3]["mean"] for s in signs]
        mixed_sign = min(means) < 0 < max(means)
        res = max((v["layer_resolution"] for v in results.values()),
                  key=lambda r: r["test_spread"])
        print()
        if pos and neg:
            print("THE SIGN FLIPS AND BOTH DIRECTIONS ARE SIGNIFICANT. The")
            print("instrument effect is not identifiable on this corpus.")
        elif mixed_sign or (pos and neg):
            # The case this corpus is actually in. Not "linear wins" -- most
            # comparisons span zero -- and not "MLP wins" either.
            print("NO IDENTIFIABLE INSTRUMENT EFFECT. The matched-layer point")
            print(f"estimates run {min(means):+.4f} to {max(means):+.4f}: the sign")
            print("depends on which layer is held fixed. "
                  f"{len(signs) - len(pos) - len(neg)} of {len(signs)} intervals")
            print("span zero" + (", and the one that does not favours the LINEAR"
                                 " probe." if neg else "."))
        elif pos:
            print("A non-linear head wins at every layer tested. That is an")
            print("instrument effect and ch. 7.6 should say so.")
        elif neg:
            print("The linear probe wins at every layer tested.")
        else:
            print("No matched-layer comparison separates the two instruments.")

        print()
        print("WHY: layer choice is not resolvable at this sample size.")
        print(f"  the selection split separates the top layers by "
              f"{res['selection_spread']:.4f}")
        print(f"  their TEST AUROCs differ by                    "
              f"{res['test_spread']:.4f}  ({res['test_spread'] / max(res['selection_spread'], 1e-9):.1f}x)")
        print("So the argmax-vs-argmax comparison is dominated by which layer")
        print("each arm happened to land on, not by the instrument.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "hidden_grid": [list(h) for h in HIDDEN],
        "alpha_grid": list(ALPHAS),
        "budget_rate": args.budget_rate,
        "results": results,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
