"""Reproduces the motivating result: adaptive calibration collapses under
censored feedback, and forced exploration fixes it.

Run:  python scripts/demo_censored_feedback.py

This is Figure 1 of the paper, in table form. It needs no GPU and no dataset --
the accept-region risk has a closed form, so the correct answer is known and
each update rule can be scored against it rather than against a guess.

Setup
-----
Scores are uniform on [0, 1] with P(wrong | score) = score. The gate accepts
when score <= q, so:

    accept-region risk = E[wrong | score <= q] = q / 2   ->  q* = 2 * alpha
    reject-region risk = E[wrong | score >  q] = (1 + q) / 2

At alpha = 0.1 the target is q* = 0.2, where the accept region carries 10%
error and the reject region carries 60%. A calibrator that consumes whichever
labels happen to arrive is therefore reading a number six times too large.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.conformal import AdaptiveCalibrator  # noqa: E402

ALPHA = 0.1
Q_STAR = 2 * ALPHA


def run(update_mode: str, *, epsilon=0.2, gamma=0.005, T=40000, seed=0):
    rng = np.random.default_rng(seed)
    cal = AdaptiveCalibrator(
        alpha=ALPHA,
        gamma=gamma,
        epsilon=epsilon,
        update_mode=update_mode,
        clip=(0.01, 1.0),
        seed=seed,
    )
    cal.fit(np.array([0.9]), None)
    cal._threshold = 0.9

    traj = np.empty(T)
    verify_calls = 0
    explore_calls = 0
    accepted_wrong = 0
    accepted_total = 0

    for t in range(T):
        s = float(rng.random())
        wrong = int(rng.random() < s)
        explore = cal.should_explore()
        accepted = s <= cal.threshold

        if accepted:
            accepted_total += 1
            accepted_wrong += wrong

        if (not accepted) or explore:
            verify_calls += 1
            explore_calls += int(explore)
            cal.update(wrong, was_exploration=explore, was_accepted=accepted)

        traj[t] = cal.threshold

    tail = traj[int(0.75 * T) :]
    return {
        "mode": update_mode,
        "threshold": float(tail.mean()),
        "std": float(tail.std()),
        "gap": abs(float(tail.mean()) - Q_STAR),
        "realised_risk": accepted_wrong / accepted_total if accepted_total else float("nan"),
        "verify_rate": verify_calls / T,
        "explore_rate": explore_calls / T,
    }


def main() -> None:
    print(f"\ntarget alpha = {ALPHA}   analytic optimum q* = {Q_STAR}\n")
    header = (
        f"{'update rule':<20}{'threshold':>11}{'gap':>9}{'std':>9}"
        f"{'risk':>9}{'verify%':>10}{'explore%':>10}"
    )
    print(header)
    print("-" * len(header))

    for mode in ("naive", "exploration_only", "ipw"):
        r = run(mode)
        print(
            f"{r['mode']:<20}{r['threshold']:>11.4f}{r['gap']:>9.4f}{r['std']:>9.4f}"
            f"{r['realised_risk']:>9.4f}{r['verify_rate'] * 100:>9.1f}%"
            f"{r['explore_rate'] * 100:>9.1f}%"
        )

    print("\nconvergence by horizon (threshold, tail mean):")
    print(f"{'T':>8}" + "".join(f"{m:>20}" for m in ("naive", "exploration_only", "ipw")))
    for T in (2000, 5000, 20000, 40000):
        row = "".join(f"{run(m, T=T)['threshold']:>20.4f}" for m in
                      ("naive", "exploration_only", "ipw"))
        print(f"{T:>8}" + row)

    print(
        "\nReading:\n"
        "  naive             pins to the clip floor at every horizon. It is fed\n"
        "                    reject-region error (~0.60) against a target of 0.10,\n"
        "                    so it verifies everything. More data does not help.\n"
        "  exploration_only  unbiased and stable, converges slowly from above.\n"
        "  ipw               same labels, scaled by 1/epsilon. Roughly 1/epsilon\n"
        "                    times the drift, so it arrives sooner, and roughly\n"
        "                    1/epsilon^2 the noise, so it sits wider around q*.\n"
    )


if __name__ == "__main__":
    main()
