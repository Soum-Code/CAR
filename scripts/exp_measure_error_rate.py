"""Measure the real per-step error rate on GSM8K.

Every propagation result so far assumed `local_error_rate = 0.15`. This
measures it, on 60k+ real Mistral-7B-SFT solutions from Math-Shepherd, and in
doing so measures the local/global gap that `car.propagation` only simulated.

Two independent signals per step:
  local  -- does the inline <<expr=result>> arithmetic hold? Deterministic.
  global -- Math-Shepherd's +/- label, i.e. does this prefix still lead to the
            right answer? Inherits corruption from bad premises.

Run: python scripts/exp_measure_error_rate.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.math_shepherd import load_solutions  # noqa: E402
from car.propagation import final_error_probability  # noqa: E402

DATA = Path("data/raw/mathshepherd/strided.jsonl")

# Mistral-7B-SFT GSM8K accuracy as reported in arXiv:2312.08935. Used only for
# post-stratification, and the sensitivity to it is reported.
MODEL_ACCURACY = 0.45


def _rate(steps, pred):
    hits = sum(1 for s in steps if pred(s))
    return hits / max(1, len(steps))


def headline(sols):
    print("=" * 84)
    print("Per-step error rates on real GSM8K solutions (Mistral-7B-SFT)")
    print("=" * 84)
    print("STRATIFIED, because Math-Shepherd is a constructed PRM training set:")
    print("it samples multiple completions per problem and keeps a deliberate mix")
    print("of good and bad ones. Its raw +/- balance is a property of THEIR")
    print("construction, not of the model's natural error rate, so a blended")
    print("unconditional number would not mean what it appears to mean.\n")

    ok_sols = [s for s in sols if s.final_correct]
    bad_sols = [s for s in sols if not s.final_correct]

    def block(group, name):
        steps = [s for sol in group for s in sol.steps]
        checkable = [s for s in steps if s.checkable]
        if not steps:
            return None
        loc = _rate(checkable, lambda s: s.local_ok is False)
        glob = _rate(steps, lambda s: not s.global_ok)
        inh = _rate(steps, lambda s: s.inherited_corruption)
        print(f"{name:<28}{len(group):>9,}{len(steps):>10,}"
              f"{loc:>12.4f}{glob:>12.4f}{inh:>12.4f}")
        return {"local": loc, "global": glob, "inherited": inh, "n": len(steps)}

    print(f"{'stratum':<28}{'solutions':>9}{'steps':>10}"
          f"{'local err':>12}{'global err':>12}{'inherited':>12}")
    print("-" * 84)
    good = block(ok_sols, "final answer CORRECT")
    bad = block(bad_sols, "final answer WRONG")
    allr = block(sols, "raw sample (biased mix)")

    print()
    print(f"sample solution-level accuracy   {len(ok_sols) / len(sols):.1%}")
    print(f"reported model accuracy          ~41-52% (arXiv:2312.08935)")
    print()

    # Post-stratify to the model's reported accuracy.
    print("Post-stratified to the reported model accuracy:")
    print(f"{'assumed acc':<14}{'local err':>12}{'global err':>12}")
    print("-" * 84)
    est = None
    for acc in (0.40, MODEL_ACCURACY, 0.52):
        loc = acc * good["local"] + (1 - acc) * bad["local"]
        glo = acc * good["global"] + (1 - acc) * bad["global"]
        if abs(acc - MODEL_ACCURACY) < 1e-9:
            est = {"local": loc, "global": glo}
        print(f"{acc:<14.0%}{loc:>12.4f}{glo:>12.4f}")
    print()
    print("The local rate is fairly insensitive to the assumed accuracy; the")
    print("global rate is not, because global correctness is almost definitional")
    print("for the two strata.")
    print()
    return {"good": good, "bad": bad, "raw": allr, "stratified": est}


def by_position(sols):
    print("=" * 84)
    print("Is the error rate position-dependent?")
    print("=" * 84)
    print("If early steps were intrinsically harder, front-loaded verification")
    print("would gain -- the one untested escape for influence weighting.\n")

    loc_bad, loc_tot = Counter(), Counter()
    glob_bad, glob_tot = Counter(), Counter()
    # Local arithmetic validity is deterministic and does not depend on the
    # +/- label mix, so position effects here are robust to the sampling bias.
    for sol in sols:
        for s in sol.steps:
            glob_tot[s.index] += 1
            glob_bad[s.index] += 0 if s.global_ok else 1
            if s.checkable:
                loc_tot[s.index] += 1
                loc_bad[s.index] += 1 if s.local_ok is False else 0

    print(f"{'position':<10}{'n':>9}{'local err':>12}{'global err':>12}")
    print("-" * 84)
    for i in sorted(loc_tot):
        if loc_tot[i] < 300:
            continue
        print(f"{i + 1:<10}{loc_tot[i]:>9,}{loc_bad[i] / loc_tot[i]:>12.4f}"
              f"{glob_bad[i] / glob_tot[i]:>12.4f}")

    xs = [i for i in sorted(loc_tot) if loc_tot[i] >= 300]
    ys = [loc_bad[i] / loc_tot[i] for i in xs]
    if len(xs) > 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        print(f"\ncorr(position, local error rate) = {r:+.3f}")
        if r > 0.3:
            print("  -> LATER steps are harder; front-loading is not favoured")
        elif r < -0.3:
            print("  -> EARLIER steps are harder; front-loading gains here")
        else:
            print("  -> roughly flat; the i.i.d. assumption in the model holds")
    print()


def propagation_evidence(sols):
    print("=" * 84)
    print("Does corruption actually propagate?")
    print("=" * 84)
    # Once a step is labelled '-', what happens downstream?
    after_bad_total = after_bad_still_bad = 0
    recovered = 0
    for sol in sols:
        fb = sol.first_bad_index
        if fb is None:
            continue
        tail = [s for s in sol.steps if s.index > fb]
        if not tail:
            continue
        after_bad_total += len(tail)
        after_bad_still_bad += sum(1 for s in tail if not s.global_ok)
        if all(s.global_ok for s in tail):
            recovered += 1

    print(f"steps after the first bad step        {after_bad_total:,}")
    print(f"  of which still labelled '-'         {after_bad_still_bad / max(1, after_bad_total):.1%}")
    print(f"solutions that fully recovered        {recovered:,}")
    print()

    # Of steps downstream of a LOCAL error, how many are locally fine but
    # globally bad -- the precise propagation signature.
    inh_tot = inh_bad = 0
    for sol in sols:
        fl = sol.first_local_error_index
        if fl is None:
            continue
        for s in sol.steps:
            if s.index > fl and s.local_ok is True:
                inh_tot += 1
                inh_bad += 0 if s.global_ok else 1
    print(f"locally-VALID steps downstream of a local error   {inh_tot:,}")
    print(f"  of which globally wrong (inherited)             {inh_bad / max(1, inh_tot):.1%}")
    print()
    print("That last number is the propagation effect, measured rather than assumed:")
    print("steps that are arithmetically perfect and still wrong because a premise was.")
    print()


def solution_level(sols):
    print("=" * 84)
    print("Solution-level rates, and the Kotte feasibility test")
    print("=" * 84)
    n = len(sols)
    final_bad = sum(1 for s in sols if not s.final_correct)
    any_local = sum(1 for s in sols if s.first_local_error_index is not None)
    lens = Counter(len(s.steps) for s in sols)

    print(f"solutions                         {n:,}")
    print(f"final answer wrong (sample)       {final_bad / n:.1%}  "
          f"<- sample composition, not model accuracy")
    print(f"contains >=1 arithmetic error     {any_local / n:.1%}")
    print(f"steps per solution (mean)         {np.mean([len(s.steps) for s in sols]):.2f}")
    print(f"length distribution               {dict(sorted(lens.items())[:9])}")
    print()

    # Kotte (arXiv:2606.29054) Prop. 3: when base risk mu > alpha, any
    # distribution-free method must abstain on >= (mu-alpha)/(1-alpha).
    #
    # mu must be post-stratified. The raw sample rate reflects Math-Shepherd's
    # construction, not the model, and feeding that number in would overstate
    # the floor by ~15 percentage points.
    good = [s for sol in sols if sol.final_correct for s in sol.steps]
    bad = [s for sol in sols if not sol.final_correct for s in sol.steps]
    mu_good = sum(1 for s in good if not s.global_ok) / max(1, len(good))
    mu_bad = sum(1 for s in bad if not s.global_ok) / max(1, len(bad))
    raw_mu = (len(good) * mu_good + len(bad) * mu_bad) / max(1, len(good) + len(bad))

    print(f"mu | correct-answer solutions   {mu_good:.4f}")
    print(f"mu | wrong-answer solutions     {mu_bad:.4f}")
    print(f"raw sample mu                   {raw_mu:.4f}  <- biased, do not use")
    print()
    print(f"{'assumed acc':<13}{'mu':>8}" + "".join(
        f"{f'a={a:.2f}':>10}" for a in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)))
    print("-" * 84)
    est_mu = None
    for acc in (0.40, MODEL_ACCURACY, 0.52):
        mu = acc * mu_good + (1 - acc) * mu_bad
        if abs(acc - MODEL_ACCURACY) < 1e-9:
            est_mu = mu
        cells = ""
        for alpha in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
            cells += "     ok   " if mu <= alpha else f"{(mu - alpha) / (1 - alpha):>9.1%} "
        print(f"{acc:<13.0%}{mu:>8.4f}{cells}")
    print()
    print("Cells show the MINIMUM fraction of steps any distribution-free method")
    print("must verify or abstain on to hit that alpha. 'ok' means mu <= alpha, so")
    print("the target is attainable without a floor.")
    print()
    return est_mu


def recalibrate_model(local_rate, sols):
    print("=" * 84)
    print("Re-running the propagation model with the MEASURED rate")
    print("=" * 84)
    depth = int(round(np.mean([len(s.steps) for s in sols])))
    print(f"chain length {depth} (mean solution length), "
          f"local error {local_rate:.4f} vs the 0.15 assumed previously\n")
    print(f"{'budget':<10}{'assumed 0.15':>16}{'measured':>14}{'delta':>10}")
    print("-" * 84)
    for v in (0.0, 0.25, 0.5, 0.75):
        a = final_error_probability(depth, v, 1.0, 0.15)
        b = final_error_probability(depth, v, 1.0, local_rate)
        print(f"{v:<10.2f}{a:>16.4f}{b:>14.4f}{b - a:>10.4f}")
    print()


def main():
    if not DATA.exists():
        print(f"missing {DATA}")
        return
    print("loading Math-Shepherd GSM8K sample...")
    sols = load_solutions(DATA)
    print(f"parsed {len(sols):,} solutions\n")

    rates = headline(sols)
    by_position(sols)
    propagation_evidence(sols)
    solution_level(sols)
    recalibrate_model(rates["stratified"]["local"], sols)


if __name__ == "__main__":
    main()
