"""Head-to-head: StrategyQA vs GSM8K as the primary propagation benchmark.

StrategyQA was found to have almost no propagation headroom (72.9% of questions
are one hop deep; 11.2% of steps have any non-terminal descendant). This checks
whether GSM8K is actually better on the dimensions that matter, using
dependency graphs derived from its inline calculator annotations.

Run: python scripts/exp_benchmark_compare.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data import gsm8k, strategyqa  # noqa: E402
from car.data.strategyqa import corpus_summary  # noqa: E402
from car.topology import SCHEDULES, DAGPropagation  # noqa: E402

SQA = Path("data/raw/strategyqa/strategyqa_train.json")
GSM = Path("data/raw/gsm8k/train.jsonl")
LOCAL_ERR = 0.15
POLICIES = ["uniform", "front", "back", "influence", "depth", "cut"]


def headroom(dags) -> float:
    tot = inter = 0
    for d in dags:
        for v in range(d.n):
            tot += 1
            if d.descendants(v) - {d.terminal}:
                inter += 1
    return inter / max(1, tot)


def evaluate(dags, policy, budget_frac, scope, decay, seed=0, trials=200):
    rng = np.random.default_rng(seed)
    errs = []
    for dag in dags:
        sched = SCHEDULES[policy](dag, budget_frac * dag.n)
        sim = DAGPropagation(
            dag, local_error_rate=LOCAL_ERR, verifier_scope=scope, scope_decay=decay
        )
        for _ in range(trials):
            errs.append(not sim.run(sched, rng, budget=dag.n).terminal_correct)
    return float(np.mean(errs))


def compare_structure(sqa_dags, gsm_dags):
    print("=" * 84)
    print("Structural comparison")
    print("=" * 84)
    a, b = corpus_summary(sqa_dags), corpus_summary(gsm_dags)
    rows = [
        ("questions", a["n_questions"], b["n_questions"]),
        ("steps per item (mean)", f"{a['steps']['mean']:.2f}", f"{b['steps']['mean']:.2f}"),
        ("longest path (mean)", f"{a['longest_path']['mean']:.2f}",
         f"{b['longest_path']['mean']:.2f}"),
        ("max depth observed", max(a["longest_path"]["dist"]), max(b["longest_path"]["dist"])),
        ("steps w/ non-terminal desc.", f"{headroom(sqa_dags):.1%}", f"{headroom(gsm_dags):.1%}"),
        ("corr(position, influence)", f"{a['mean_corr_pos_influence']:.3f}",
         f"{b['mean_corr_pos_influence']:.3f}"),
    ]
    print(f"{'property':<32}{'StrategyQA':>16}{'GSM8K':>16}")
    print("-" * 84)
    for name, x, y in rows:
        print(f"{name:<32}{str(x):>16}{str(y):>16}")
    print()
    print(f"{'shapes':<32}")
    for k in sorted(set(a["shapes"]) | set(b["shapes"])):
        sa = a["shapes"].get(k, 0) / a["n_questions"]
        sb = b["shapes"].get(k, 0) / b["n_questions"]
        print(f"  {k:<30}{sa:>15.1%}{sb:>16.1%}")
    print()


def compare_allocation(sqa_dags, gsm_dags, budget_frac=0.375, scope=1.0, decay=1.0):
    print("=" * 84)
    print("Allocation policy spread -- does the choice matter on each benchmark?")
    print("=" * 84)
    print(f"budget {budget_frac:.1%}, scope={scope}, decay={decay}\n")
    print(f"{'policy':<14}{'StrategyQA':>14}{'GSM8K':>14}")
    print("-" * 84)
    res = {}
    for p in POLICIES:
        x = evaluate(sqa_dags, p, budget_frac, scope, decay)
        y = evaluate(gsm_dags, p, budget_frac, scope, decay)
        res[p] = (x, y)
        print(f"{p:<14}{x:>14.4f}{y:>14.4f}")
    sx = max(v[0] for v in res.values()) - min(v[0] for v in res.values())
    sy = max(v[1] for v in res.values()) - min(v[1] for v in res.values())
    bx = min(res, key=lambda k: res[k][0])
    by = min(res, key=lambda k: res[k][1])
    print("-" * 84)
    print(f"{'spread':<14}{sx:>14.4f}{sy:>14.4f}")
    print(f"{'best':<14}{bx:>14}{by:>14}")
    print()


def gsm_by_depth(gsm_dags, budget_frac=0.375):
    print("=" * 84)
    print("GSM8K: policy spread by reasoning depth")
    print("=" * 84)
    print("depth = longest directed path; this is what propagation needs\n")
    from car.data.strategyqa import dag_stats

    buckets: dict[int, list] = {}
    for d in gsm_dags:
        buckets.setdefault(dag_stats(d)["longest_path"], []).append(d)

    print(f"{'depth':<8}{'count':>8}" + "".join(f"{p:>11}" for p in POLICIES) + f"{'spread':>10}")
    print("-" * 84)
    for depth in sorted(buckets):
        grp = buckets[depth]
        if len(grp) < 30:
            continue
        vals = {p: evaluate(grp, p, budget_frac, 1.0, 1.0, trials=150) for p in POLICIES}
        spread = max(vals.values()) - min(vals.values())
        print(f"{depth:<8}{len(grp):>8}"
              + "".join(f"{vals[p]:>11.4f}" for p in POLICIES) + f"{spread:>10.4f}")
    print()


def main():
    if not (SQA.exists() and GSM.exists()):
        print("missing datasets; run the downloads first")
        return

    print("GSM8K extraction quality")
    st = gsm8k.extraction_stats(GSM)
    print(f"  rows                {st['rows']}")
    print(f"  usable (>=2 steps)  {st['usable']}  ({st['usable'] / st['rows']:.1%})")
    print(f"  operand link rate   {st['link_rate']:.1%}  (rest are givens from the problem)")
    print(f"  ambiguous links     {st['ambiguous_rate']:.1%}  (operand matches a result AND a question number)")
    print(f"  orphan steps        {st['orphan_step_rate']:.1%}  (recompute from givens = parallel branch)")
    print()

    sqa_dags = strategyqa.load_dags(SQA)
    gsm_dags = gsm8k.load_dags(GSM)

    compare_structure(sqa_dags, gsm_dags)
    compare_allocation(sqa_dags, gsm_dags)
    gsm_by_depth(gsm_dags)


if __name__ == "__main__":
    main()
