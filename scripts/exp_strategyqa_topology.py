"""Allocation policies on REAL StrategyQA dependency graphs.

The synthetic topology experiments used hand-built shapes. This runs the same
comparison on the 2272 human-annotated dependency graphs extracted from
StrategyQA decompositions, which is the difference between a synthetic argument
and an empirical one.

Run: python scripts/exp_strategyqa_topology.py
Requires: data/raw/strategyqa/strategyqa_train.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.strategyqa import (  # noqa: E402
    classify_shape,
    corpus_summary,
    dag_stats,
    load_dags,
)
from car.topology import SCHEDULES, DAGPropagation  # noqa: E402

DATA = Path("data/raw/strategyqa/strategyqa_train.json")
LOCAL_ERR = 0.15
TRIALS_PER_DAG = 400
POLICIES = ["uniform", "front", "back", "influence", "depth", "cut"]


def evaluate_corpus(dags, policy, budget_frac, scope, decay, seed=0, trials=TRIALS_PER_DAG):
    """Mean final-answer error across the corpus under one policy."""
    rng = np.random.default_rng(seed)
    errs, calls = [], []
    for dag in dags:
        sched = SCHEDULES[policy](dag, budget_frac * dag.n)
        sim = DAGPropagation(
            dag,
            local_error_rate=LOCAL_ERR,
            verifier_scope=scope,
            scope_decay=decay,
        )
        for _ in range(trials):
            r = sim.run(sched, rng, budget=dag.n)
            errs.append(not r.terminal_correct)
            calls.append(r.verification_calls)
    return float(np.mean(errs)), float(np.mean(calls))


def structure_report(dags):
    print("=" * 84)
    print("Real StrategyQA dependency graphs")
    print("=" * 84)
    s = corpus_summary(dags)
    print(f"questions              {s['n_questions']}")
    print(f"shapes                 {s['shapes']}")
    print(f"steps per question     mean {s['steps']['mean']:.2f}   {s['steps']['dist']}")
    print(f"longest path (depth)   mean {s['longest_path']['mean']:.2f}   "
          f"{s['longest_path']['dist']}")
    print(f"root premises          mean {s['roots']['mean']:.2f}   {s['roots']['dist']}")
    print(f"max fan-in             {s['max_fan_in']}")
    print(f"dead-end steps         {s['dead_ends']}")
    print(f"corr(position, influence)  {s['mean_corr_pos_influence']:.3f}")
    print()

    print("Most common exact structures (parents tuple):")
    sig = Counter(
        (d.n, tuple(d.parents)) for d in dags
    )
    for (n, parents), c in sig.most_common(6):
        desc = " ; ".join(
            f"[{i+1}]<-{sorted(p+1 for p in ps) if ps else 'root'}"
            for i, ps in enumerate(parents)
        )
        print(f"  {c:>5}  n={n}  {desc}")
    print()


def allocation_report(dags, scope, decay, label, budget_frac=0.375):
    print("=" * 84)
    print(f"Allocation on real graphs -- {label}")
    print("=" * 84)
    print(f"budget {budget_frac:.1%} of nodes, scope={scope}, decay={decay}\n")
    print(f"{'policy':<14}{'final error':>14}{'calls/question':>17}{'vs uniform':>13}")
    print("-" * 84)
    base = None
    results = {}
    for p in POLICIES:
        e, c = evaluate_corpus(dags, p, budget_frac, scope, decay)
        results[p] = e
        if p == "uniform":
            base = e
        print(f"{p:<14}{e:>14.4f}{c:>17.3f}{(e - base) if base is not None else 0:>13.4f}")
    best = min(results, key=results.get)
    spread = max(results.values()) - min(results.values())
    print(f"\nbest: {best}   spread across policies: {spread:.4f}\n")
    return results


def by_size_report(dags, scope=1.0, decay=1.0, budget_frac=0.375):
    print("=" * 84)
    print("Does policy choice matter more on the larger graphs?")
    print("=" * 84)
    print(f"scope={scope}, decay={decay}, budget {budget_frac:.1%}\n")
    buckets = {}
    for d in dags:
        buckets.setdefault(d.n, []).append(d)

    print(f"{'n steps':<10}{'count':>8}" + "".join(f"{p:>11}" for p in POLICIES)
          + f"{'spread':>10}")
    print("-" * 84)
    for n in sorted(buckets):
        group = buckets[n]
        vals = {}
        for p in POLICIES:
            vals[p], _ = evaluate_corpus(
                group, p, budget_frac, scope, decay, trials=300
            )
        spread = max(vals.values()) - min(vals.values())
        print(f"{n:<10}{len(group):>8}" + "".join(f"{vals[p]:>11.4f}" for p in POLICIES)
              + f"{spread:>10.4f}")
    print()


def propagation_headroom(dags):
    """How much room does error propagation actually have here?

    Propagation needs depth. If the longest path is 2, an error at a root
    reaches the answer in one hop and there is no snowball to catch.
    """
    print("=" * 84)
    print("Propagation headroom")
    print("=" * 84)
    stats = [dag_stats(d) for d in dags]
    paths = np.array([s["longest_path"] for s in stats])
    print(f"longest path == 2 (root -> answer):   {(paths == 2).mean():.1%}")
    print(f"longest path >= 3 (any intermediate): {(paths >= 3).mean():.1%}")
    print(f"longest path >= 4:                    {(paths >= 4).mean():.1%}")
    print()
    print("A step can only be 'caught before it corrupts downstream' if something")
    print("downstream of it exists. Fraction of steps with >=1 descendant that is")
    print("not the terminal:")
    tot = inter = 0
    for d in dags:
        for v in range(d.n):
            tot += 1
            if len(d.descendants(v) - {d.terminal}) > 0:
                inter += 1
    print(f"  {inter}/{tot} = {inter / tot:.1%}")
    print()


def main():
    if not DATA.exists():
        print(f"missing {DATA}; download strategyqa_dataset.zip first")
        return
    dags = load_dags(DATA)

    structure_report(dags)
    propagation_headroom(dags)
    allocation_report(dags, 1.0, 1.0, "global verifier, no decay")
    allocation_report(dags, 1.0, 0.377, "global verifier, decay=0.377")
    allocation_report(dags, 0.3, 1.0, "weak scope (0.3)")
    by_size_report(dags)

    # Save the extracted structures so downstream work does not re-parse.
    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    payload = [
        {"qid": d.name, "n": d.n, "parents": [list(p) for p in d.parents],
         "terminal": d.terminal, "shape": classify_shape(d)}
        for d in dags
    ]
    (out / "strategyqa_dags.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {len(payload)} graphs to {out / 'strategyqa_dags.json'}")


if __name__ == "__main__":
    main()
