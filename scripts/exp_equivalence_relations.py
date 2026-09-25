"""Three meaning-equivalence relations over one fixed set of samples. [CPU]

Chapter 7 reports that semantic divergence barely ranks step error (AUROC
0.5740 on test, 0.5488 over all steps) and that the result is *relation-
dependent*: re-clustering the same samples by string equality drives it to
0.4904, below chance. That raises an obvious objection to the negative result.
`numeric_equivalence` is a cheap stand-in for the relation the literature
actually uses, so 0.5740 might be a property of the stand-in rather than of
sampling-based step uncertainty.

This settles it by measuring the reference relation. Kuhn et al. and Farquhar
et al. cluster samples by **bidirectional entailment** under an NLI model; that
is now `EntailmentEquivalence`, and

    python scripts/gpu_semantic_divergence.py --cluster-only \
        --checkpoint runs/semantic_samples.jsonl \
        --equivalence entailment \
        --out runs/uncertainty_qwen25_7b_entail.jsonl

re-clusters the committed K=5 samples with it. That pass is 27,936 NLI pairs
and took 19.6 hours on 16 CPU cores, so the verdicts are cached in
runs/entailment_cache.jsonl and committed; this script only reads the result.

The comparison is clean in a way a fresh sampling run would not be: all three
relations see the *same* 12,865 generations, so nothing here is confounded with
a new draw from the generator.

    python scripts/exp_equivalence_relations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from car.data.splits import make_splits  # noqa: E402
from car.eval.metrics import step_detection_auroc  # noqa: E402
from car.generation.replay import attach_gold, load_corpus  # noqa: E402
from car.uncertainty.composite import CompositeScorer  # noqa: E402

CORPUS = Path("runs/generated_qwen25_7b.jsonl")
GSM8K = Path("data/raw/gsm8k/test.jsonl")

RELATIONS = [
    ("numeric equivalence", Path("runs/uncertainty_qwen25_7b_sem.jsonl")),
    ("exact string match", Path("runs/uncertainty_qwen25_7b_exactmatch.jsonl")),
    ("bidirectional entailment", Path("runs/uncertainty_qwen25_7b_entail.jsonl")),
]

# Chapter 7's composite, with semantic divergence carrying a weight.
WEIGHTS = {
    "token_entropy": 1.0,
    "max_surprisal": 0.5,
    "mean_logprob": 1.0,
    "semantic_divergence": 1.0,
    "task_verifier_signal": 0.0,
}
OUT = Path("runs/equivalence_relations.json")


def measure(name, features, corpus_path, gsm8k):
    """Divergence stats plus AUROC, alone and inside the composite."""
    corpus = load_corpus(corpus_path, features, prefix="qwen")
    attach_gold(corpus, gsm8k)
    by_id = {ex.example_id: ex for ex in corpus}
    splits = make_splits([ex.to_example() for ex in corpus],
                         dev_frac=0.3, cal_frac=0.3, salt="car-v1")

    div, ok = [], []
    for ex in corpus:
        for st in ex.steps:
            div.append(st.features.semantic_divergence)
            ok.append(st.global_ok)
    div, ok = np.asarray(div, dtype=float), np.asarray(ok, dtype=bool)

    # Scaler on dev only, exactly as the pipeline fits it -- otherwise the
    # composite AUROC here would not be the one chapter 7 reports.
    scorer = CompositeScorer(WEIGHTS).fit(
        [st.features for ex in splits.dev for st in by_id[ex.example_id].steps])

    def split_arrays(split):
        s, c, y = [], [], []
        for ex in split:
            for st in by_id[ex.example_id].steps:
                s.append(st.features.semantic_divergence)
                c.append(scorer.score(st.features))
                y.append(st.global_ok)
        return np.asarray(s), np.asarray(c), np.asarray(y)

    test_div, test_comp, test_y = split_arrays(splits.test)

    return {
        "relation": name,
        "features": str(features),
        "n_steps": int(div.size),
        "mean_divergence": float(div.mean()),
        "unanimous": int((div <= 1e-9).sum()),
        "auroc_all": step_detection_auroc(div, ok),
        "auroc_test": step_detection_auroc(test_div, test_y),
        "auroc_composite_test": step_detection_auroc(test_comp, test_y),
        "_div_all": div,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--gsm8k", type=Path, default=GSM8K)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows, missing = [], []
    for name, path in RELATIONS:
        if not path.exists():
            missing.append((name, path))
            continue
        rows.append(measure(name, path, args.corpus, args.gsm8k))

    for name, path in missing:
        print(f"skipping {name}: missing {path}")
    if not rows:
        return 1

    print("=" * 92)
    print("MEANING EQUIVALENCE: the same 12,865 samples, three relations")
    print("=" * 92)
    print(f"{'relation':<28}{'mean div':>10}{'unanimous':>12}"
          f"{'AUROC all':>12}{'AUROC test':>12}{'composite':>12}")
    print("-" * 92)
    for r in rows:
        print(f"{r['relation']:<28}{r['mean_divergence']:>10.4f}"
              f"{r['unanimous']:>7,} /{r['n_steps']:>4,}"
              f"{r['auroc_all']:>12.4f}{r['auroc_test']:>12.4f}"
              f"{r['auroc_composite_test']:>12.4f}")

    # How much do the relations actually disagree? Two relations can reach the
    # same AUROC by partitioning the samples quite differently, and if they
    # agree step-for-step then the choice was never load-bearing at all.
    if len(rows) > 1:
        print()
        print("Pairwise agreement on the per-step divergence value:")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i]["_div_all"], rows[j]["_div_all"]
                same = float(np.mean(np.abs(a - b) < 1e-9))
                r = float(np.corrcoef(a, b)[0, 1])
                print(f"  {rows[i]['relation']:<26} vs {rows[j]['relation']:<26}"
                      f"  identical on {same:>6.1%} of steps,  r = {r:+.4f}")

    best = max(rows, key=lambda r: r["auroc_test"])
    spread = best["auroc_test"] - min(r["auroc_test"] for r in rows)
    print()
    print(f"Highest on test: {best['relation']} at {best['auroc_test']:.4f}; "
          f"spread across relations {spread:.4f}.")
    # Deliberately not phrased as a ranking. A solution-clustered bootstrap puts
    # the entailment-minus-numeric gap at 95% CI [-0.087, +0.059], so ordering
    # these three by a fourth decimal would be reading noise.
    if best["auroc_test"] < 0.60:
        print("No relation reaches 0.60. Differences of this size are inside")
        print("the sampling error on 925 test steps, so the result to take is")
        print("that none of them ranks step error -- not which ranks it least")
        print("badly. The ch. 7 negative is a property of sampling-based step")
        print("uncertainty on this corpus, not of the clustering function.")
    else:
        print("A relation clears 0.60, so the ch. 7 number was relation-limited")
        print("and the chapter needs revisiting.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
