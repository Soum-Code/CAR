"""Three meaning-equivalence relations over one fixed set of samples. [CPU]

Chapter 7 reports that semantic divergence barely ranks step error (AUROC 0.5740
on test, 0.5488 over all steps), and an earlier draft added that the result is
*relation-dependent*: re-clustering the same samples by string equality gives
0.4904. That raises an obvious objection to the negative result.
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

re-clusters the committed K=5 samples with it, on CPU. Budget about four hours:
the pass is ~28k ordered NLI pairs at ~2/s on 16 cores. (An earlier version of
this docstring said half an hour, which was a guess, and wrong.) It needs
`microsoft/deberta-large-mnli` from the Hugging Face hub, so it will not run in a
sandbox with no route to huggingface.co.

READ THIS BEFORE SPENDING THE FOUR HOURS
----------------------------------------
`scripts/exp_significance.py` prices this experiment in advance, and the answer
changes how its result should be read. The paired SE between two relations on
these 925 steps is 0.0272, so the smallest difference detectable at 80% power is
0.076: a third relation registers only above **0.6503** AUROC. Chapter 7's
separate score-quality sweep puts the point where the verifier starts paying for
itself at ≈ 0.65.

So this run has exactly two possible outcomes. Entailment lands below ~0.65, in
which case it is indistinguishable from numeric equivalence and the negative
result stands unchanged -- which is worth having, because it answers the
objection. Or it lands above, in which case it is not a better stand-in for a
weak signal, it is a usable signal, and it belongs in ch. 7.6 beside the probe. A
small positive difference is not a third outcome; it is noise, and it should not
be reported as vindication.

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

#: The AUROC a new relation has to beat to be distinguishable from numeric
#: equivalence on these 925 steps: 0.5740 plus the 0.076 minimum detectable
#: difference `exp_significance.py` derives from the paired bootstrap SE.
DETECTABLE_ABOVE = 0.6503


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
    print()
    print(f"Best relation on test: {best['relation']} at "
          f"{best['auroc_test']:.4f}.")
    # The bar is the smallest difference this corpus can resolve, not a round
    # number. exp_significance.py measures it at 0.076 over numeric
    # equivalence's 0.5740; anything under it is a point estimate with an
    # interval through zero, whatever its direction.
    if best["auroc_test"] < DETECTABLE_ABOVE:
        print(f"No relation clears {DETECTABLE_ABOVE:.4f}, the smallest AUROC")
        print("this corpus can distinguish from numeric equivalence at 80%")
        print("power. So the negative result in ch. 7 is a property of")
        print("sampling-based step uncertainty on this corpus, not of the")
        print("equivalence function -- and no ORDERING among the relations")
        print("above is established by these point estimates.")
    else:
        print(f"A relation clears {DETECTABLE_ABOVE:.4f}, which this corpus can")
        print("resolve. It is also above ch. 7.4's crossing, so it is not a")
        print("better stand-in for a weak signal -- it is a usable signal, and")
        print("ch. 7.6 is where it belongs. Re-run exp_significance.py to get")
        print("the paired interval before writing it up.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
        indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
