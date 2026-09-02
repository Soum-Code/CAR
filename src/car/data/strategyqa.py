"""StrategyQA loader and dependency-graph extraction.

StrategyQA (Geva et al., TACL 2021) ships human-annotated decompositions in
BREAK style, where a step refers to earlier steps by index:

    [1] How many kids did Julius Caesar have?
    [2] How many kids did Genghis Khan have?
    [3] Is #2 greater than #1?

Those `#N` markers are an explicit, human-authored dependency graph. This is
the reason StrategyQA is the right benchmark for the propagation work: the DAG
does not have to be inferred from generated text, it is annotated.

Extraction rule: step i depends on every `#N` it mentions with N < i+1. A step
with no references is a root -- an independent premise to be looked up rather
than derived. The last step is the terminal, since StrategyQA decompositions
end with the operation that produces the answer.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from car.topology import DAG
from car.types import Example

_REF = re.compile(r"#(\d+)")


def parse_references(step_text: str) -> list[int]:
    """Zero-based indices of steps referenced by `step_text`."""
    return [int(m) - 1 for m in _REF.findall(step_text)]


def decomposition_to_dag(decomposition: list[str], name: str = "") -> DAG:
    """Build a dependency DAG from an annotated decomposition.

    Forward and self references are dropped rather than raising: a handful of
    annotations contain them, and silently discarding a whole question would
    bias the topology statistics more than repairing the edge does.
    """
    n = len(decomposition)
    parents: list[tuple[int, ...]] = []
    for i, step in enumerate(decomposition):
        refs = sorted({r for r in parse_references(step) if 0 <= r < i})
        parents.append(tuple(refs))
    return DAG(n=n, parents=tuple(parents), terminal=n - 1, name=name)


def load_raw(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_examples(path: str | Path) -> list[Example]:
    """Load StrategyQA into the project's normalised `Example` type."""
    out = []
    for row in load_raw(path):
        evidence: list[str] = []
        for ann in row.get("evidence", []):
            for per_step in ann:
                for item in per_step:
                    if isinstance(item, list):
                        evidence.extend(str(x) for x in item)
                    elif isinstance(item, str):
                        evidence.append(item)
        out.append(
            Example(
                example_id=row["qid"],
                question=row["question"],
                gold_answer=str(row["answer"]),
                decomposition=row["decomposition"],
                evidence=sorted(set(evidence)),
                metadata={"term": row.get("term", ""), "facts": row.get("facts", [])},
            )
        )
    return out


def load_dags(path: str | Path, min_steps: int = 2) -> list[DAG]:
    """Extract one dependency DAG per question."""
    dags = []
    for row in load_raw(path):
        dec = row["decomposition"]
        if len(dec) < min_steps:
            continue
        dags.append(decomposition_to_dag(dec, name=row["qid"]))
    return dags


# ---- topology statistics ---------------------------------------------


def dag_stats(dag: DAG) -> dict:
    """Structural summary of one reasoning graph."""
    inf = dag.influence_profile()
    fan_in = [len(dag.parents[v]) for v in range(dag.n)]
    children = dag.children
    fan_out = [len(children[v]) for v in range(dag.n)]
    roots = [v for v in range(dag.n) if not dag.parents[v]]

    # Longest directed path, i.e. how deep the reasoning actually goes.
    depth = [0] * dag.n
    for v in range(dag.n):
        if dag.parents[v]:
            depth[v] = 1 + max(depth[p] for p in dag.parents[v])

    return {
        "n": dag.n,
        "n_roots": len(roots),
        "max_fan_in": max(fan_in),
        "max_fan_out": max(fan_out) if fan_out else 0,
        "longest_path": max(depth) + 1,
        "mean_influence": float(inf.mean()),
        "corr_pos_influence": dag.position_influence_correlation(),
        "n_edges": sum(fan_in),
        # A step nobody uses and that is not the answer is a dead end -- its
        # descendant count is zero, so influence weighting would never verify
        # it no matter how uncertain it is.
        "n_dead_ends": sum(
            1 for v in range(dag.n) if fan_out[v] == 0 and v != dag.terminal
        ),
    }


def classify_shape(dag: DAG) -> str:
    """Coarse shape label, for comparison against the synthetic topologies."""
    if dag.n <= 1:
        return "trivial"
    children = dag.children
    fan_in = [len(dag.parents[v]) for v in range(dag.n)]
    fan_out = [len(children[v]) for v in range(dag.n)]
    n_roots = sum(1 for v in range(dag.n) if not dag.parents[v])

    if all(f <= 1 for f in fan_in) and all(f <= 1 for f in fan_out) and n_roots == 1:
        return "chain"
    if n_roots > 1 and max(fan_in) > 1:
        return "converging"
    if max(fan_out) > 1:
        return "branching"
    return "other"


def corpus_summary(dags: list[DAG]) -> dict:
    stats = [dag_stats(d) for d in dags]
    shapes = Counter(classify_shape(d) for d in dags)

    def col(k):
        return [s[k] for s in stats]

    import numpy as np

    return {
        "n_questions": len(dags),
        "shapes": dict(shapes),
        "steps": {
            "mean": float(np.mean(col("n"))),
            "dist": dict(sorted(Counter(col("n")).items())),
        },
        "roots": {
            "mean": float(np.mean(col("n_roots"))),
            "dist": dict(sorted(Counter(col("n_roots")).items())),
        },
        "longest_path": {
            "mean": float(np.mean(col("longest_path"))),
            "dist": dict(sorted(Counter(col("longest_path")).items())),
        },
        "max_fan_in": dict(sorted(Counter(col("max_fan_in")).items())),
        "dead_ends": dict(sorted(Counter(col("n_dead_ends")).items())),
        "mean_corr_pos_influence": float(
            np.mean([c for c in col("corr_pos_influence") if not np.isnan(c)])
        ),
    }
