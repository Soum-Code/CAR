"""GSM8K loader and dependency-graph extraction.

GSM8K (Cobbe et al., 2021) has no annotated decomposition, but its solutions
carry inline calculator markers:

    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether.
    #### 72

That is enough to *derive* the dependency graph: line i depends on line j when
an operand of line i equals the result of line j. Values that match no earlier
result are givens taken from the problem statement, i.e. roots.

    He eats 32 ... because 2 x 16 = <<2*16=32>>      roots: 2, 16
    He eats 16 ... because 2 x 8  = <<2*8=16>>       roots: 2, 8
    He eats 48 ... because 32 + 16 = <<32+16=48>>    depends on BOTH earlier lines

Note the structure there is converging, not a chain -- GSM8K is not purely
linear, which is what makes it a usable replacement for StrategyQA rather than
just a deeper version of the same shape.

Ambiguity, stated honestly
--------------------------
An operand can match both an earlier result and a number in the question (the
`16` above does). Linking to the earlier result is right far more often than
not -- a solution line that re-uses a computed subtotal is the norm -- so that
is the default, and `extraction_stats` reports how often the case arises so the
rate is visible rather than assumed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from car.topology import DAG
from car.types import Example

_CALC = re.compile(r"<<([^>]*?)=([^>]*?)>>")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", "").strip())
    except (ValueError, AttributeError):
        return None


def parse_calc_steps(answer: str) -> list[tuple[str, float]]:
    """Extract (expression, result) for each calculator annotation, in order."""
    steps = []
    for expr, res in _CALC.findall(answer):
        val = _to_float(res)
        if val is not None:
            steps.append((expr.strip(), val))
    return steps


def question_numbers(question: str) -> set[float]:
    vals = {_to_float(m) for m in _NUM.findall(question)}
    return {v for v in vals if v is not None}


def solution_to_dag(
    question: str, answer: str, name: str = ""
) -> tuple[DAG, dict] | tuple[None, dict]:
    """Derive a dependency DAG from a GSM8K solution.

    Returns (dag, stats), or (None, stats) when there are fewer than two
    calculator steps -- a single-step solution has no dependency structure to
    study and would only dilute the corpus statistics.
    """
    steps = parse_calc_steps(answer)
    stats = {"n_steps": len(steps), "n_ambiguous": 0, "n_operands": 0, "n_linked": 0}
    if len(steps) < 2:
        return None, stats

    q_nums = question_numbers(question)
    results = [r for _, r in steps]
    parents: list[tuple[int, ...]] = []

    for i, (expr, _) in enumerate(steps):
        operands = [_to_float(m) for m in _NUM.findall(expr)]
        operands = [o for o in operands if o is not None]
        deps: set[int] = set()
        for o in operands:
            stats["n_operands"] += 1
            # Most recent earlier line producing this value.
            match = None
            for j in range(i - 1, -1, -1):
                if abs(results[j] - o) < 1e-9:
                    match = j
                    break
            if match is None:
                continue
            if o in q_nums:
                # Could equally be a given restated. Linked anyway; counted.
                stats["n_ambiguous"] += 1
            deps.add(match)
            stats["n_linked"] += 1
        parents.append(tuple(sorted(deps)))

    dag = DAG(n=len(steps), parents=tuple(parents), terminal=len(steps) - 1, name=name)
    return dag, stats


def load_raw(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def final_answer(answer: str) -> str:
    return answer.split("####")[-1].strip()


def load_examples(path: str | Path) -> list[Example]:
    out = []
    for i, row in enumerate(load_raw(path)):
        steps = parse_calc_steps(row["answer"])
        out.append(
            Example(
                example_id=f"gsm8k_{i}",
                question=row["question"],
                gold_answer=final_answer(row["answer"]),
                # Each calculator step is a verifiable atomic transformation --
                # exactly the unit CalculatorVerifier checks.
                decomposition=[f"{e} = {r:g}" for e, r in steps],
                metadata={"solution": row["answer"]},
            )
        )
    return out


def load_dags(path: str | Path, limit: int | None = None) -> list[DAG]:
    dags = []
    for i, row in enumerate(load_raw(path)):
        if limit is not None and len(dags) >= limit:
            break
        dag, _ = solution_to_dag(row["question"], row["answer"], name=f"gsm8k_{i}")
        if dag is not None:
            dags.append(dag)
    return dags


def extraction_stats(path: str | Path) -> dict:
    """Corpus-level extraction quality, so the derivation can be audited."""
    total = {"rows": 0, "usable": 0, "n_operands": 0, "n_linked": 0, "n_ambiguous": 0}
    orphan_steps = 0
    step_count = 0
    for row in load_raw(path):
        total["rows"] += 1
        dag, st = solution_to_dag(row["question"], row["answer"])
        for k in ("n_operands", "n_linked", "n_ambiguous"):
            total[k] += st[k]
        if dag is not None:
            total["usable"] += 1
            for v in range(1, dag.n):
                step_count += 1
                if not dag.parents[v]:
                    orphan_steps += 1
    total["link_rate"] = total["n_linked"] / max(1, total["n_operands"])
    total["ambiguous_rate"] = total["n_ambiguous"] / max(1, total["n_linked"])
    # A non-first step with no parents recomputes from givens rather than
    # building on prior work -- a parallel branch, not an extraction failure,
    # but worth watching.
    total["orphan_step_rate"] = orphan_steps / max(1, step_count)
    return total
