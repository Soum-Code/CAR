"""Hand-validation of derived GSM8K dependency graphs.

Every GSM8K dependency edge in this project is DERIVED, not annotated: line i is
linked to line j when an operand of i equals the result of j. 9.6% of those
links are ambiguous, because the operand also appears as a number in the
question, so it could equally be a restated given. That is the largest
unaddressed caveat under every GSM8K result.

This script does two jobs:

  extract   emit adjudication packets -- question, numbered solution lines,
            and the derived edges -- for a stratified sample
  score     compare recorded human/LLM judgements against the algorithm and
            report precision, recall, and a stratified error estimate

Sampling is stratified because ambiguity is concentrated: only 11.6% of graphs
contain an ambiguous link, so a uniform sample of 50 would contain ~5 of the
cases actually at risk and could not resolve the rate that matters. Half the
sample is drawn from graphs with >=1 ambiguous link, half from clean graphs,
and the strata are recombined at their true weights.

Adjudication asks two questions per graph, because operand matching can fail in
both directions:

  SPURIOUS  an edge the algorithm asserted that the text does not support
  MISSING   a dependency the text implies that the algorithm did not find
            (e.g. a value carried in prose rather than as a numeric operand)

Reporting only spurious edges would measure precision and call it accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from car.data.gsm8k import (  # noqa: E402
    expression_operands,
    load_raw,
    parse_calc_steps,
    question_numbers,
    solution_to_dag,
)

DATA = Path("data/raw/gsm8k/train.jsonl")
PACKETS = Path("data/processed/depgraph_packets.json")
JUDGEMENTS = Path("data/processed/depgraph_judgements.json")

# Corpus-level stratum weights, measured over all 6,974 usable graphs.
P_AMBIGUOUS_GRAPH = 0.1163
P_AMBIGUOUS_LINK = 0.0961


def build_packets(n_per_stratum=25, seed=0):
    rows = load_raw(DATA)
    amb, clean = [], []

    for i, r in enumerate(rows):
        dag, st = solution_to_dag(r["question"], r["answer"], name=f"gsm8k_{i}")
        if dag is None:
            continue
        (amb if st["n_ambiguous"] > 0 else clean).append((i, r, dag, st))

    rng = random.Random(seed)
    picked = rng.sample(amb, n_per_stratum) + rng.sample(clean, n_per_stratum)

    packets = []
    for i, r, dag, st in picked:
        steps = parse_calc_steps(r["answer"])
        q_nums = question_numbers(r["question"])
        results = [res for _, res in steps]

        lines = []
        for k, (expr, res) in enumerate(steps):
            operands = expression_operands(expr)
            detail = []
            for o in operands:
                # Which earlier line, if any, produced this operand?
                src = next(
                    (j for j in range(k - 1, -1, -1) if abs(results[j] - o) < 1e-9),
                    None,
                )
                detail.append({
                    "operand": o,
                    "linked_to_line": (src + 1) if src is not None else None,
                    "also_in_question": o in q_nums,
                    "ambiguous": src is not None and o in q_nums,
                })
            lines.append({
                "line": k + 1,
                "expr": expr,
                "result": res,
                "operands": detail,
            })

        packets.append({
            "id": f"gsm8k_{i}",
            "stratum": "ambiguous" if st["n_ambiguous"] > 0 else "clean",
            "question": r["question"].strip(),
            "solution": r["answer"].split("####")[0].strip(),
            "lines": lines,
            "derived_edges": [
                {"from": p + 1, "to": v + 1}
                for v, ps in enumerate(dag.parents) for p in ps
            ],
            "n_ambiguous": st["n_ambiguous"],
        })
    return packets


def render(pkt) -> str:
    out = [f"=== {pkt['id']}  [{pkt['stratum']}]  ambiguous_links={pkt['n_ambiguous']}",
           f"Q: {pkt['question']}", "", "SOLUTION:"]
    for ln in pkt["solution"].splitlines():
        if ln.strip():
            out.append(f"   {ln.strip()}")
    out.append("")
    out.append("DERIVED OPERAND LINKS:")
    for line in pkt["lines"]:
        parts = []
        for o in line["operands"]:
            tag = ""
            if o["linked_to_line"]:
                tag = f"<-L{o['linked_to_line']}"
                if o["ambiguous"]:
                    tag += "(AMB: also a question number)"
            elif o["also_in_question"]:
                tag = "<-given"
            else:
                tag = "<-?"
            parts.append(f"{o['operand']:g}{tag}")
        out.append(f"   L{line['line']}: {line['expr']} = {line['result']:g}"
                   f"   [{', '.join(parts)}]")
    out.append(f"EDGES: {[(e['from'], e['to']) for e in pkt['derived_edges']]}")
    return "\n".join(out)


def score(judgements_path=JUDGEMENTS):
    """Combine recorded judgements into precision, recall and a stratified rate."""
    j = json.loads(Path(judgements_path).read_text(encoding="utf-8"))
    packets = {p["id"]: p for p in json.loads(PACKETS.read_text(encoding="utf-8"))}

    per_stratum = {}
    for rec in j["judgements"]:
        pkt = packets[rec["id"]]
        s = pkt["stratum"]
        d = per_stratum.setdefault(
            s, {"graphs": 0, "edges": 0, "spurious": 0, "missing": 0, "bad_graphs": 0}
        )
        d["graphs"] += 1
        d["edges"] += len(pkt["derived_edges"])
        d["spurious"] += len(rec.get("spurious", []))
        d["missing"] += len(rec.get("missing", []))
        if rec.get("spurious") or rec.get("missing"):
            d["bad_graphs"] += 1

    print("=" * 78)
    print("Hand-validation of derived GSM8K dependency graphs")
    print("=" * 78)
    print(f"{'stratum':<12}{'graphs':>8}{'edges':>8}{'spurious':>10}"
          f"{'missing':>9}{'edge err':>10}{'graph err':>11}")
    print("-" * 78)
    for s in ("ambiguous", "clean"):
        d = per_stratum.get(s)
        if not d:
            continue
        ee = (d["spurious"] + d["missing"]) / max(1, d["edges"])
        ge = d["bad_graphs"] / max(1, d["graphs"])
        print(f"{s:<12}{d['graphs']:>8}{d['edges']:>8}{d['spurious']:>10}"
              f"{d['missing']:>9}{ee:>10.4f}{ge:>11.4f}")

    a, c = per_stratum.get("ambiguous"), per_stratum.get("clean")
    if a and c:
        ea = (a["spurious"] + a["missing"]) / max(1, a["edges"])
        ec = (c["spurious"] + c["missing"]) / max(1, c["edges"])
        ga = a["bad_graphs"] / max(1, a["graphs"])
        gc = c["bad_graphs"] / max(1, c["graphs"])
        print("-" * 78)
        print(f"\nStratified to corpus weights "
              f"({P_AMBIGUOUS_GRAPH:.1%} of graphs contain an ambiguous link):")
        print(f"  corpus edge error rate   "
              f"{P_AMBIGUOUS_GRAPH * ea + (1 - P_AMBIGUOUS_GRAPH) * ec:.4f}")
        print(f"  corpus graph error rate  "
              f"{P_AMBIGUOUS_GRAPH * ga + (1 - P_AMBIGUOUS_GRAPH) * gc:.4f}")
        tot_edges = a["edges"] + c["edges"]
        tot_bad = a["spurious"] + a["missing"] + c["spurious"] + c["missing"]
        print(f"\n  (unweighted over the 50 sampled graphs: "
              f"{tot_bad}/{tot_edges} edges = {tot_bad / max(1, tot_edges):.4f})")
    print()
    if "notes" in j:
        print("Adjudicator notes:")
        for n in j["notes"]:
            print(f"  - {n}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "show", "score"])
    ap.add_argument("--n", type=int, default=25, help="graphs per stratum")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "extract":
        pkts = build_packets(args.n, args.seed)
        PACKETS.parent.mkdir(parents=True, exist_ok=True)
        PACKETS.write_text(json.dumps(pkts, indent=1), encoding="utf-8")
        print(f"wrote {len(pkts)} packets to {PACKETS}")
    elif args.mode == "show":
        for p in json.loads(PACKETS.read_text(encoding="utf-8")):
            print(render(p))
            print()
    else:
        score()


if __name__ == "__main__":
    main()
