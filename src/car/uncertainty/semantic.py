"""Semantic divergence at the level of a single reasoning step.

Farquhar et al. (Nature 2024) compute semantic entropy over whole answers by
clustering samples into meaning-equivalence classes with bidirectional
entailment, then taking entropy over the cluster distribution. CAR applies the
same principle to one intermediate step.

Worth knowing before you rely on this: recent work (arXiv 2602.02427) reports
that sampling-agreement methods are weaker at pinpointing *intermediate* step
uncertainty than they are at whole-answer uncertainty. That is an open question
for this project, not a settled result -- which is exactly why the equivalence
function is pluggable and why the composite score does not assume this feature
carries the load.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

# Given two texts, return True if they mean the same thing.
EquivalenceFn = Callable[[str, str], bool]


def exact_match_equivalence(a: str, b: str) -> bool:
    """Cheapest possible baseline: normalised string equality.

    Deliberately weak. It exists so the pipeline runs without an NLI model,
    and as an ablation floor showing what lexical-only clustering buys you.
    """
    return a.strip().lower() == b.strip().lower()


def numeric_equivalence(a: str, b: str) -> bool:
    """Two arithmetic steps mean the same thing if they assert the same value.

    Bidirectional entailment needs an NLI model and a second GPU pass. On
    arithmetic reasoning there is a cheaper equivalence that is *closer to the
    intended meaning*, not merely cheaper: what an arithmetic step asserts is a
    number, so two samples that reach the same number agree regardless of how
    they phrase it.

        "He sold 48/2 = <<48/2=24>>24 clips"
        "Half of 48 is \\( 48 / 2 = 24 \\)"

    are one cluster here and two under exact match, which would report
    disagreement where the model has none. That failure mode inflates
    divergence exactly on the verbose, well-hedged steps, i.e. it correlates
    with style rather than uncertainty.

    Falls back to normalised string equality when neither side asserts a
    number, so prose steps still cluster sensibly.
    """
    from car.data.generated import arithmetic_claims

    va = _asserted_value(a, arithmetic_claims)
    vb = _asserted_value(b, arithmetic_claims)
    if va is None or vb is None:
        return exact_match_equivalence(a, b)
    return abs(va - vb) < 1e-6


def _asserted_value(text: str, claims_fn) -> float | None:
    """The last number this step claims, or None if it claims none."""
    pairs, _ = claims_fn(text, notation="any")
    for _, result in reversed(pairs):
        try:
            return float(str(result).replace(",", "").strip())
        except (ValueError, AttributeError):
            continue
    return None


class EntailmentEquivalence:
    """Bidirectional entailment, the relation Kuhn et al. and Farquhar et al. use.

    Two samples mean the same thing when each entails the other under an NLI
    model. This is the reference relation for semantic entropy, and the reason
    it is worth running here is that Chapter 7's result depends on the relation:
    the same samples give AUROC 0.5488 under `numeric_equivalence` and 0.4904 --
    below chance -- under `exact_match_equivalence`. Measuring a third relation
    is what separates "sampling-based step uncertainty is weak" from "the cheap
    relation was hiding a signal".

    BATCHING, AND WHY `prime_many` EXISTS
    -------------------------------------
    `cluster_by_equivalence` asks one pair at a time, which on CPU means one
    forward pass per question. `prime` fixes that for a single step, but a step
    has only ~11 distinct ordered pairs -- far too few to fill a batch, and
    measured at under 0.2 steps/s on 16 cores. `prime_many` collects the pairs
    for *every* step first, sorts them by length so padding is not paid on
    short pairs, and runs them in full batches. Same answers, an order of
    magnitude less wall time.

    The cache is keyed on (context, premise, hypothesis). Dropping the context
    from that key would be wrong rather than merely approximate: the same two
    step fragments can entail each other under one question and not under
    another, and a context-blind cache silently reuses the first verdict.
    """

    #: MNLI label order for this checkpoint family: contradiction/neutral/entail.
    ENTAILMENT_ID = 2

    def __init__(self, model_id="microsoft/deberta-large-mnli", *, context="",
                 max_length=192, batch_size=64, device=None, threads=None,
                 cache_path=None):
        self.model_id = model_id
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.threads = threads
        self._context = context
        self._cache: dict[tuple[str, str, str], bool] = {}
        self._model = None
        self._tok = None
        self.calls = 0
        self.cached_hits = 0
        self.cache_path = Path(cache_path) if cache_path else None
        self._fh = None
        if self.cache_path and self.cache_path.exists():
            self._load_cache()

    def _load_cache(self) -> None:
        """Resume. 27,936 pairs measured 19.6 hours on 16 CPU cores.

        Losing that to a reboot, a sleeping laptop or a stray Ctrl-C is the
        same failure that cost this project a GPU session once already, so the
        verdicts are written as they are produced and read back on restart.
        """
        with self.cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a half-written final line after a hard kill
                self._cache[(rec["c"], rec["a"], rec["b"])] = bool(rec["e"])
        self.cached_hits = len(self._cache)

    def _append_cache(self, items) -> None:
        if not self.cache_path:
            return
        if self._fh is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.cache_path.open("a", encoding="utf-8")
        for (c, a, b), ok in items:
            self._fh.write(json.dumps({"c": c, "a": a, "b": b, "e": bool(ok)}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # -- the model is loaded on first use, so importing this module stays cheap
    def _ensure_model(self):
        if self._model is not None:
            return
        import os

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # torch defaults to half the cores on this box; the whole run is one
        # long stretch of CPU matmul, so there is nothing to leave headroom for.
        torch.set_num_threads(self.threads or os.cpu_count() or 1)
        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id).eval()
        if self.device:
            self._model.to(self.device)

    def set_context(self, context: str) -> None:
        """Question text prepended to both sides.

        Kuhn et al. condition entailment on the question, because two answers
        can be mutually entailing as bare strings and disagree as answers. The
        samples here are step continuations, so the context is the question the
        step belongs to.
        """
        self._context = context or ""

    def _pair(self, a: str, b: str) -> tuple[str, str]:
        if self._context:
            return (f"{self._context} {a}", f"{self._context} {b}")
        return (a, b)

    def _entails_batch(self, keys: list[tuple[str, str, str]]) -> list[bool]:
        """Directed entailment for each (context, premise, hypothesis)."""
        if not keys:
            return []
        self._ensure_model()
        torch = self._torch
        out: list[bool] = []
        for i in range(0, len(keys), self.batch_size):
            chunk = keys[i: i + self.batch_size]
            left = [f"{c} {a}".strip() for c, a, _ in chunk]
            right = [f"{c} {b}".strip() for c, _, b in chunk]
            enc = self._tok(left, right, return_tensors="pt", padding=True,
                            truncation=True, max_length=self.max_length)
            if self.device:
                enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            out.extend((logits.argmax(-1) == self.ENTAILMENT_ID).tolist())
            self.calls += len(chunk)
        return out

    def _pending(self, context: str, texts: Sequence[str], seen: set) -> list:
        keys = []
        for a in texts:
            for b in texts:
                k = (context, a, b)
                if a == b or k in self._cache or k in seen:
                    continue
                seen.add(k)
                keys.append(k)
        return keys

    def prime(self, texts: Sequence[str]) -> None:
        """Cache every ordered pair among one step's `texts`."""
        self.prime_many([(self._context, texts)])

    def prime_many(self, groups, *, progress=None) -> None:
        """Cache every ordered pair across many (context, texts) groups.

        Sorted by tokenised length before batching: a batch pads to its longest
        member, so mixing a 30-token pair with a 190-token one makes the short
        one cost the same as the long one.
        """
        seen: set = set()
        keys: list = []
        for context, texts in groups:
            keys.extend(self._pending(context or "", texts, seen))
        if not keys:
            return
        keys.sort(key=lambda k: len(k[0]) + len(k[1]) + len(k[2]))

        done = 0
        for i in range(0, len(keys), self.batch_size):
            chunk = keys[i: i + self.batch_size]
            verdicts = list(zip(chunk, self._entails_batch(chunk), strict=True))
            for k, ok in verdicts:
                self._cache[k] = ok
            self._append_cache(verdicts)
            done += len(chunk)
            if progress is not None:
                progress(done, len(keys))

    def entails(self, a: str, b: str) -> bool:
        if a == b:
            return True
        k = (self._context or "", a, b)
        if k not in self._cache:
            self._cache[k] = self._entails_batch([k])[0]
        return self._cache[k]

    def __call__(self, a: str, b: str) -> bool:
        return self.entails(a, b) and self.entails(b, a)


def cluster_by_equivalence(
    texts: Sequence[str], equivalent: EquivalenceFn = exact_match_equivalence
) -> list[int]:
    """Greedy transitive clustering into meaning classes.

    Assigns each text to the first existing cluster whose representative it is
    equivalent to. O(n * k) comparisons for n samples and k clusters, which is
    fine for the n=5-10 regime semantic entropy actually uses.
    """
    reps: list[str] = []
    assignments: list[int] = []
    for t in texts:
        for ci, rep in enumerate(reps):
            if equivalent(t, rep):
                assignments.append(ci)
                break
        else:
            reps.append(t)
            assignments.append(len(reps) - 1)
    return assignments


def semantic_entropy(
    cluster_ids: Sequence[int], logprobs: Sequence[float] | None = None
) -> float:
    """Shannon entropy over the cluster distribution.

    If per-sample sequence logprobs are supplied, clusters are weighted by
    their summed probability mass (the Rao-Blackwellised variant); otherwise
    clusters are weighted by raw sample counts (the discrete variant).
    """
    ids = np.asarray(list(cluster_ids))
    if ids.size == 0:
        return 0.0

    n_clusters = int(ids.max()) + 1
    if logprobs is None:
        counts = np.bincount(ids, minlength=n_clusters).astype(float)
        probs = counts / counts.sum()
    else:
        lp = np.asarray(list(logprobs), dtype=float)
        # Stabilise before exponentiating; step logprobs can be very negative.
        w = np.exp(lp - lp.max())
        mass = np.bincount(ids, weights=w, minlength=n_clusters)
        total = mass.sum()
        if total <= 0:
            return 0.0
        probs = mass / total

    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())


def normalised_semantic_divergence(
    cluster_ids: Sequence[int], logprobs: Sequence[float] | None = None
) -> float:
    """Semantic entropy scaled to [0, 1] by the maximum entropy for n samples.

    Normalising matters because the composite score fuses this with token
    entropy, and an unnormalised feature would let sample count silently
    dominate the weighting.
    """
    ids = list(cluster_ids)
    n = len(ids)
    if n <= 1:
        return 0.0
    h = semantic_entropy(ids, logprobs)
    return float(h / np.log(n))
