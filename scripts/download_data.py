"""Fetch the datasets. Run once after cloning.

    python scripts/download_data.py              # everything
    python scripts/download_data.py gsm8k        # just one
    python scripts/download_data.py --verify     # check what is already there

Idempotent: anything already present is skipped unless --force.

The Math-Shepherd fetch is the reason this file exists rather than three curl
commands in a README. That file is sorted into contiguous blocks by label and
by task, so ANY prefix read is close to single-class -- the first 80MB is 100%
wrong-answer GSM8K, and reading it produces a plausible-looking table in which
every measured error rate is wrong. It cost two debugging rounds to find, twice.

So `fetch_math_shepherd` reads 48 strided range requests spread across the whole
file and then ASSERTS the resulting class balance is sane. A silently biased
sample is worse than a failed download, so this fails loudly instead.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RAW = Path("data/raw")

STRATEGYQA_URL = "https://storage.googleapis.com/ai2i/strategyqa/data/strategyqa_dataset.zip"
GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/train.jsonl"
)
MATH_SHEPHERD_URL = (
    "https://huggingface.co/datasets/peiyi9979/Math-Shepherd/"
    "resolve/main/math-shepherd.jsonl"
)
# Full size of math-shepherd.jsonl. Used to space the strided reads; a mismatch
# means the file changed upstream and the stride plan should be rechecked.
MATH_SHEPHERD_BYTES = 793_059_998
N_CHUNKS = 48
CHUNK_BYTES = 2_500_000


def _get(url: str, timeout: int = 300, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _trim_partial_lines(blob: bytes, drop_first: bool = True) -> bytes:
    """A range request slices mid-line at both ends; drop both fragments."""
    start = blob.find(b"\n") + 1 if drop_first else 0
    end = blob.rfind(b"\n") + 1
    return blob[start:end] if end > start else b""


# ---- StrategyQA ------------------------------------------------------


def fetch_strategyqa(force: bool = False) -> Path:
    out = RAW / "strategyqa"
    target = out / "strategyqa_train.json"
    if target.exists() and not force:
        print(f"  strategyqa      already present ({target})")
        return target

    print("  strategyqa      downloading...")
    out.mkdir(parents=True, exist_ok=True)
    blob = _get(STRATEGYQA_URL)
    zipfile.ZipFile(io.BytesIO(blob)).extractall(out)
    rows = json.loads(target.read_text(encoding="utf-8"))
    print(f"  strategyqa      OK  {len(rows):,} questions")
    return target


def verify_strategyqa() -> bool:
    p = RAW / "strategyqa" / "strategyqa_train.json"
    if not p.exists():
        print("  strategyqa      MISSING")
        return False
    rows = json.loads(p.read_text(encoding="utf-8"))
    with_decomp = sum(1 for r in rows if len(r.get("decomposition", [])) >= 2)
    ok = len(rows) > 2000 and with_decomp > 2000
    print(f"  strategyqa      {'OK ' if ok else 'BAD'} {len(rows):,} questions, "
          f"{with_decomp:,} with a usable decomposition")
    return ok


# ---- GSM8K -----------------------------------------------------------


def fetch_gsm8k(force: bool = False) -> Path:
    out = RAW / "gsm8k"
    target = out / "train.jsonl"
    if target.exists() and not force:
        print(f"  gsm8k           already present ({target})")
        return target

    print("  gsm8k           downloading...")
    out.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_get(GSM8K_URL))
    n = sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"  gsm8k           OK  {n:,} problems")
    return target


def verify_gsm8k() -> bool:
    p = RAW / "gsm8k" / "train.jsonl"
    if not p.exists():
        print("  gsm8k           MISSING")
        return False
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    with_calc = sum(1 for r in rows if "<<" in r.get("answer", ""))
    ok = len(rows) > 7000 and with_calc > 6000
    print(f"  gsm8k           {'OK ' if ok else 'BAD'} {len(rows):,} problems, "
          f"{with_calc:,} with calculator annotations")
    return ok


# ---- Math-Shepherd ---------------------------------------------------


def fetch_math_shepherd(force: bool = False) -> Path:
    """Strided fetch. See the module docstring for why a prefix read is wrong."""
    out = RAW / "mathshepherd"
    target = out / "strided.jsonl"
    if target.exists() and not force:
        print(f"  math-shepherd   already present ({target})")
        return target

    print(f"  math-shepherd   downloading {N_CHUNKS} strided chunks "
          f"(~{N_CHUNKS * CHUNK_BYTES // 1_000_000} MB of {MATH_SHEPHERD_BYTES // 1_000_000} MB)...")
    out.mkdir(parents=True, exist_ok=True)

    written = 0
    failed = 0
    with target.open("wb") as fh:
        for i in range(N_CHUNKS):
            start = int(MATH_SHEPHERD_BYTES * i / N_CHUNKS)
            end = min(start + CHUNK_BYTES, MATH_SHEPHERD_BYTES - 1)
            try:
                blob = _get(MATH_SHEPHERD_URL, byte_range=(start, end))
            except Exception as exc:  # noqa: BLE001 - report and continue
                failed += 1
                print(f"    chunk {i} failed: {exc}")
                continue
            # Drop the leading fragment on every chunk INCLUDING the first:
            # a mid-line start is unparseable either way.
            clean = _trim_partial_lines(blob)
            fh.write(clean)
            written += len(clean)
            if i % 12 == 0:
                print(f"    chunk {i:>2}/{N_CHUNKS}  {written // 1_000_000} MB")

    if failed > N_CHUNKS // 4:
        raise RuntimeError(f"{failed}/{N_CHUNKS} chunks failed; sample may be skewed")
    print(f"  math-shepherd   fetched {written // 1_000_000} MB")

    balance = verify_math_shepherd(strict=True)
    print(f"  math-shepherd   OK  class balance {balance:.1%}")
    return target


def verify_math_shepherd(strict: bool = False) -> float:
    """Return the fraction of GSM8K solutions whose final answer is correct.

    Near 0 or 1 means the sample landed inside one of the sorted blocks and
    every rate computed from it will be wrong.
    """
    from car.data.math_shepherd import class_balance, load_solutions

    p = RAW / "mathshepherd" / "strided.jsonl"
    if not p.exists():
        print("  math-shepherd   MISSING")
        return 0.0

    sols = load_solutions(p)
    bal = class_balance(sols)
    ok = 0.05 < bal < 0.95
    if not ok:
        msg = (
            f"class balance {bal:.1%} -- the sample is effectively single-class. "
            f"This is the block-sorting trap; re-run with --force."
        )
        if strict:
            raise RuntimeError(msg)
        print(f"  math-shepherd   BAD {msg}")
    else:
        print(f"  math-shepherd   OK  {len(sols):,} GSM8K solutions, "
              f"class balance {bal:.1%}")
    return bal


# ---- driver ----------------------------------------------------------

DATASETS = {
    "strategyqa": (fetch_strategyqa, verify_strategyqa),
    "gsm8k": (fetch_gsm8k, verify_gsm8k),
    "math-shepherd": (fetch_math_shepherd, verify_math_shepherd),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch CAR datasets")
    ap.add_argument("datasets", nargs="*", choices=list(DATASETS), default=[],
                    metavar="DATASET",
                    help=f"which to fetch: {', '.join(DATASETS)} (default: all)")
    ap.add_argument("--verify", action="store_true", help="check what is present, fetch nothing")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    names = args.datasets or list(DATASETS)

    if args.verify:
        print("verifying:")
        ok = all(DATASETS[n][1]() for n in names)
        print("\nall present and sane" if ok else "\nsomething is missing or skewed")
        return 0 if ok else 1

    print(f"fetching into {RAW}/ :")
    for n in names:
        DATASETS[n][0](force=args.force)
    print("\ndone. run: python -m pytest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
