"""Tests for the dataset bootstrap script.

Network is never touched here. What is tested is the logic that made the script
necessary: line-fragment trimming on range reads, and the class-balance guard
that catches the Math-Shepherd block-sorting trap.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "download_data", Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
)
download_data = importlib.util.module_from_spec(_SPEC)
sys.modules["download_data"] = download_data
_SPEC.loader.exec_module(download_data)


def test_all_datasets_registered():
    assert set(download_data.DATASETS) == {"strategyqa", "gsm8k", "math-shepherd"}
    for fetch, verify in download_data.DATASETS.values():
        assert callable(fetch) and callable(verify)


def test_stride_plan_covers_the_whole_file():
    """The point of striding: chunks must span the file, not cluster at the
    front, or the sample lands inside a single sorted block."""
    total = download_data.MATH_SHEPHERD_BYTES
    n = download_data.N_CHUNKS
    starts = [int(total * i / n) for i in range(n)]

    assert starts[0] == 0
    assert starts[-1] > total * 0.9, "last chunk must reach the end of the file"
    gaps = [b - a for a, b in zip(starts[:-1], starts[1:], strict=True)]
    assert min(gaps) > download_data.CHUNK_BYTES, "chunks must not overlap"


def test_trim_drops_fragments_at_both_ends():
    """A range request slices mid-line at both ends; both fragments are
    unparseable and must go."""
    blob = b'{"partial": tr\n{"good": 1}\n{"also": 2}\n{"trunc'
    out = download_data._trim_partial_lines(blob)
    assert out == b'{"good": 1}\n{"also": 2}\n'


def test_trim_handles_no_newline():
    assert download_data._trim_partial_lines(b"no newlines here") == b""


def test_trim_can_keep_first_line():
    blob = b'{"a": 1}\n{"b": 2}\n'
    assert download_data._trim_partial_lines(blob, drop_first=False) == blob


@pytest.mark.parametrize("balance", [0.0, 0.02, 0.98, 1.0])
def test_single_class_balance_is_rejected(balance, monkeypatch, tmp_path):
    """The trap this script exists to prevent: a prefix read of Math-Shepherd
    is ~100% one class, and every rate computed from it is wrong."""
    monkeypatch.setattr(download_data, "RAW", tmp_path)
    d = tmp_path / "mathshepherd"
    d.mkdir(parents=True)
    (d / "strided.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "car.data.math_shepherd.load_solutions", lambda *a, **k: ["x"]
    )
    monkeypatch.setattr("car.data.math_shepherd.class_balance", lambda s: balance)

    with pytest.raises(RuntimeError, match="single-class"):
        download_data.verify_math_shepherd(strict=True)


def test_healthy_balance_is_accepted(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "RAW", tmp_path)
    d = tmp_path / "mathshepherd"
    d.mkdir(parents=True)
    (d / "strided.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "car.data.math_shepherd.load_solutions", lambda *a, **k: ["x"]
    )
    monkeypatch.setattr("car.data.math_shepherd.class_balance", lambda s: 0.267)

    assert download_data.verify_math_shepherd(strict=True) == pytest.approx(0.267)


def test_verify_reports_missing_without_raising(monkeypatch, tmp_path):
    monkeypatch.setattr(download_data, "RAW", tmp_path)
    assert download_data.verify_math_shepherd() == 0.0
    assert download_data.verify_gsm8k() is False
    assert download_data.verify_strategyqa() is False
