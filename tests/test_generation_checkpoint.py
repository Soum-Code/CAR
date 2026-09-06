"""Tests for the generator-transfer run's checkpoint and resume.

The Qwen run took 7h40m with no ability to resume, and a separate run lost two
hours to an OOM that arrived after all the easy work was done. Both are cheap
to prevent and expensive to hit, so the resume logic is tested rather than
trusted -- especially the part that decides what NOT to redo, because a bug
there silently produces a corpus with holes in it.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "gpu_generator_transfer",
    Path(__file__).resolve().parents[1] / "scripts" / "gpu_generator_transfer.py",
)
gt = importlib.util.module_from_spec(_SPEC)
sys.modules["gpu_generator_transfer"] = gt
_SPEC.loader.exec_module(gt)


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_checkpoint_absent_is_empty(tmp_path):
    assert gt.load_checkpoint(None, "solve") == {}
    assert gt.load_checkpoint(tmp_path / "nope.jsonl", "solve") == {}


def test_checkpoint_reloads_by_index(tmp_path):
    p = write(tmp_path / "c.jsonl", [
        {"tag": "solve", "i": 0, "text": "a"},
        {"tag": "solve", "i": 3, "text": "d"},
    ])
    assert gt.load_checkpoint(p, "solve") == {0: "a", 3: "d"}


def test_checkpoint_is_scoped_by_tag(tmp_path):
    """`solve` and `rollout` share a file and index space; mixing them would
    hand a solution's text back as a rollout."""
    p = write(tmp_path / "c.jsonl", [
        {"tag": "solve", "i": 0, "text": "solution"},
        {"tag": "rollout", "i": 0, "text": "continuation"},
    ])
    assert gt.load_checkpoint(p, "solve") == {0: "solution"}
    assert gt.load_checkpoint(p, "rollout") == {0: "continuation"}


def test_torn_final_line_is_skipped(tmp_path):
    """A hard kill truncates the last write. Losing one record is fine; failing
    to load the other 8,000 is not."""
    p = tmp_path / "c.jsonl"
    p.write_text(
        json.dumps({"tag": "solve", "i": 0, "text": "a"}) + "\n"
        + '{"tag": "solve", "i": 1, "te',
        encoding="utf-8",
    )
    assert gt.load_checkpoint(p, "solve") == {0: "a"}


def test_resume_skips_done_and_keeps_order(tmp_path):
    """The whole point: finished prompts are not regenerated, and the returned
    list still lines up with the caller's ordering."""
    prompts = ["p0", "p1", "p2", "p3"]
    p = write(tmp_path / "c.jsonl", [
        {"tag": "solve", "i": 1, "text": "ONE"},
        {"tag": "solve", "i": 3, "text": "THREE"},
    ])
    done = gt.load_checkpoint(p, "solve")
    out = [done.get(i, "") for i in range(len(prompts))]
    todo = [i for i in range(len(prompts)) if i not in done]
    assert out == ["", "ONE", "", "THREE"]
    assert todo == [0, 2]


def test_gated_model_auth_is_not_defaulted_to_a_literal():
    """No token may be hardcoded in this repository. One was found in an older
    kernel of this account, published to a public repo; the fix is that the
    only path in is an explicit argument."""
    src = (Path(__file__).resolve().parents[1]
           / "scripts" / "gpu_generator_transfer.py").read_text(encoding="utf-8")
    assert "hf_" not in src.replace("--hf-token", "").replace("hf_token", "")
    assert 'default=None,\n                    help="passed to from_pretrained' in src


@pytest.mark.parametrize("tag", ["solve", "rollout"])
def test_checkpoint_round_trips_through_a_real_file(tmp_path, tag):
    p = tmp_path / "c.jsonl"
    with p.open("a", encoding="utf-8") as fh:
        for i, text in enumerate(["x", "y", "z"]):
            fh.write(json.dumps({"tag": tag, "i": i, "text": text}) + "\n")
    assert gt.load_checkpoint(p, tag) == {0: "x", 1: "y", 2: "z"}
