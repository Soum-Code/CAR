"""Language-model backends."""

from car.backends.base import Generation, LMBackend
from car.backends.mock import MockBackend

__all__ = ["Generation", "LMBackend", "MockBackend", "load_backend"]


def load_backend(kind: str, **kwargs):
    """Factory so configs can name a backend as a string.

    The HF backend is imported lazily -- CPU-only machines have no reason to
    pay the torch/transformers import cost just to run the conformal tests.
    """
    if kind == "mock":
        return MockBackend(**kwargs)
    if kind in ("hf", "huggingface"):
        from car.backends.hf import HFBackend

        return HFBackend(**kwargs)
    raise ValueError(f"unknown backend: {kind!r} (expected 'mock' or 'hf')")
