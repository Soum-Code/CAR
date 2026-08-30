"""Content-addressed generation cache.

This is the module that turns "I need a GPU for my whole project" into "I need
a GPU for a few hours". Generation is expensive and happens once; the conformal
layer, the gate, every baseline and every ablation then run on CPU against the
cached scores.

Cache key is a hash of (model name, prompt, n, temperature, seed). Any change
to sampling parameters produces a different key, so a stale cache cannot
silently contaminate a run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from car.backends.base import Generation


class CachedBackend:
    """Wraps any LMBackend with a disk cache."""

    def __init__(self, backend, cache_dir: str | Path = ".cache/generations") -> None:
        self.backend = backend
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self.backend.name

    def _key(self, prompt: str, n: int, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": self.backend.name,
                "prompt": prompt,
                "n": n,
                "temperature": round(float(temperature), 6),
            },
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        # Shard by first two chars; a flat dir with 100k files is painful on
        # Windows and on network filesystems.
        return self.cache_dir / key[:2] / f"{key}.npz"

    def generate(
        self, prompt: str, *, n: int = 1, temperature: float = 1.0
    ) -> list[Generation]:
        key = self._key(prompt, n, temperature)
        path = self._path(key)

        if path.exists():
            self.hits += 1
            return self._load(path)

        self.misses += 1
        gens = self.backend.generate(prompt, n=n, temperature=temperature)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._save(path, gens)
        return gens

    @staticmethod
    def _save(path: Path, gens: list[Generation]) -> None:
        blob: dict[str, np.ndarray] = {"n": np.array([len(gens)])}
        texts = []
        for i, g in enumerate(gens):
            texts.append(g.text)
            blob[f"lp_{i}"] = g.token_logprobs
            blob[f"ent_{i}"] = g.token_entropies
            if g.content_token_mask is not None:
                blob[f"mask_{i}"] = g.content_token_mask
        # Stored as JSON in a 0-d unicode array rather than an object array, so
        # loading never needs allow_pickle. A cache file should not be able to
        # execute anything.
        blob["texts_json"] = np.array(json.dumps(texts))

        # Atomic write: a half-written cache file that looks valid is worse
        # than no cache at all. Note np.savez appends '.npz' unless handed an
        # open file object, which would break the rename below.
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as fh:
            np.savez_compressed(fh, **blob)
        tmp.replace(path)

    @staticmethod
    def _load(path: Path) -> list[Generation]:
        with np.load(path) as data:
            n = int(data["n"][0])
            texts = json.loads(str(data["texts_json"]))
            gens = []
            for i in range(n):
                mask_key = f"mask_{i}"
                gens.append(
                    Generation(
                        text=texts[i],
                        token_logprobs=data[f"lp_{i}"],
                        token_entropies=data[f"ent_{i}"],
                        content_token_mask=data[mask_key] if mask_key in data else None,
                    )
                )
        return gens

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
        }
