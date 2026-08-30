"""Make `src/` importable without requiring an editable install.

Keeps `pytest` working straight from a fresh clone, which matters when the
same repo gets pulled onto a rented GPU box.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
