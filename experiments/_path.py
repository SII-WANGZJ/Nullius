"""Make the repository root importable without installing the package.

Import this first in every experiment script, so a fresh clone runs with
nothing but the conda environment.
"""

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
