"""Make ``src/`` importable without installing the package.

Import this first in every experiment script, so the repository runs from a
fresh clone with nothing but the conda environment.
"""

import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
