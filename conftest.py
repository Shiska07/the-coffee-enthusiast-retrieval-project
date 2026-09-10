"""Make the `src/` layout importable during tests without requiring an
editable install. `pip install -e .` also works and is preferred for running
the scripts in `scripts/` outside pytest.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
