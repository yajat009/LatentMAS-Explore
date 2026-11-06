"""Put architecture/ on the import path, so that experiments can import the upstream modules by bare name."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "architecture"))
