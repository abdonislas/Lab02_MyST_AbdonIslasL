"""Hace importable el paquete src/ al correr pytest desde cualquier forma."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
