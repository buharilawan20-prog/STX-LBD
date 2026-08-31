"""Root entry point for the STX-LBD Explorer."""

from pathlib import Path
import runpy
import sys


APP_DIR = (
    Path(__file__).resolve().parent
    / "scripts"
    / "proof_of_concept"
)

APP_PATH = APP_DIR / "app.py"

if not APP_PATH.exists():
    raise FileNotFoundError(
        f"Streamlit application not found: {APP_PATH}"
    )

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

runpy.run_path(str(APP_PATH), run_name="__main__")
