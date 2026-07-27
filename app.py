"""Root entry point for the STX-LBD Explorer."""

from pathlib import Path
import runpy


APP_PATH = (
    Path(__file__).resolve().parent
    / "scripts"
    / "proof_of_concept"
    / "app.py"
)

if not APP_PATH.exists():
    raise FileNotFoundError(
        f"Streamlit application not found: {APP_PATH}"
    )

runpy.run_path(str(APP_PATH), run_name="__main__")
