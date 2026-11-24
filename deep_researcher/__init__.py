"""Multi-Agent Deep Researcher Package."""
import sys
from pathlib import Path

# If google.genai is not in current environment, auto-link local .venv if present
try:
    import google.genai
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    venv_site_packages = project_root / ".venv" / "Lib" / "site-packages"
    if venv_site_packages.exists() and str(venv_site_packages) not in sys.path:
        sys.path.insert(0, str(venv_site_packages))

__version__ = "0.1.0"
