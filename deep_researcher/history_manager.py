"""Session history manager for saving and loading past research reports."""
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from deep_researcher.models.schemas import ResearchBrief

HISTORY_DIR = Path("history")


def _slugify(text: str) -> str:
    """Converts topic text into a filesystem-safe slug."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "_", text)[:40]


def save_session(brief: ResearchBrief) -> Dict[str, str]:
    """
    Saves a research session as both a Markdown brief and a structured JSON artifact.
    Returns paths to the saved files.
    """
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(brief.topic)
    base_name = f"{timestamp}_{slug}"

    md_path = HISTORY_DIR / f"{base_name}.md"
    json_path = HISTORY_DIR / f"{base_name}.json"

    # Write Markdown
    md_path.write_text(brief.full_markdown, encoding="utf-8")

    # Write JSON metadata
    data = brief.model_dump()
    data["saved_at"] = datetime.now().isoformat()
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "md_path": str(md_path),
        "json_path": str(json_path),
        "base_name": base_name,
    }


def list_sessions() -> List[Dict[str, Any]]:
    """Returns a list of saved research sessions ordered by most recent first."""
    if not HISTORY_DIR.exists():
        return []

    sessions = []
    for json_file in sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            saved_at = data.get("saved_at", "")
            try:
                dt = datetime.fromisoformat(saved_at)
                display_time = dt.strftime("%b %d, %H:%M")
            except Exception:
                display_time = json_file.stem[:15]

            sessions.append({
                "id": json_file.stem,
                "topic": data.get("topic", "Untitled Research"),
                "display_label": f"{display_time} - {data.get('topic', 'Untitled')[:45]}...",
                "json_path": str(json_file),
                "md_path": str(json_file.with_suffix(".md")),
                "sources_count": len(data.get("bibliography", [])),
                "claims_count": len(data.get("sections", [])),
                "saved_at": saved_at,
            })
        except Exception:
            continue

    return sessions


def load_session(json_path: str) -> Optional[Dict[str, Any]]:
    """Loads and returns the complete research session data from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
