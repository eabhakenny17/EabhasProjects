from pathlib import Path

from .errors import CaseFileError


def load_case(path: Path) -> str:
    try:
        case_text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CaseFileError(f"Could not read case file '{path}': {exc}") from exc

    if not case_text.strip():
        raise CaseFileError(f"Case file '{path}' is empty.")

    return case_text
