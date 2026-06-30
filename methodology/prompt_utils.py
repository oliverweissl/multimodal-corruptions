import re
from typing import Optional


_PROMPT_OBJECTS_RE = re.compile(r'\bobject(?:\s*\(s\)|s)?\s+"([^"]*)"', re.IGNORECASE)


def extract_prompt_object_text(prompt: str) -> Optional[str]:
    """Return the comma-separated object clause from a detection prompt."""
    if not isinstance(prompt, str) or not prompt:
        return None
    match = _PROMPT_OBJECTS_RE.search(prompt)
    return match.group(1) if match else None


def extract_prompt_objects(prompt: str) -> list[str]:
    """Return object labels from supported detection prompt formats."""
    object_text = extract_prompt_object_text(prompt)
    if object_text is None:
        return []
    return [item.strip() for item in object_text.split(",") if item.strip()]


def replace_prompt_object_text(prompt: str, replacement: str) -> Optional[str]:
    """Replace only the quoted object clause in a supported detection prompt."""
    if not isinstance(prompt, str) or not prompt:
        return None
    match = _PROMPT_OBJECTS_RE.search(prompt)
    if not match:
        return None
    return f"{prompt[:match.start(1)]}{replacement}{prompt[match.end(1):]}"
