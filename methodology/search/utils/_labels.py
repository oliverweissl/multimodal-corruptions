import difflib
import json
import re
from typing import Optional


def _bbox_from_dict(d: dict) -> Optional[list]:
    """Convert a coord-keyed dict to a ``[x1, y1, x2, y2]`` list.

    :param d: Dict with corner-coordinate keys (e.g. ``x1/y1/x2/y2`` or ``xmin/ymin/xmax/ymax``).
    :returns: Coordinate list, or ``None`` if the dict lacks recognised keys.
    """
    if "x1" in d and "y1" in d and "x2" in d and "y2" in d:
        return [d["x1"], d["y1"], d["x2"], d["y2"]]
    if "xmin" in d and "ymin" in d and "xmax" in d and "ymax" in d:
        return [d["xmin"], d["ymin"], d["xmax"], d["ymax"]]
    return None


def _dict_to_list(obj: dict) -> list:
    """Convert a ``{label: [{coord-dict}]}`` response dict to a flat list of prediction dicts.

    :param obj: Outer dict where keys are label strings and values are lists of bbox dicts.
    :returns: List of ``{"label": ..., "bbox_2d": [...]}`` dicts.
    """
    result = []
    for label, entries in obj.items():
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, list):
                result.append({"label": label, "bbox_2d": entry})
            elif isinstance(entry, dict):
                bbox = _bbox_from_dict(entry)
                if bbox:
                    result.append({"label": label, "bbox_2d": bbox})
    return result


def extract_json_array(pred_str: str) -> list:
    """Extract and normalise a bounding-box prediction list from a free-form VLM response.

    Handles three common output formats:
    - Array of dicts: ``[{"label": ..., "bbox_2d": [...]}, ...]``
    - Dict of label→bbox-list: ``{"dog": [{"x1": ..., "y1": ..., "x2": ..., "y2": ...}]}``
    - Mixed/markdown-wrapped variants of the above.

    :param pred_str: Raw VLM output text.
    :returns: Parsed list of prediction dicts, or an empty list if parsing fails.
    """
    text = pred_str.strip()

    # Try array format first
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Try outer dict format: {"label": [{...}], ...}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return _dict_to_list(obj)
        except json.JSONDecodeError:
            pass

    return []


def extract_target_objects(prompt: str) -> list[str]:
    """Parse the ``objects "..."`` segment of a prompt and return normalised label strings.

    :param prompt: Prompt string possibly containing an ``objects "..."`` clause.
    :returns: List of normalised label strings, or an empty list if the pattern is absent.
    """
    if not prompt:
        return []
    match = re.search(r'objects "(.*?)"', prompt)
    if match:
        return [_normalize_label(x) for x in match.group(1).split(",")]
    return []


def _normalize_label(text: str) -> str:
    """Strip whitespace and zero-width Unicode characters, then lowercase.

    :param text: Raw label string.
    :returns: Normalised label string.
    """
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s​‍]+", "", text).lower()


def _are_labels_compatible(pred_label: str, gt_label: str) -> bool:
    """Return ``True`` if a predicted label is an acceptable match for a ground-truth label.

    :param pred_label: Label string from the VLM prediction.
    :param gt_label: Ground-truth label string.
    :returns: ``True`` if labels are equal, anagram-equal, or have sequence similarity >= 0.75.
    """
    p = _normalize_label(pred_label)
    g = _normalize_label(gt_label)
    if p == g:
        return True
    if sorted(list(p)) == sorted(list(g)):
        return True
    if difflib.SequenceMatcher(None, p, g).ratio() >= 0.75:
        return True
    return False
