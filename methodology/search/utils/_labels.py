import difflib
import json
import re


def extract_json_array(pred_str: str) -> list:
    text = pred_str.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []


def extract_target_objects(prompt: str) -> list[str]:
    if not prompt:
        return []
    match = re.search(r'objects "(.*?)"', prompt)
    if match:
        return [_normalize_label(x) for x in match.group(1).split(",")]
    return []


def _normalize_label(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s​‍]+", "", text).lower()


def _are_labels_compatible(pred_label: str, gt_label: str) -> bool:
    p = _normalize_label(pred_label)
    g = _normalize_label(gt_label)
    if p == g:
        return True
    if sorted(list(p)) == sorted(list(g)):
        return True
    if difflib.SequenceMatcher(None, p, g).ratio() >= 0.75:
        return True
    return False
