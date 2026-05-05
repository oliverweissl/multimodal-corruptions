from typing import Optional

from ._labels import _are_labels_compatible

_BBOX_KEYS = ("bbox", "bbox_2d", "bounding_box", "box")
_LABEL_KEYS = ("label", "object", "class", "name", "category")


def _extract_bbox(pred: dict) -> Optional[list]:
    """Return the bounding box list from a prediction dict, trying multiple key names.

    :param pred: Prediction dict from VLM output.
    :returns: Bounding box as ``[x1, y1, x2, y2]``, or ``None`` if no bbox key found.
    """
    for key in _BBOX_KEYS:
        if key in pred:
            return pred[key]
    return None


def _extract_label(pred: dict) -> str:
    """Return the label string from a prediction dict, trying multiple key names.

    :param pred: Prediction dict from VLM output.
    :returns: Label string, or empty string if no label key found.
    """
    for key in _LABEL_KEYS:
        if key in pred:
            return str(pred[key])
    return ""


def _to_pixel_box(
    bbox: list[float], ref_w: int, ref_h: int, coord_scale: Optional[int] = None,
    bbox_order: str = "xyxy",
) -> list[float]:
    """Normalise a bounding box to pixel coordinates regardless of input scale.

    Three cases are handled:
    - Fractional [0, 1] coordinates → multiply by image dimensions.
    - Pixel coordinates (fit within image bounds) → used as-is.
    - Normalised coordinates → rescaled using ``coord_scale`` if provided,
      otherwise the scale is inferred as the nearest power of ten.

    :param bbox: Bounding box in the model's native order.
    :param ref_w: Reference image width in pixels.
    :param ref_h: Reference image height in pixels.
    :param coord_scale: Model's internal coordinate scale (e.g. 1000 for Qwen, 896 for Gemma).
        When ``None`` the scale is inferred automatically.
    :param bbox_order: Axis order of the bbox — ``"xyxy"`` (standard) or ``"yxyx"``
        (e.g. Gemma outputs ``[y1, x1, y2, x2]``).
    :returns: Bounding box in ``[x1, y1, x2, y2]`` pixel coordinates.
    """
    a, b, c, d = bbox
    if bbox_order == "yxyx":
        x1, y1, x2, y2 = b, a, d, c  # swap: [y1,x1,y2,x2] → [x1,y1,x2,y2]
    else:
        x1, y1, x2, y2 = a, b, c, d
    scale = coord_scale or 1.0
    return [x1 * ref_w / scale, y1 * ref_h / scale, x2 * ref_w / scale, y2 * ref_h / scale]


def _calculate_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute the Intersection over Union (IoU) for two axis-aligned bounding boxes.

    :param box_a: First box as ``[x1, y1, x2, y2]``.
    :param box_b: Second box as ``[x1, y1, x2, y2]``.
    :returns: IoU score in [0.0, 1.0].
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / float(denom) if denom > 0 else 0.0


def compute_mean_iou(
    gt_dict: dict,
    pred_list: list,
    ref_w: int,
    ref_h: int,
    valid_prompt_labels: Optional[list[str]] = None,
    coord_scale: Optional[int] = None,
    bbox_order: str = "xyxy",
) -> float:
    """Compute mean IoU between ground-truth boxes and VLM predictions.

    :param gt_dict: Ground-truth dict ``{label_key: {xmin, ymin, xmax, ymax}}``.
    :param pred_list: List of prediction dicts from VLM output.
    :param ref_w: Image width in pixels (used for coordinate normalisation).
    :param ref_h: Image height in pixels (used for coordinate normalisation).
    :param valid_prompt_labels: Optional label whitelist for relaxed label matching.
    :param coord_scale: Model's internal coordinate scale; passed to :func:`_to_pixel_box`.
    :param bbox_order: Axis order of predicted bboxes — ``"xyxy"`` or ``"yxyx"``.
    :returns: Mean IoU across all GT objects; 0.0 if no GT or no matches.
    """
    if not gt_dict:
        return 0.0
    gt_items = [
        (key.split("_")[0], [box["xmin"], box["ymin"], box["xmax"], box["ymax"]])
        for key, box in gt_dict.items()
    ]

    ious = []
    for gt_label, gt_box in gt_items:
        best_iou = 0.0
        for p in pred_list:
            if not isinstance(p, dict):
                continue
            bbox = _extract_bbox(p)
            if bbox is None:
                continue
            p_label = _extract_label(p)
            is_match = _are_labels_compatible(p_label, gt_label)
            if not is_match and valid_prompt_labels:
                for pl in valid_prompt_labels:
                    if _are_labels_compatible(p_label, pl):
                        is_match = True
                        break
            if is_match:
                cur = _calculate_iou(gt_box, _to_pixel_box(bbox, ref_w, ref_h, coord_scale, bbox_order))
                if cur > best_iou:
                    best_iou = cur
        ious.append(best_iou)
    return sum(ious) / len(ious) if ious else 0.0


def is_perfect(
    iou: float,
    img_dist: float,
    txt_sim: float,
    iou_max: float,
    img_dist_max: float,
    txt_sim_min: float,
) -> bool:
    """Return ``True`` when a test case satisfies all early-stop thresholds.

    :param iou: Mean IoU score (lower is a stronger perturbation).
    :param img_dist: Normalised Frobenius image distance.
    :param txt_sim: Text cosine similarity to original prompt.
    :param iou_max: Maximum IoU allowed for a perfect test case.
    :param img_dist_max: Maximum image distance allowed.
    :param txt_sim_min: Minimum text similarity required.
    :returns: ``True`` if all thresholds are satisfied simultaneously.
    """
    return iou <= iou_max and img_dist < img_dist_max and txt_sim > txt_sim_min
