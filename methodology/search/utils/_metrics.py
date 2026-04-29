from .. import _config as _cfg
from ._labels import _are_labels_compatible


def _scale_pred_box(pred_box, target_w, target_h):
    return [
        pred_box[0] * target_w / _cfg.QWEN_SCALE_FACTOR,
        pred_box[1] * target_h / _cfg.QWEN_SCALE_FACTOR,
        pred_box[2] * target_w / _cfg.QWEN_SCALE_FACTOR,
        pred_box[3] * target_h / _cfg.QWEN_SCALE_FACTOR,
    ]


def _calculate_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / float(denom) if denom > 0 else 0.0


def compute_mean_iou(gt_dict, pred_list, ref_w, ref_h, valid_prompt_labels=None):
    if not gt_dict:
        return 0.0
    gt_items = []
    for key, box in gt_dict.items():
        label = key.split("_")[0]
        gt_items.append((label, [box["xmin"], box["ymin"], box["xmax"], box["ymax"]]))

    ious = []
    for gt_label, gt_box in gt_items:
        best_iou = 0.0
        for p in pred_list:
            if not isinstance(p, dict) or "bbox_2d" not in p:
                continue
            p_label = p.get("label", "")
            is_match = _are_labels_compatible(p_label, gt_label)
            if not is_match and valid_prompt_labels:
                for pl in valid_prompt_labels:
                    if _are_labels_compatible(p_label, pl):
                        is_match = True
                        break
            if is_match:
                cur = _calculate_iou(gt_box, _scale_pred_box(p["bbox_2d"], ref_w, ref_h))
                if cur > best_iou:
                    best_iou = cur
        ious.append(best_iou)
    return sum(ious) / len(ious) if ious else 0.0


def _is_perfect(iou, img_dist, txt_sim, iou_max, img_dist_max, txt_sim_min):
    return (
        iou <= iou_max
        and img_dist < img_dist_max
        and txt_sim > txt_sim_min
    )
