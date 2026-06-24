"""Shared computation and plot-setup utilities for RQ notebooks."""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.ndimage
import seaborn as sns
from PIL import Image
import json
import os
from xml.etree import ElementTree
from scipy import stats

from ._loader import IMG_CORRUPTIONS, RESULTS_ROOT, TXT_CORRUPTIONS

IMG_COLS = [f"img_{c}" for c in IMG_CORRUPTIONS]
TXT_COLS = [f"txt_{c}" for c in TXT_CORRUPTIONS]
ALL_CORRUPT_COLS = IMG_COLS + TXT_COLS

PALETTE = sns.color_palette("tab10")

SPLIT_ORDER = [
    ('multimodal', 'multi',  'multi'),
    ('multimodal', 'multi',  'single/multi'),
    ('multimodal', 'multi',  'single/solo'),
    ('unimodal',   'image',  'multi'),
    ('unimodal',   'image',  'single/multi'),
    ('unimodal',   'image',  'single/solo'),
    ('unimodal',   'text',   'multi'),
    ('unimodal',   'text',   'single/multi'),
    ('unimodal',   'text',   'single/solo'),
]

SPLIT_LABEL = {
    ('multimodal', 'multi',  'multi'):         'multimodal-multi',
    ('multimodal', 'multi',  'single/multi'):  'multimodal-single-multi',
    ('multimodal', 'multi',  'single/solo'):   'multimodal-single-solo',
    ('unimodal',   'image',  'multi'):         'image-multi',
    ('unimodal',   'image',  'single/multi'):  'image-single-multi',
    ('unimodal',   'image',  'single/solo'):   'image-single-solo',
    ('unimodal',   'text',   'multi'):         'text-multi',
    ('unimodal',   'text',   'single/multi'):  'text-single-multi',
    ('unimodal',   'text',   'single/solo'):   'text-single-solo',
}


# ── Display helpers ───────────────────────────────────────────────────────────

def tex(s: str) -> str:
    """Escape a plain string for LaTeX text mode (underscores → spaces)."""
    return s.replace("_", " ")


# ── Matplotlib setup ──────────────────────────────────────────────────────────

def setup_matplotlib() -> None:
    """Configure matplotlib: LaTeX rendering, serif font, 18 pt."""
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.family": "serif",
            "font.size": 18,
            "axes.titlesize": 20,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "figure.titlesize": 22,
        }
    )


# ── Attack-success metrics ────────────────────────────────────────────────────

THRESHOLDS = np.linspace(0.0, 1.0, 201)


def asr_curve(sub_df: pd.DataFrame,
              thresholds: Optional[np.ndarray] = None) -> np.ndarray:
    """ASR at each IoU threshold."""
    t = thresholds if thresholds is not None else THRESHOLDS
    vals = sub_df["final_iou"].values
    return np.array([(vals <= v).mean() for v in t])


# ── Text metrics ──────────────────────────────────────────────────────────────

def normalised_edit_distance(s1: str, s2: str) -> float:
    """NED = 1 − SequenceMatcher ratio (character level)."""
    if not s1 and not s2:
        return 0.0
    return 1.0 - difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'ned' column to df for text-corrupted rows."""
    df = df.copy()
    ned = []
    for _, row in df.iterrows():
        orig = row.get("original_prompt", "") or ""
        pert = row.get("perturbed_prompt", "") or ""
        if not pert or pert.startswith("Error:"):
            ned.append(np.nan)
        else:
            ned.append(normalised_edit_distance(orig, pert))
    df["ned"] = ned
    return df

# ── Image metrics — MS-SSIM ───────────────────────────────────────────────────

def _ssim_scale(a: np.ndarray, b: np.ndarray,
                k1: float = 0.01, k2: float = 0.03,
                sigma: float = 1.5) -> float:
    c1, c2 = k1 ** 2, k2 ** 2
    def f(x): return scipy.ndimage.gaussian_filter(x, sigma, truncate=5.5)
    mu1, mu2 = f(a), f(b)
    s1 = f(a * a) - mu1 ** 2
    s2 = f(b * b) - mu2 ** 2
    s12 = f(a * b) - mu1 * mu2
    lum  = (2 * mu1 * mu2 + c1) / (mu1 ** 2 + mu2 ** 2 + c1)
    cont = (2 * s12 + c2) / (s1 + s2 + c2)
    return float((lum * cont).mean())


def ms_ssim(img_ref: Image.Image, img_dis: Image.Image,
            n_scales: int = 5) -> float:
    """Multi-Scale SSIM (Wang et al. 2003)."""
    w = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333][:n_scales]
    w = [v / sum(w) for v in w]

    def lum(img: Image.Image) -> np.ndarray:
        a = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        return 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]

    a = lum(img_ref)
    h, ww = a.shape
    b = lum(img_dis.resize((ww, h), Image.BILINEAR))
    score = 1.0
    for s in range(n_scales):
        if a.shape[0] < 12 or a.shape[1] < 12:
            break
        score *= _ssim_scale(a, b) ** w[s]
        a = scipy.ndimage.zoom(a, 0.5, order=1)
        b = scipy.ndimage.zoom(b, 0.5, order=1)
    return float(score)


def compute_image_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute LPIPS and MS-SSIM for all rows. Adds lpips/ms_ssim/ms_ssim_dist cols."""
    df = df.copy()
    ms = []
    for _, row in df.iterrows():
        orig = RESULTS_ROOT.parent / row["_orig_img_folder"] / "data_point.JPEG"
        if not orig.exists():
            orig = orig.with_suffix(".jpg")
        pth = Path(row["_best_img_path"])
        if not orig.exists() or not pth.exists():
            ms.append(np.nan)
            continue
        try:
            ref = Image.open(orig)
            dis = Image.open(pth)
            ms.append(ms_ssim(ref, dis))
        except Exception:
            ms.append(np.nan)
    df["ms_ssim"] = ms
    df["ms_ssim_dist"] = 1.0 - df["ms_ssim"]
    return df


# ── Corruption analysis ───────────────────────────────────────────────────────

def corruption_importance(sub: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Spearman $\\rho$ of each corruption intensity vs iou_reduction."""
    y = sub["iou_reduction"].values
    rows = []
    for col in cols:
        x = sub[col].values
        rho, pval = (0.0, 1.0) if np.std(x) < 1e-10 else stats.spearmanr(x, y)
        rows.append(
            {
                "corruption": tex(col.removeprefix("img_").removeprefix("txt_")),
                "rho": rho,
                "pval": pval,
                "ctype": "img" if col in IMG_COLS else "txt",
            }
        )
    return pd.DataFrame(rows).sort_values("rho", ascending=False)


def extract_classes(row: pd.Series) -> list[str]:
    """Parse object class labels from original_prompt."""
    m = re.search(r'Detect the object\(s\) "([^"]+)"',
                  row.get("original_prompt", "") or "")
    return [lbl.strip() for lbl in m.group(1).split(",")] if m else []


# ── RQ3 — Validity ───────────────────────────────────────────────────────────

_SCENE_MAP = {
    "single/solo":  "Isolated",
    "single/multi": "Clustered",
    "multi":        "Mixed",
}

_REJECT_REASON_CODE = {
    "unclear_image": 0,
    "unclear_label": 1,
}


def _xml_gt_boxes(xml_dir: Path, img_fn: str) -> list:
    """Return list of [xmin, ymin, xmax, ymax] from ILSVRC XML for *img_fn*."""
    stem     = os.path.splitext(img_fn)[0]
    xml_path = xml_dir / (stem + ".xml")
    if not xml_path.exists():
        return []
    try:
        root_el = ElementTree.parse(str(xml_path)).getroot()
    except Exception:
        return []
    boxes = []
    for obj in root_el.findall("object"):
        bb = obj.find("bndbox")
        boxes.append([
            int(bb.find("xmin").text), int(bb.find("ymin").text),
            int(bb.find("xmax").text), int(bb.find("ymax").text),
        ])
    return boxes


def _fmt(s: pd.Series) -> str:
    return f"{s.mean():.3f} $\\pm$ {s.std():.3f}"


def _bbox_iou(a: list, b: list) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter  = max(0, xb - xa) * max(0, yb - ya)
    union  = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _parse_prompt_labels(prompt: str) -> list[str]:
    m = re.search(r'objects\s+"(.*?)"', prompt)
    if not m:
        m = re.search(r'"([^"]+)"', prompt)
    return [l.strip() for l in m.group(1).split(",")] if m else []


def load_rq3_data(root: str | Path) -> pd.DataFrame:
    """Load survey + GT and return one row per (image, session).

    GT source: ILSVRC XML annotations matched by survey image filename.
    Survey folder IDs are from a separate experiment (multimodal/variant_a/b,
    now deleted) and do NOT correspond to results/selection or results/qwen IDs.
    IoU uses max-over-all-GT-bboxes (best-match) because the original
    label→GT mapping requires the deleted variant_a/b result files.

    Images without XML annotations are excluded.

    Columns
    -------
    filename, scene_type, session, method,
    human_iou    : mean IoU vs GT for bbox labels only (reject labels skipped)
    n_labels     : number of bbox labels
    rejected     : True if every label in this session was rejected
    reject_reason: 0=unclear_image, 1=unclear_label, NaN if not rejected
                   (unclear_image takes priority over unclear_label)
    bbox_by_idx  : {label_index: [x1,y1,x2,y2]} for bbox labels
    """
    root    = Path(root)
    ann_dir = root / "dataset" / "2017" / "ILSVRC" / "Annotations" / "DET" / "val"

    with open(root / "analysis" / "survey.json") as f:
        survey = json.load(f)

    records = []
    for img_fn, img_data in survey.items():
        cat_path   = img_data["internal_mapping"].rsplit("/", 1)[0]
        scene_type = _SCENE_MAP.get(cat_path, "Unknown")

        gt_boxes = _xml_gt_boxes(ann_dir, img_fn)
        if not gt_boxes:
            continue

        for ann in img_data["annotations"]:
            label_ious:   list[float] = []
            bbox_by_idx:  dict        = {}
            any_reject    = False
            all_reject    = True
            has_img_rej   = False   # any label rejected as unclear_image

            for lbl in ann["labels"]:
                resp = lbl["response_type"]
                rej  = lbl.get("reject_reason")
                idx  = lbl["label_index"]

                if resp == "reject" and rej not in ("unclear_image", "unclear_label"):
                    continue  # skip non-standard rejects
                if resp == "reject":
                    label_ious.append(0.0)
                    any_reject = True
                    if rej == "unclear_image":
                        has_img_rej = True
                    continue

                all_reject = False
                hbs = [[b["xmin"], b["ymin"], b["xmax"], b["ymax"]]
                       for b in lbl.get("bboxes", [])]
                if not hbs:
                    continue
                bbox_by_idx[idx] = hbs[0]
                iou_val = max(_bbox_iou(gt, hb) for gt in gt_boxes for hb in hbs)
                label_ious.append(iou_val)

            if not label_ious:
                continue

            # reject_reason code: unclear_image=0 takes priority over unclear_label=1
            if all_reject and any_reject:
                rej_code = 0 if has_img_rej else 1
            else:
                rej_code = np.nan

            records.append(dict(
                filename      = img_fn,
                scene_type    = scene_type,
                session       = ann["session"],
                method        = ann.get("method", ""),
                human_iou     = float(np.mean([v for v in label_ious if not np.isnan(v)])),
                n_labels      = len([v for v in label_ious if not np.isnan(v)]),
                rejected      = bool(all_reject and any_reject),
                reject_reason = rej_code,
                bbox_by_idx   = bbox_by_idx,
            ))

    return pd.DataFrame(records)


_OBJ_CAT_TO_SCENE = {
    "single/solo":  "Isolated",
    "single/multi": "Clustered",
    "multi":        "Mixed",
}


def rq3_validity_table(human_df: pd.DataFrame,
                        vlm_df:   pd.DataFrame,
                        scene_order: list[str] | None = None) -> pd.DataFrame:
    """Validity summary table: accept rate + human IoU + VLM IoU by scene.

    Parameters
    ----------
    human_df : output of load_rq3_data()
    vlm_df   : output of success_only(load_all_results()) filtered to the
               model(s) and modality of interest (used for post-attack VLM IoU)
    """
    if scene_order is None:
        scene_order = ["Isolated", "Clustered", "Mixed"]

    # VLM IoU per scene (map obj_category → scene label)
    vlm_df = vlm_df.copy()
    vlm_df["scene_type"] = vlm_df["obj_category"].map(_OBJ_CAT_TO_SCENE)
    vlm_scene = (vlm_df.groupby("scene_type")["final_iou"]
                        .agg(["mean", "std"])
                        .rename(columns={"mean": "vlm_mean", "std": "vlm_std"}))
    vlm_overall_mean = vlm_df["final_iou"].mean()
    vlm_overall_std  = vlm_df["final_iou"].std()

    rows = []
    for sc in scene_order:
        sub      = human_df if sc == "Overall" else human_df[human_df.scene_type == sc]
        img_bbox = sub[~sub.rejected].groupby("filename")["human_iou"].mean()
        accept   = 1.0 - sub["rejected"].mean()

        if sc == "Overall":
            vlm_str = f"{vlm_overall_mean:.3f} $\\pm$ {vlm_overall_std:.3f}"
        elif sc in vlm_scene.index:
            r = vlm_scene.loc[sc]
            vlm_str = f"{r.vlm_mean:.3f} $\\pm$ {r.vlm_std:.3f}"
        else:
            vlm_str = "—"

        rows.append({
            "Scene":                   sc,
            "N sessions":              len(sub),
            "Accept rate":             f"{accept:.1%}",
            "Human IoU (bbox, mean $\\pm$ std)": _fmt(img_bbox),
            "VLM IoU (post-attack, mean $\\pm$ std)":    vlm_str,
        })

    return pd.DataFrame(rows).set_index("Scene")


_GENOME_MODE_LABEL = {
    ("multimodal", "multi"):  "Multimodal (img+txt)",
    ("unimodal",   "image"):  "Image-only",
    ("unimodal",   "text"):   "Text-only",
}


def rq3_validity_by_modality(human_df: pd.DataFrame,
                              vlm_df:   pd.DataFrame) -> pd.DataFrame:
    """Validity table grouped by attack modality instead of scene type.

    VLM IoU is broken down by genome_mode (multimodal / image-only / text-only).
    Human data (accept rate, IoU) is only available for the multimodal row because
    the human evaluation survey was conducted exclusively on multimodal
    (variant_a/b) attacks; the source files for image-only and text-only
    conditions no longer exist, so those rows show '---' for human columns.

    Parameters
    ----------
    human_df : output of load_rq3_data()
    vlm_df   : output of success_only(load_all_results()), all models and modalities
    """
    human_bbox   = human_df[~human_df.rejected].groupby("filename")["human_iou"].mean()
    human_accept = 1.0 - human_df["rejected"].mean()
    human_str    = _fmt(human_bbox)
    accept_str   = f"{human_accept:.1%}"

    rows = []
    for (modality, gmode), label in _GENOME_MODE_LABEL.items():
        sub = vlm_df[(vlm_df["modality"] == modality) & (vlm_df["genome_mode"] == gmode)]
        if len(sub) == 0:
            continue
        # Human data only exists for multimodal attacks
        is_multimodal = (modality == "multimodal")
        rows.append({
            "Manipulation":                          label,
            "N (VLM)":                               len(sub),
            "Accept rate":                           accept_str   if is_multimodal else "---",
            "Human IoU (bbox, mean $\\pm$ std)":     human_str    if is_multimodal else "---",
            "VLM IoU (post-attack, mean $\\pm$ std)": f"{sub['final_iou'].mean():.3f} $\\pm$ {sub['final_iou'].std():.3f}",
        })

    return pd.DataFrame(rows).set_index("Manipulation")


def rq3_validity_by_model(human_df: pd.DataFrame,
                           vlm_df:   pd.DataFrame,
                           model_label: dict | None = None) -> pd.DataFrame:
    """Validity table grouped by VLM model.

    Parameters
    ----------
    human_df    : output of load_rq3_data()
    vlm_df      : output of success_only(load_all_results()), all models
    model_label : optional display-name mapping {raw_name: label}
    """
    human_bbox   = human_df[~human_df.rejected].groupby("filename")["human_iou"].mean()
    human_accept = 1.0 - human_df["rejected"].mean()
    human_str    = _fmt(human_bbox)
    accept_str   = f"{human_accept:.1%}"

    if model_label is None:
        model_label = {}

    rows = []
    for model in sorted(vlm_df["model"].unique()):
        sub = vlm_df[vlm_df["model"] == model]
        rows.append({
            "Model":                                 model_label.get(model, tex(model)),
            "N (VLM)":                               len(sub),
            "Accept rate":                           accept_str,
            "Human IoU (bbox, mean $\\pm$ std)":     human_str,
            "VLM IoU (post-attack, mean $\\pm$ std)": f"{sub['final_iou'].mean():.3f} $\\pm$ {sub['final_iou'].std():.3f}",
        })

    return pd.DataFrame(rows).set_index("Model")


def _class_stats(g: pd.DataFrame) -> pd.Series:
    suc = g[g["status"] == "success"]
    return pd.Series({
        "n_total":            len(g),
        "baseline_fail_rate": (g["status"] == "baseline_fail").mean(),
        "asr_25":             (suc["final_iou"] <= 0.25).mean() if len(suc) else np.nan,
        "asr_50":             (suc["final_iou"] <= 0.50).mean() if len(suc) else np.nan,
        "med_iou_reduction":  suc["iou_reduction"].median(),
        "med_baseline_iou":   g["baseline_iou"].median(),
    })


def compute_class_stats(df_all: pd.DataFrame,
                         min_cases: int = 3) -> pd.DataFrame:
    """Per-class vulnerability stats (ASR + baseline-fail rate)."""
    rows = []
    for _, row in df_all.iterrows():
        for cls in extract_classes(row):
            rows.append({
                "cls":           cls,
                "status":        row["status"],
                "final_iou":     row["final_iou"],
                "iou_reduction": row["iou_reduction"],
                "baseline_iou":  row["baseline_iou"],
            })
    cdf = pd.DataFrame(rows)
    out = cdf.groupby("cls").apply(_class_stats).reset_index()
    return out[out["n_total"] >= min_cases].sort_values("asr_25", ascending=False)
