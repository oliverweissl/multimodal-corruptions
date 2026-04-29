import logging
import re

import numpy as np
from image_perturbator import ImagePerturbator
from PIL import Image
from qwen_3_embedding import Qwen3EmbeddingInstance
from text_perturbator import TextPerturbator
from vlm import VLMBase

from config import search as _search
from .utils import (
    compute_mean_iou,
    ensure_rgb,
    extract_json_array,
    extract_target_objects,
)

logger = logging.getLogger(__name__)


def _bbox_list(gt_bboxes: dict) -> list[list[int]]:
    return [[v["xmin"], v["ymin"], v["xmax"], v["ymax"]] for v in gt_bboxes.values()]


def _apply_image_perturbations(
    image_perturbator: ImagePerturbator,
    clean_np: np.ndarray,
    img_scales: np.ndarray,
    bboxes: list[list[int]],
    image_perturbations: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    current_np = ensure_rgb(clean_np).copy()
    applied = {}
    for i, name in enumerate(image_perturbations):
        scale = float(img_scales[i])
        kwargs = {"bboxes": bboxes} if name == "cutout" else {}
        current_np = image_perturbator.apply_perturbation(current_np, name, scale=scale, **kwargs)
        applied[name] = scale
    return ensure_rgb(current_np), applied


def _apply_text_perturbations(
    text_perturbator: TextPerturbator,
    original_prompt: str,
    txt_scales: np.ndarray,
    text_perturbations: list[str],
) -> tuple[str, dict[str, float]]:
    current_prompt = original_prompt
    applied = {}
    for i, name in enumerate(text_perturbations):
        scale = float(txt_scales[i])
        current_prompt = text_perturbator.process_prompt(current_prompt, name, scale=scale)
        applied[name] = scale
    return current_prompt, applied


class FitnessEvaluator:
    def __init__(
        self,
        *,
        vlm: VLMBase,
        seed: int,
        image_perturbations: list[str] = _search.IMAGE_PERTURBATIONS,
        text_perturbations: list[str] = _search.TEXT_PERTURBATIONS,
        mode: str = "multi",
    ) -> None:
        logger.info("Initialising FitnessEvaluator: loading models ...")
        self.image_perturbator = ImagePerturbator()
        self.text_perturbator = TextPerturbator()
        self.vlm = vlm
        self.qwen_emb = Qwen3EmbeddingInstance(seed=seed)
        self.mode = mode
        self.image_perturbations = image_perturbations
        self.text_perturbations = text_perturbations
        self.n_img = len(image_perturbations)
        self.n_txt = len(text_perturbations)
        logger.info("All models loaded.")

    def _expand_genome(self, x: np.ndarray) -> np.ndarray:
        if self.mode == "image":
            return np.concatenate([x, np.zeros(self.n_txt)])
        if self.mode == "text":
            return np.concatenate([np.zeros(self.n_img), x])
        return x

    @staticmethod
    def _normalised_frobenius(clean_f64: np.ndarray, corrupt_np: np.ndarray) -> float:
        corrupt_f64 = corrupt_np.astype(np.float64) / 255.0
        h, w, c = clean_f64.shape
        return np.linalg.norm(clean_f64 - corrupt_f64) / np.sqrt(c * h * w)

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return dot / denom if denom > 0 else 0.0

    @staticmethod
    def _extract_object_list(prompt: str) -> list[str]:
        match = re.search(r'objects "(.*?)"', prompt)
        if match:
            return [item.strip() for item in match.group(1).split(",")]
        return [prompt]

    def _batch_embed(
        self, labels_a: list[str], labels_b_list: list[list[str]]
    ) -> dict[str, np.ndarray]:
        unique = set(labels_a)
        for labels_b in labels_b_list:
            unique.update(labels_b)
        unique_list = list(unique)
        embs, _, _ = self.qwen_emb.run_batch_inference(unique_list)
        return dict(zip(unique_list, embs))

    def _txt_similarity(self, objs_orig, objs_corr, emb_map):
        pair_sims = []
        for orig_label, corr_label in zip(objs_orig, objs_corr):
            if orig_label == corr_label:
                pair_sims.append(1.0)
            else:
                pair_sims.append(self._cosine_similarity(emb_map[orig_label], emb_map[corr_label]))
        return sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

    def evaluate_baseline(self, sample_data):
        orig_w, orig_h = sample_data["orig_dims"]
        prompt = sample_data["original_prompt"]
        response_text, _, _, _ = self.vlm.run_inference(sample_data["clean_image_pil"], prompt)
        parsed_preds = extract_json_array(response_text)
        iou = compute_mean_iou(
            sample_data["gt_bboxes"],
            parsed_preds,
            orig_w,
            orig_h,
            valid_prompt_labels=extract_target_objects(prompt),
        )
        return float(f"{iou:.5f}")

    def evaluate_single(self, x, sample_data):
        """Evaluate one genome vector; used when the metrics cache has no entry.

        :param x: Genome vector of shape (N_VAR,).
        :param sample_data: Sample dict containing the image, prompt, and GT bboxes.
        :returns: Metrics dict with iou, img_dist, txt_sim, vlm outputs, and applied corruptions.
        """
        full_x = self._expand_genome(x)
        clean_np = ensure_rgb(np.array(sample_data["clean_image_pil"]))
        bboxes = _bbox_list(sample_data["gt_bboxes"])

        corrupt_np, applied_img = _apply_image_perturbations(
            self.image_perturbator, clean_np, full_x[: self.n_img], bboxes, self.image_perturbations
        )
        corrupt_pil = Image.fromarray(corrupt_np.astype(np.uint8))

        original_prompt = sample_data["original_prompt"]
        corrupt_prompt, applied_txt = _apply_text_perturbations(
            self.text_perturbator, original_prompt, full_x[self.n_img :], self.text_perturbations
        )

        response_text, token_count, raw_token_count, runtime = self.vlm.run_inference(
            corrupt_pil, corrupt_prompt
        )

        parsed_preds = extract_json_array(response_text)
        orig_w, orig_h = sample_data["orig_dims"]
        iou = compute_mean_iou(
            sample_data["gt_bboxes"],
            parsed_preds,
            orig_w,
            orig_h,
            valid_prompt_labels=extract_target_objects(corrupt_prompt),
        )

        clean_f64 = clean_np.astype(np.float64) / 255.0
        img_dist = self._normalised_frobenius(clean_f64, corrupt_np)

        objs_orig = self._extract_object_list(original_prompt)
        objs_corr = self._extract_object_list(corrupt_prompt)
        emb_map = self._batch_embed(objs_orig, [objs_corr])
        txt_sim = self._txt_similarity(objs_orig, objs_corr, emb_map)

        return {
            "iou": float(f"{iou:.5f}"),
            "img_dist": float(f"{img_dist:.5f}"),
            "txt_sim": float(f"{txt_sim:.5f}"),
            "vlm_response": response_text,
            "vlm_parsed": parsed_preds,
            "corrupt_prompt": corrupt_prompt,
            "token_count": token_count,
            "raw_token_count": raw_token_count,
            "runtime_seconds": float(f"{runtime:.4f}"),
            "applied_img_corruptions": applied_img,
            "applied_txt_corruptions": applied_txt,
        }

    def evaluate_batch(self, xs, sample_data):
        """Evaluate a batch of genome vectors on one sample.

        VLM inference and embedding inference are each reduced to a single
        forward pass for the whole batch.

        :param xs: List of genome vectors, each of shape (N_VAR,).
        :param sample_data: Sample dict containing the image, prompt, and GT bboxes.
        :returns: List of metrics dicts, one per genome, same format as :meth:`evaluate_single`.
        """
        clean_np = ensure_rgb(np.array(sample_data["clean_image_pil"]))
        clean_f64 = clean_np.astype(np.float64) / 255.0
        original_prompt = sample_data["original_prompt"]
        orig_w, orig_h = sample_data["orig_dims"]
        bboxes = _bbox_list(sample_data["gt_bboxes"])

        corrupt_pils, corrupt_nps, corrupt_prompts = [], [], []
        all_applied_img, all_applied_txt = [], []

        for x in xs:
            full_x = self._expand_genome(x)
            corrupt_np, applied_img = _apply_image_perturbations(
                self.image_perturbator,
                clean_np,
                full_x[: self.n_img],
                bboxes,
                self.image_perturbations,
            )
            corrupt_nps.append(corrupt_np)
            corrupt_pils.append(Image.fromarray(corrupt_np.astype(np.uint8)))
            all_applied_img.append(applied_img)

            corrupt_prompt, applied_txt = _apply_text_perturbations(
                self.text_perturbator,
                original_prompt,
                full_x[self.n_img :],
                self.text_perturbations,
            )
            corrupt_prompts.append(corrupt_prompt)
            all_applied_txt.append(applied_txt)

        vlm_responses, vlm_token_counts, vlm_raw_token_counts, vlm_runtime = (
            self.vlm.run_batch_inference(corrupt_pils, corrupt_prompts)
        )

        objs_orig = self._extract_object_list(original_prompt)
        corr_obj_lists = [self._extract_object_list(cp) for cp in corrupt_prompts]
        emb_map = self._batch_embed(objs_orig, corr_obj_lists)

        batch_size = len(xs)
        metrics_list = []
        for idx in range(batch_size):
            parsed_preds = extract_json_array(vlm_responses[idx])
            iou = compute_mean_iou(
                sample_data["gt_bboxes"],
                parsed_preds,
                orig_w,
                orig_h,
                valid_prompt_labels=extract_target_objects(corrupt_prompts[idx]),
            )
            metrics_list.append(
                {
                    "iou": float(f"{iou:.5f}"),
                    "img_dist": float(
                        f"{self._normalised_frobenius(clean_f64, corrupt_nps[idx]):.5f}"
                    ),
                    "txt_sim": float(
                        f"{self._txt_similarity(objs_orig, corr_obj_lists[idx], emb_map):.5f}"
                    ),
                    "vlm_response": vlm_responses[idx],
                    "vlm_parsed": parsed_preds,
                    "corrupt_prompt": corrupt_prompts[idx],
                    "token_count": vlm_token_counts[idx],
                    "raw_token_count": vlm_raw_token_counts[idx],
                    "runtime_seconds": float(f"{vlm_runtime / batch_size:.4f}"),
                    "applied_img_corruptions": all_applied_img[idx],
                    "applied_txt_corruptions": all_applied_txt[idx],
                }
            )

        return metrics_list
