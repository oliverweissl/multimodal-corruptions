import re
import logging
import numpy as np
from PIL import Image

from . import _config as _cfg
from .utils import ensure_rgb, extract_json_array, extract_target_objects, compute_mean_iou

from image_perturbator import ImagePerturbator
from text_perturbator import TextPerturbator
from vlm import VLMBase
from qwen_3_embedding import Qwen3EmbeddingInstance

logger = logging.getLogger(__name__)


def _bbox_list(gt_bboxes):
    return [[v["xmin"], v["ymin"], v["xmax"], v["ymax"]] for v in gt_bboxes.values()]


class FitnessEvaluator:
    """
    Initialises all heavy models once and exposes evaluate methods.

    Pass a pre-constructed VLMBase instance via `vlm` to swap in any supported
    vision-language model (Gemma, Kimi-VL, Hunyuan, etc.).  Defaults to
    Qwen3VLInstance when omitted.
    """

    def __init__(self, vlm: VLMBase, seed=42669, mode: str = "multi"):
        logger.info("Initialising FitnessEvaluator: loading models ...")
        self.image_perturbator = ImagePerturbator()
        self.text_perturbator = TextPerturbator()
        self.vlm = vlm
        self.qwen_emb = Qwen3EmbeddingInstance(seed=seed)
        self.mode = mode
        logger.info("All models loaded.")

    def _expand_genome(self, x):
        """Expand a mode-specific genome to full N_VAR length."""
        if self.mode == "image":
            return np.concatenate([x, np.zeros(_cfg.N_TXT)])
        if self.mode == "text":
            return np.concatenate([np.zeros(_cfg.N_IMG), x])
        return x  # multi

    @staticmethod
    def _normalised_frobenius(clean_np, corrupt_np):
        clean = clean_np.astype(np.float64) / 255.0
        corrupt = corrupt_np.astype(np.float64) / 255.0
        diff = clean - corrupt
        h, w = clean.shape[:2]
        c = clean.shape[2]
        return np.linalg.norm(diff) / np.sqrt(c * h * w)

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        dot = np.dot(vec_a, vec_b)
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        return dot / denom if denom > 0 else 0.0

    @staticmethod
    def _extract_object_list(prompt):
        match = re.search(r'objects "(.*?)"', prompt)
        if match:
            return [item.strip() for item in match.group(1).split(",")]
        return [prompt]

    def evaluate_baseline(self, sample_data):
        """
        Run the VLM on the clean image with the original prompt and return IoU.
        Called once per sample before NSGA-II to record baseline performance.
        A low or zero value means the VLM already struggles on the clean input.
        """
        clean_pil = sample_data["clean_image_pil"]
        prompt = sample_data["original_prompt"]
        orig_w, orig_h = sample_data["orig_dims"]

        response_text, _, _, _ = self.vlm.run_inference(clean_pil, prompt)
        parsed_preds = extract_json_array(response_text)
        valid_labels = extract_target_objects(prompt)
        iou = compute_mean_iou(
            sample_data["gt_bboxes"], parsed_preds, orig_w, orig_h,
            valid_prompt_labels=valid_labels,
        )
        return float(f"{iou:.5f}")

    def evaluate_single(self, x, sample_data):
        """
        Evaluate one genome vector. Used for cache-miss re-evaluation
        during Pareto-front saving.
        """
        full_x = self._expand_genome(x)
        img_scales = full_x[:_cfg.N_IMG]
        txt_scales = full_x[_cfg.N_IMG:]

        clean_pil = sample_data["clean_image_pil"]
        clean_np = ensure_rgb(np.array(clean_pil))
        current_np = clean_np.copy()
        bboxes = _bbox_list(sample_data["gt_bboxes"])

        applied_img = {}
        for i, attack_name in enumerate(_cfg.IMAGE_ATTACKS):
            scale = float(img_scales[i])
            kwargs = {"bboxes": bboxes} if attack_name == "cutout" else {}
            current_np = self.image_perturbator.apply_perturbation(
                current_np, attack_name, scale=scale, **kwargs
            )
            current_np = ensure_rgb(current_np)
            applied_img[attack_name] = scale

        corrupt_pil = Image.fromarray(current_np.astype(np.uint8))

        original_prompt = sample_data["original_prompt"]
        current_prompt = original_prompt
        applied_txt = {}
        for i, attack_name in enumerate(_cfg.TEXT_ATTACKS):
            scale = float(txt_scales[i])
            current_prompt = self.text_perturbator.process_prompt(
                current_prompt, attack_name, scale=scale,
            )
            applied_txt[attack_name] = scale

        response_text, token_count, raw_token_count, runtime = self.vlm.run_inference(
            corrupt_pil, current_prompt,
        )

        parsed_preds = extract_json_array(response_text)
        orig_w, orig_h = sample_data["orig_dims"]
        valid_prompt_labels = extract_target_objects(current_prompt)
        iou = compute_mean_iou(
            sample_data["gt_bboxes"], parsed_preds, orig_w, orig_h,
            valid_prompt_labels=valid_prompt_labels,
        )

        img_dist = self._normalised_frobenius(clean_np, current_np)

        objs_orig = self._extract_object_list(original_prompt)
        objs_corr = self._extract_object_list(current_prompt)
        pair_sims = []
        for orig_label, corrupt_label in zip(objs_orig, objs_corr):
            orig_emb, _, _ = self.qwen_emb.run_inference(orig_label)
            corrupt_emb, _, _ = self.qwen_emb.run_inference(corrupt_label)
            pair_sims.append(self._cosine_similarity(orig_emb, corrupt_emb))
        txt_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

        return {
            "iou": float(f"{iou:.5f}"),
            "img_dist": float(f"{img_dist:.5f}"),
            "txt_sim": float(f"{txt_sim:.5f}"),
            "vlm_response": response_text,
            "vlm_parsed": parsed_preds,
            "corrupt_prompt": current_prompt,
            "token_count": token_count,
            "raw_token_count": raw_token_count,
            "runtime_seconds": float(f"{runtime:.4f}"),
            "applied_img_corruptions": applied_img,
            "applied_txt_corruptions": applied_txt,
        }

    def evaluate_batch(self, xs, sample_data):
        """
        Evaluate a batch of genome vectors on a single data sample.

        Parameters
        ----------
        xs : list[ndarray]
            Each element is a genome vector of shape (N_VAR,).
        sample_data : dict
            The sample to evaluate against.

        Returns
        -------
        metrics_list : list[dict]
            One metrics dict per individual, same format as evaluate_single.
        """
        batch_size = len(xs)
        clean_pil = sample_data["clean_image_pil"]
        clean_np = ensure_rgb(np.array(clean_pil))
        original_prompt = sample_data["original_prompt"]
        orig_w, orig_h = sample_data["orig_dims"]
        bboxes = _bbox_list(sample_data["gt_bboxes"])

        corrupt_pils = []
        corrupt_nps = []
        corrupt_prompts = []
        all_applied_img = []
        all_applied_txt = []

        for x in xs:
            full_x = self._expand_genome(x)
            img_scales = full_x[:_cfg.N_IMG]
            txt_scales = full_x[_cfg.N_IMG:]

            current_np = clean_np.copy()
            applied_img = {}
            for i, name in enumerate(_cfg.IMAGE_ATTACKS):
                scale = float(img_scales[i])
                kwargs = {"bboxes": bboxes} if name == "cutout" else {}
                current_np = self.image_perturbator.apply_perturbation(
                    current_np, name, scale=scale, **kwargs
                )
                current_np = ensure_rgb(current_np)
                applied_img[name] = scale

            corrupt_nps.append(current_np)
            corrupt_pils.append(Image.fromarray(current_np.astype(np.uint8)))
            all_applied_img.append(applied_img)

            current_prompt = original_prompt
            applied_txt = {}
            for i, name in enumerate(_cfg.TEXT_ATTACKS):
                scale = float(txt_scales[i])
                current_prompt = self.text_perturbator.process_prompt(
                    current_prompt, name, scale=scale,
                )
                applied_txt[name] = scale

            corrupt_prompts.append(current_prompt)
            all_applied_txt.append(applied_txt)

        vlm_responses, vlm_token_counts, vlm_raw_token_counts, vlm_runtime = (
            self.vlm.run_batch_inference(corrupt_pils, corrupt_prompts)
        )

        # Per-label embedding similarity (iterative; caches across individuals)
        objs_orig = self._extract_object_list(original_prompt)

        orig_emb_cache = {}
        for label in objs_orig:
            if label not in orig_emb_cache:
                orig_emb_cache[label], _, _ = self.qwen_emb.run_inference(label)

        corr_obj_lists = [self._extract_object_list(cp) for cp in corrupt_prompts]
        corr_emb_cache = {}

        metrics_list = []
        for idx in range(batch_size):
            parsed_preds = extract_json_array(vlm_responses[idx])
            valid_prompt_labels = extract_target_objects(corrupt_prompts[idx])
            iou = compute_mean_iou(
                sample_data["gt_bboxes"], parsed_preds, orig_w, orig_h,
                valid_prompt_labels=valid_prompt_labels,
            )
            img_dist = self._normalised_frobenius(clean_np, corrupt_nps[idx])

            objs_corr = corr_obj_lists[idx]
            pair_sims = []
            for orig_label, corrupt_label in zip(objs_orig, objs_corr):
                if orig_label == corrupt_label:
                    pair_sims.append(1.0)
                else:
                    if corrupt_label not in corr_emb_cache:
                        corr_emb_cache[corrupt_label], _, _ = self.qwen_emb.run_inference(corrupt_label)
                    pair_sims.append(
                        self._cosine_similarity(orig_emb_cache[orig_label], corr_emb_cache[corrupt_label])
                    )
            txt_sim = sum(pair_sims) / len(pair_sims) if pair_sims else 0.0

            metrics_list.append({
                "iou": float(f"{iou:.5f}"),
                "img_dist": float(f"{img_dist:.5f}"),
                "txt_sim": float(f"{txt_sim:.5f}"),
                "vlm_response": vlm_responses[idx],
                "vlm_parsed": parsed_preds,
                "corrupt_prompt": corrupt_prompts[idx],
                "token_count": vlm_token_counts[idx],
                "raw_token_count": vlm_raw_token_counts[idx],
                "runtime_seconds": float(f"{vlm_runtime / batch_size:.4f}"),
                "applied_img_corruptions": all_applied_img[idx],
                "applied_txt_corruptions": all_applied_txt[idx],
            })

        return metrics_list