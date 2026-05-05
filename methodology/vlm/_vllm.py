"""
Unified vLLM-based VLM wrapper.

All models load via vLLM — memory management, batching, and flash attention
are handled automatically. No per-model transformers patches needed.

Subclass and set MODEL_ID (and optionally COORD_SCALE / MAX_MODEL_LEN).
"""

import base64
import time
from io import BytesIO
from typing import Optional

from PIL import Image

from ._base import VLMBase


def _pil_to_b64_url(img: Image.Image) -> str:
    """Encode a PIL image as a data-URI string for the vLLM chat API.

    :param img: PIL image to encode.
    :returns: ``data:image/jpeg;base64,...`` URL string.
    """
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


class VLLMInstance(VLMBase):
    """Base class for all vLLM-hosted vision-language models.

    Set ``MODEL_ID`` in subclasses. Override ``COORD_SCALE`` if the model
    outputs bounding boxes in a normalised coordinate space other than
    auto-detected pixel or 0–1000 space.  Set ``MAX_MODEL_LEN`` to cap the
    context window and reduce VRAM usage for smaller GPUs.
    """

    MODEL_ID: str = None
    COORD_SCALE: Optional[int] = None  # None → auto-detect in _metrics.py
    MAX_MODEL_LEN: Optional[int] = None
    TENSOR_PARALLEL_SIZE: int = 1

    def __init__(
        self,
        seed: int,
        max_new_tokens: int = 1024,
        gpu_memory_utilization: float = 0.8,
        model_id: Optional[str] = None,
    ) -> None:
        """Load the model via vLLM.

        :param seed: Random seed forwarded to vLLM for reproducible sampling.
        :param max_new_tokens: Maximum tokens to generate per call.
        :param gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        :param model_id: Override the class-level ``MODEL_ID``.
        :raises ValueError: If no model identifier is set.
        :raises NotImplementedError: If ``COORD_SCALE`` is not set in the subclass.
        """
        from vllm import LLM, SamplingParams

        model_id = model_id or self.MODEL_ID
        if model_id is None:
            raise ValueError(f"{type(self).__name__}: MODEL_ID not set")
        if self.COORD_SCALE is None:
            raise NotImplementedError(
                f"{type(self).__name__}: COORD_SCALE must be set explicitly. "
                "Inspect the model's bbox output range and set COORD_SCALE in the subclass."
            )

        llm_kwargs: dict = dict(
            model=model_id,
            limit_mm_per_prompt={"image": 1},
            max_model_len=4096,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            tensor_parallel_size=self.TENSOR_PARALLEL_SIZE,
        )
        if self.MAX_MODEL_LEN is not None:
            llm_kwargs["max_model_len"] = self.MAX_MODEL_LEN

        self.llm = LLM(**llm_kwargs)
        self.sampling = SamplingParams(max_tokens=max_new_tokens, temperature=0)

    def _messages(self, image: Image.Image, prompt: str) -> list[dict]:
        """Build a single-turn chat message with image and text.

        :param image: PIL image for the user turn.
        :param prompt: Text prompt string.
        :returns: List containing one user message dict.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _pil_to_b64_url(image)}},
                {"type": "text", "text": prompt},
            ],
        }]

    def run_inference(self, image: Image.Image, prompt: str):
        """Run single-sample inference.

        :param image: PIL image input.
        :param prompt: Text prompt string.
        :returns: Tuple of (response text, token count, raw token count, runtime seconds).
        """
        t0 = time.time()
        outputs = self.llm.chat(
            messages=self._messages(image, prompt),
            sampling_params=self.sampling,
        )
        runtime = time.time() - t0
        text = outputs[0].outputs[0].text
        count = len(outputs[0].outputs[0].token_ids)
        return text, count, count, runtime

    def run_batch_inference(self, images: list, prompts: list):
        """Run batched inference in a single vLLM call.

        :param images: List of PIL images.
        :param prompts: List of text prompt strings (same length as ``images``).
        :returns: Tuple of (text list, token count list, raw count list, total runtime seconds).
        """
        t0 = time.time()
        all_messages = [self._messages(img, p) for img, p in zip(images, prompts)]
        all_outputs = self.llm.chat(
            messages=all_messages,
            sampling_params=self.sampling,
        )
        runtime = time.time() - t0
        texts = [o.outputs[0].text for o in all_outputs]
        counts = [len(o.outputs[0].token_ids) for o in all_outputs]
        return texts, counts, counts, runtime
