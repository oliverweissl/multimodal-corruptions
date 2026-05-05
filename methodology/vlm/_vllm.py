"""Unified vLLM-based VLM wrapper."""

import base64
import logging
import time
from io import BytesIO
from typing import Optional

import requests
from PIL import Image
from vllm import LLM, SamplingParams

from ._base import VLMBase

logger = logging.getLogger(__name__)

_DEFAULT_PORTS = (8700, 8701, 8702, 8703, 8704)
_TIMEOUT = 10  # seconds for health/models check



class VLLMInstance(VLMBase):
    """Base class for all vLLM-hosted vision-language models.

    Set ``MODEL_ID`` in subclasses. Override ``COORD_SCALE`` if the model
    outputs bounding boxes in a normalised coordinate space other than
    auto-detected pixel or 0–1000 space.  Set ``MAX_MODEL_LEN`` to cap the
    context window and reduce VRAM usage for smaller GPUs.

    When a vLLM server is already running for this model, the instance
    automatically uses HTTP (``/v1/completions``) instead of loading the
    model in-process. Subclass ``_messages()`` overrides are always applied.
    """

    MODEL_ID: str = None
    COORD_SCALE: Optional[int] = None
    BBOX_ORDER: str = "xyxy"  # "xyxy" (standard) or "yxyx" (e.g. Gemma outputs y1 x1 y2 x2)
    MAX_MODEL_LEN: Optional[int] = None
    TENSOR_PARALLEL_SIZE: int = 1

    def __init__(
        self,
        seed: int,
        max_new_tokens: int = 1024,
        gpu_memory_utilization: float = 0.8,
        model_id: Optional[str] = None,
    ) -> None:
        """Load the model via vLLM or connect to an existing server.

        :param seed: Random seed forwarded to vLLM for reproducible sampling.
        :param max_new_tokens: Maximum tokens to generate per call.
        :param gpu_memory_utilization: Fraction of GPU memory vLLM may use.
        :param model_id: Override the class-level ``MODEL_ID``.
        :raises ValueError: If no model identifier is set.
        :raises NotImplementedError: If ``COORD_SCALE`` is not set in the subclass.
        """
        model_id = model_id or self.MODEL_ID
        if model_id is None:
            raise ValueError(f"{type(self).__name__}: MODEL_ID not set")
        if self.COORD_SCALE is None:
            raise NotImplementedError(
                f"{type(self).__name__}: COORD_SCALE must be set explicitly. "
                "Inspect the model's bbox output range and set COORD_SCALE in the subclass."
            )

        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._served_url = None

        self.find_served_url()
        if self._served_url is None:  # If no vLLM server is running it, spawn it.
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
        """Build a single-turn message list with image and text.

        :param image: PIL image for the user turn.
        :param prompt: Text prompt string.
        :returns: List containing one user message dict.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": self._pil_to_b64_url(image)}},
                {"type": "text", "text": prompt},
            ],
        }]

    def _post_served(self, image: Image.Image, prompt: str) -> tuple[str, int, float]:
        """Send one request to the running vLLM server using self._messages().

        Uses the subclass ``_messages()`` override so model-specific prompt
        formatting (e.g. DeepSeek grounding tags, Gemma image resize) is applied.

        :param image: PIL image input.
        :param prompt: Text prompt string.
        :returns: Tuple of (response text, token count, runtime seconds).
        """
        payload = {
            "model": self._model_id,
            "messages": self._messages(image, prompt),
            "max_tokens": self._max_new_tokens,
            "temperature": 0,
        }
        t0 = time.time()
        resp = requests.post(self._served_url, json=payload, timeout=300)
        runtime = time.time() - t0
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        count = data["usage"]["completion_tokens"]
        return text, count, runtime

    @staticmethod
    def _pil_to_b64_url(img: Image.Image) -> str:
        """Encode a PIL image as a data-URI string for the vLLM completions API.

        :param img: PIL image to encode.
        :returns: ``data:image/jpeg;base64,...`` URL string.
        """
        buf = BytesIO()
        img.save(buf, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    def find_served_url(self) -> None:
        """Scan common localhost ports for a vLLM server already serving ``model_id``.

        :returns: Base URL of the matching server, or ``None`` if not found.
        """
        for port in _DEFAULT_PORTS:
            url = f"http://localhost:{port}"
            try:
                resp = requests.get(f"{url}/v1/models", timeout=_TIMEOUT)
                if resp.status_code == 200:
                    served_ids = [m["id"] for m in resp.json().get("data", [])]
                    if any(self._model_id in sid or sid in self._model_id for sid in served_ids):
                        logger.info("Detected vLLM server for %s at %s — HTTP mode.", self._model_id, url)
                        self._served_url = f"{url.rstrip('/')}/v1/chat/completions"
                        return
            except requests.exceptions.RequestException:
                continue

    def run_inference(self, image: Image.Image, prompt: str):
        """Run single-sample inference via server or in-process vLLM.

        :param image: PIL image input.
        :param prompt: Text prompt string.
        :returns: Tuple of (response text, token count, raw token count, runtime seconds).
        """
        if self._served_url is not None:
            text, count, runtime = self._post_served(image, prompt)
            return text, count, count, runtime
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
        """Run batched inference via server (sequential) or in-process vLLM (true batch).

        :param images: List of PIL images.
        :param prompts: List of text prompt strings (same length as ``images``).
        :returns: Tuple of (text list, token count list, raw count list, total runtime seconds).
        """
        if self._served_url is not None:
            # Fire all requests concurrently — vLLM server batches them internally.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            t0 = time.time()
            futures_map = {}
            with ThreadPoolExecutor(max_workers=len(images)) as pool:
                for idx, (image, prompt) in enumerate(zip(images, prompts)):
                    futures_map[pool.submit(self._post_served, image, prompt)] = idx
                results = [None] * len(images)
                for fut in as_completed(futures_map):
                    results[futures_map[fut]] = fut.result()
            total_rt = time.time() - t0
            texts = [r[0] for r in results]
            counts = [r[1] for r in results]
            return texts, counts, counts, total_rt
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
