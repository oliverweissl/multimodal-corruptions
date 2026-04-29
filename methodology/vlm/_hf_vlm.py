"""
Semi-abstract base for HuggingFace-hosted vision-language models.

Minimum to add a new model:

    class MyVLM(HuggingFaceVLM):
        MODEL_ID = "org/model-name"

Override any of the three hook methods when the model deviates from the
standard HuggingFace VLM pattern:

    _build_message(image, prompt)   - chat message list (default: image placeholder + text)
    _prepare_single(image, prompt)  - tokenise one sample  (default: processor(text, images))
    _prepare_batch(images, prompts) - tokenise a batch     (default: processor(texts, images))
"""

import random
import time
import logging

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from ._base import VLMBase

logger = logging.getLogger(__name__)


class HuggingFaceVLM(VLMBase):
    MODEL_ID: str = None             # required — set in subclass or pass model_id to __init__
    MODEL_CLASS = None               # None → AutoModelForCausalLM
    DTYPE = torch.bfloat16
    ATTN_IMPL: str = "flash_attention_2"   # set None to disable
    TRUST_REMOTE_CODE: bool = False

    def __init__(
        self,
        seed: int,
        max_new_tokens: int = 1024,
        device: str = "gpu",
        model_id: str = None,
    ):
        self.max_new_tokens = max_new_tokens
        self.seed = seed

        model_id = model_id or self.MODEL_ID
        if model_id is None:
            raise ValueError(f"{type(self).__name__}: MODEL_ID not set")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = (
            torch.device("cuda")
            if device.lower() == "gpu" and torch.cuda.is_available()
            else torch.device("cpu")
        )

        load_kwargs: dict = {"torch_dtype": self.DTYPE}
        if self.ATTN_IMPL:
            load_kwargs["attn_implementation"] = self.ATTN_IMPL
        if self.TRUST_REMOTE_CODE:
            load_kwargs["trust_remote_code"] = True
        if self.device.type == "cuda":
            load_kwargs["device_map"] = "cuda"

        model_cls = self.MODEL_CLASS or AutoModelForCausalLM
        self.model = model_cls.from_pretrained(model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=self.TRUST_REMOTE_CODE,
        )

    # ------------------------------------------------------------------
    # Hooks — override in subclasses as needed
    # ------------------------------------------------------------------

    def _build_message(self, image, prompt: str) -> list:
        """
        Build a single-turn chat message.

        Default format uses an image placeholder ({"type": "image"}) and
        passes the actual PIL image to the processor separately.  Subclasses
        that embed the image inside the message dict (e.g. Qwen) override this.
        """
        return [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }]

    def _prepare_single(self, image, prompt: str) -> "BatchEncoding":
        messages = self._build_message(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )

    def _prepare_batch(self, images, prompts) -> "BatchEncoding":
        self.processor.tokenizer.padding_side = "left"
        texts = [
            self.processor.apply_chat_template(
                self._build_message(img, prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for img, prompt in zip(images, prompts)
        ]
        return self.processor(
            text=texts, images=images, padding=True, return_tensors="pt"
        )

    # ------------------------------------------------------------------
    # Shared decode helper
    # ------------------------------------------------------------------

    def _decode_outputs(self, inputs, generated_ids):
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        raw_counts = [len(t) for t in trimmed]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        visible_counts = [
            len(tokenizer.encode(t, add_special_tokens=False)) for t in texts
        ]
        return texts, visible_counts, raw_counts

    # ------------------------------------------------------------------
    # VLMBase implementation
    # ------------------------------------------------------------------

    def run_inference(self, image, prompt: str):
        inputs = self._prepare_single(image, prompt).to(self.device)

        t0 = time.time()
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        runtime = time.time() - t0

        texts, visible, raw = self._decode_outputs(inputs, generated_ids)
        return texts[0], visible[0], raw[0], runtime

    def run_batch_inference(self, images, prompts):
        inputs = self._prepare_batch(images, prompts).to(self.device)

        t0 = time.time()
        generated_ids = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
        )
        runtime = time.time() - t0

        texts, visible, raw = self._decode_outputs(inputs, generated_ids)
        return texts, visible, raw, runtime
