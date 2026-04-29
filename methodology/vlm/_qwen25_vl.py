"""
Qwen2.5-VL vision-language model.

Same preprocessing overrides as Qwen3-VL (shared via inheritance).
Uses Qwen2_5_VLForConditionalGeneration instead of Qwen3VLForConditionalGeneration.

Available sizes:
    Qwen/Qwen2.5-VL-3B-Instruct
    Qwen/Qwen2.5-VL-7B-Instruct   ← default
    Qwen/Qwen2.5-VL-72B-Instruct
"""

from ._qwen3_vl import Qwen3VLInstance


class Qwen25VLInstance(Qwen3VLInstance):
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    MODEL_CLASS = None  # set in __init__ via lazy import

    def __init__(self, seed: int, max_new_tokens: int = 1024, device: str = "gpu",
                 model_id: str = None):
        from transformers import Qwen2_5_VLForConditionalGeneration
        self.MODEL_CLASS = Qwen2_5_VLForConditionalGeneration
        super().__init__(seed=seed, max_new_tokens=max_new_tokens, device=device,
                         model_id=model_id)
