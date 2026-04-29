"""
Moonshot Kimi-VL vision-language model.

Uses trust_remote_code=True (custom modeling code on HuggingFace).

Default: Kimi-VL-A3B-Instruct. Pass model_id to override.
"""

from ._hf_vlm import HuggingFaceVLM


class KimiVLInstance(HuggingFaceVLM):
    MODEL_ID = "moonshotai/Kimi-VL-A3B-Instruct"
    TRUST_REMOTE_CODE = True
