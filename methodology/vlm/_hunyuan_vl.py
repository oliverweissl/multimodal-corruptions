"""
Tencent Hunyuan Embodied vision-language model.

NOTE: verify the exact HuggingFace model ID before use — update MODEL_ID or
pass model_id="org/model-name" to __init__.

Uses trust_remote_code=True (Tencent custom code).
"""

from ._hf_vlm import HuggingFaceVLM


class HunyuanVLInstance(HuggingFaceVLM):
    MODEL_ID = "Tencent-Hunyuan/HunyuanVL"   # ← verify on HuggingFace
    TRUST_REMOTE_CODE = True
