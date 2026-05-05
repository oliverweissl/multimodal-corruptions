"""OpenGVLab InternVL3.5 — vLLM-based (InternVL2 uses HuggingFace due to flash-attn issues)."""

from ._vllm import VLLMInstance


class InternVL3Instance(VLLMInstance):
    """InternVL3.5-8B served via vLLM. Requires trust_remote_code=True."""

    MODEL_ID = "OpenGVLab/InternVL3_5-8B"
    COORD_SCALE = 1000
