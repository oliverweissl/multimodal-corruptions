"""Moonshot Kimi-VL-A3B-Instruct."""

from ._vllm import VLLMInstance


class KimiVLInstance(VLLMInstance):
    MODEL_ID = "moonshotai/Kimi-VL-A3B-Instruct"
    COORD_SCALE = 1  # Only source: https://github.com/MoonshotAI/Kimi-VL/issues/56
