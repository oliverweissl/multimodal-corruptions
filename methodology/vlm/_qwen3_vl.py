"""Qwen3-VL-4B-Instruct. Outputs bounding boxes in 0-1000 normalised coordinates."""

from ._vllm import VLLMInstance


class Qwen3VLInstance(VLLMInstance):
    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    COORD_SCALE = 1000
