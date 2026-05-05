"""Qwen2.5-VL-7B-Instruct. Outputs bounding boxes in 0-1000 normalised coordinates."""

from ._vllm import VLLMInstance


class Qwen25VLInstance(VLLMInstance):
    MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
    COORD_SCALE = 1000
