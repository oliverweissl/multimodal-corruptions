from ._base import VLMBase
from ._vllm import VLLMInstance
from ._qwen3_vl import Qwen3VLInstance
from ._qwen25_vl import Qwen25VLInstance
from ._gemma_vl import GemmaVLInstance
from ._kimi_vl import KimiVLInstance
from ._deepseek_vl2 import DeepSeekVL2Instance
from ._intern_vl3 import InternVL3Instance
from ._ernie_vl import ErnieVLInstance

__all__ = [
    "VLMBase",
    "VLLMInstance",
    "Qwen3VLInstance",
    "Qwen25VLInstance",
    "GemmaVLInstance",
    "KimiVLInstance",
    "DeepSeekVL2Instance",
    "InternVL3Instance",
    "ErnieVLInstance",
]
