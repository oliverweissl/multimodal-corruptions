"""Baidu ERNIE-4.5-VL — vLLM-based.

Heterogeneous MoE architecture: torch.compile and CUDA graphs are not supported.
vLLM's enforce_eager=True (set in VLLMInstance) handles this automatically.

Default: 28B-A3B-PT (fits on a single 80 GB GPU).
COORD_SCALE must be set empirically via investigate_scales.ipynb.
"""

from ._vllm import VLLMInstance


class ErnieVLInstance(VLLMInstance):
    """ERNIE-4.5-VL-28B served via vLLM."""

    MODEL_ID = "baidu/ERNIE-4.5-VL-28B-A3B-PT"
    COORD_SCALE = 1000
    TRUST_REMOTE_CODE = True
    TENSOR_PARALLEL_SIZE = 2
