"""
Qwen3-Embedding-0.6B served via vLLM for GPU-efficient text embedding.

Uses a small 0.6B model so the VLM can claim most of the GPU budget.
vLLM manages memory independently from the VLM instance, eliminating
OOM conflicts from separate HF model allocations.
"""

import time
from typing import Optional, Union

import numpy as np


class Qwen3EmbeddingInstance:
    """vLLM-backed text embedding using Qwen3-Embedding-0.6B.

    :param seed: Random seed for vLLM.
    :param gpu_memory_utilization: Fraction of GPU memory reserved for the embedding model.
    :param model_id: Override the default model identifier.
    """

    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        seed: int,
        gpu_memory_utilization: float = 0.1,  # For 43GB VRAM -> ~2.15GB Space
        model_id: Optional[str] = None,
    ) -> None:
        """Load the embedding model via vLLM.

        :param seed: Random seed forwarded to vLLM.
        :param gpu_memory_utilization: Fraction of GPU memory this model may use.
        :param model_id: Override the class-level ``MODEL_ID``.
        """
        from vllm import LLM

        self.llm = LLM(
            model=model_id or self.MODEL_ID,
            seed=seed,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def run_inference(
        self, text: str, instruction: Optional[str] = None
    ) -> tuple[np.ndarray, int, float]:
        """Embed a single text string.

        :param text: Input text to embed.
        :param instruction: Optional instruction prefix prepended to text.
        :returns: Tuple of (L2-normalised embedding vector, token count, runtime seconds).
        """
        full_text = f"{instruction}{text}" if instruction else text
        t0 = time.time()
        outputs = self.llm.embed([full_text])
        runtime = time.time() - t0

        emb = np.array(outputs[0].outputs.embedding, dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        token_count = len(outputs[0].prompt_token_ids)
        return emb, token_count, runtime

    def run_batch_inference(
        self,
        texts: list[str],
        instructions: Optional[Union[str, list[str]]] = None,
    ) -> tuple[list[np.ndarray], list[int], float]:
        """Embed a batch of text strings in a single vLLM call.

        :param texts: List of input texts to embed.
        :param instructions: Optional instruction prefix(es): a single string applied to all, a
            list of per-text strings, or None.
        :returns: Tuple of (list of embedding vectors, list of token counts, runtime in seconds).
        :raises ValueError: If ``instructions`` is a list whose length differs from ``texts``.
        """
        if instructions is None:
            full_texts = texts
        elif isinstance(instructions, str):
            full_texts = [f"{instructions}{t}" for t in texts]
        elif isinstance(instructions, list):
            if len(instructions) != len(texts):
                raise ValueError("Instructions list length must match texts list length.")
            full_texts = [f"{inst}{t}" for inst, t in zip(instructions, texts)]
        else:
            full_texts = texts

        t0 = time.time()
        outputs = self.llm.embed(full_texts)
        runtime = time.time() - t0

        vectors = []
        token_counts = []
        for o in outputs:
            emb = np.array(o.outputs.embedding, dtype=np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            vectors.append(emb)
            token_counts.append(len(o.prompt_token_ids))

        return vectors, token_counts, runtime
