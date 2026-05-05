"""
Qwen3-Embedding-0.6B served via vLLM for GPU-efficient text embedding.

When a vLLM embedding server is detected on port 8699, uses the
``/v1/embeddings`` HTTP endpoint instead of loading the model in-process.
Start the server with:

    vllm serve Qwen/Qwen3-Embedding-0.6B --port 8699 --task embed
"""

import time
from typing import Optional, Union

import logging
import numpy as np
import requests
from vllm import LLM

_SERVED_PORT = 8699
_SERVED_URL = f"http://localhost:{_SERVED_PORT}"


def _find_embedding_server(model_id: str) -> Optional[str]:
    """Check whether a vLLM embedding server is running on port 8699 for ``model_id``.

    :param model_id: Embedding model identifier to look for.
    :returns: Base URL if found, ``None`` otherwise.
    """
    try:
        resp = requests.get(f"{_SERVED_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            served_ids = [m["id"] for m in resp.json().get("data", [])]
            if any(model_id in sid or sid in model_id for sid in served_ids):
                return _SERVED_URL
    except requests.exceptions.RequestException:
        pass
    return None


class Qwen3EmbeddingInstance:
    """Qwen3-Embedding-0.6B — uses a running vLLM server when available, else loads in-process.

    :param seed: Random seed for vLLM (in-process mode only).
    :param gpu_memory_utilization: Fraction of GPU memory for in-process mode.
    :param model_id: Override the default model identifier.
    """
    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        seed: int,
        gpu_memory_utilization: float = 0.05,
        model_id: Optional[str] = None,
    ) -> None:
        """Connect to a running embedding server or load the model via vLLM.

        :param seed: Random seed forwarded to vLLM (in-process mode only).
        :param gpu_memory_utilization: Fraction of GPU memory this model may use.
        :param model_id: Override the class-level ``MODEL_ID``.
        """
        model_id = model_id or self.MODEL_ID
        self._model_id = model_id

        served_url = _find_embedding_server(model_id)
        if served_url:
            logging.getLogger(__name__).info(
                "Detected vLLM embedding server at %s — HTTP mode.", served_url
            )
            self._served_url = f"{served_url}/v1/embeddings"
            self.llm = None
            return

        self._served_url = None

        self.llm = LLM(
            model=model_id,
            seed=seed,
            max_model_len=4096,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
        )

    def _embed_http(self, texts: list[str]) -> tuple[list[np.ndarray], list[int], float]:
        """Embed a batch of texts via the HTTP server.

        :param texts: List of input strings.
        :returns: Tuple of (embeddings, token counts, runtime seconds).
        """
        payload = {"model": self._model_id, "input": texts}
        t0 = time.time()
        resp = requests.post(self._served_url, json=payload, timeout=120)
        runtime = time.time() - t0
        resp.raise_for_status()
        data = resp.json()["data"]
        vectors = []
        for item in sorted(data, key=lambda x: x["index"]):
            emb = np.array(item["embedding"], dtype=np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            vectors.append(emb)
        # token counts not returned by /v1/embeddings — approximate with text length
        counts = [len(t.split()) for t in texts]
        return vectors, counts, runtime

    def run_inference(
        self, text: str, instruction: Optional[str] = None
    ) -> tuple[np.ndarray, int, float]:
        """Embed a single text string.

        :param text: Input text to embed.
        :param instruction: Optional instruction prefix prepended to text.
        :returns: Tuple of (L2-normalised embedding vector, token count, runtime seconds).
        """
        full_text = f"{instruction}{text}" if instruction else text
        if self._served_url:
            vectors, counts, runtime = self._embed_http([full_text])
            return vectors[0], counts[0], runtime

        t0 = time.time()
        outputs = self.llm.embed([full_text])
        runtime = time.time() - t0
        emb = np.array(outputs[0].outputs.embedding, dtype=np.float32)
        emb /= np.linalg.norm(emb) + 1e-9
        return emb, len(outputs[0].prompt_token_ids), runtime

    def run_batch_inference(
        self,
        texts: list[str],
        instructions: Optional[Union[str, list[str]]] = None,
    ) -> tuple[list[np.ndarray], list[int], float]:
        """Embed a batch of text strings in a single call.

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

        if self._served_url:
            return self._embed_http(full_texts)

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
