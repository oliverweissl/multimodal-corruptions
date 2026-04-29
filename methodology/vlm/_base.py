from abc import ABC, abstractmethod
from typing import List, Tuple
import PIL.Image


class VLMBase(ABC):
    """
    Abstract interface for all vision-language model wrappers.

    Both methods return a 4-tuple with the same semantics:
        response_text(s)   - decoded output
        token_count(s)     - visible output tokens (excludes thinking blocks)
        raw_token_count(s) - total generated tokens (includes thinking blocks)
        runtime            - wall-clock seconds for model.generate call
    """

    @abstractmethod
    def run_inference(
        self, image: PIL.Image.Image, prompt: str
    ) -> Tuple[str, int, int, float]:
        ...

    @abstractmethod
    def run_batch_inference(
        self,
        images: List[PIL.Image.Image],
        prompts: List[str],
    ) -> Tuple[List[str], List[int], List[int], float]:
        ...
