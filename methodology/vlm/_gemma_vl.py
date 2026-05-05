"""Google Gemma-3-4b-it. Processes images at 896×896; coord scale inferred automatically."""

from ._vllm import VLLMInstance
from PIL import Image


class GemmaVLInstance(VLLMInstance):
    MODEL_ID = "google/gemma-3-4b-it"
    COORD_SCALE = 896

    def _messages(self, image: Image.Image, prompt: str) -> list[dict]:
        """Resize image to 896×896 (Gemma's training resolution) before encoding.

        :param image: PIL image for the user turn.
        :param prompt: Text prompt string.
        :returns: Single-element list containing the user message dict.
        """
        image = image.resize((self.COORD_SCALE, self.COORD_SCALE))
        return super()._messages(image, prompt)
