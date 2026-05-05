"""DeepSeek-VL2-tiny."""

from PIL import Image

from ._vllm import VLLMInstance


class DeepSeekVL2Instance(VLLMInstance):
    MODEL_ID = "deepseek-ai/deepseek-vl2-tiny"  # Bigger ones do not seem to work out of the box on vLLM.
    COORD_SCALE = 999
    MAX_MODEL_LEN = 4096

    def _messages(self, image: Image.Image, prompt: str) -> list[dict]:
        """Wrap prompt in DeepSeek grounding tags required for bbox output.

        :param image: PIL image for the user turn.
        :param prompt: Text prompt string.
        :returns: Single-element list containing the user message dict.
        """
        return super()._messages(image, f" <|ref|>{prompt}<|/ref|>.")
