"""
Google Gemma vision-language model.

Gemma uses AutoModelForImageTextToText and the standard HF processor pattern.
The image placeholder in the message is {"type": "image"} (no embedded PIL),
with the actual image passed to processor(images=...) separately — this is
already what HuggingFaceVLM._build_message produces, so no overrides needed.

Default: gemma-3-4b-it. Pass model_id to use a different variant:
    google/gemma-3-4b-it    ← default (4B)
    google/gemma-3-12b-it
    google/gemma-3-27b-it
"""

from transformers import AutoModelForImageTextToText

from ._hf_vlm import HuggingFaceVLM


class GemmaVLInstance(HuggingFaceVLM):
    MODEL_ID = "google/gemma-3-4b-it"
    MODEL_CLASS = AutoModelForImageTextToText
