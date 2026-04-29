"""
Qwen3-VL vision-language model.

Deviates from the standard HF pattern:
  1. PIL image is embedded directly inside the message dict.
  2. Single-sample preprocessing requires qwen_vl_utils.process_vision_info.
  3. Batch preprocessing uses apply_chat_template(tokenize=True).
  4. Raw vs visible token counts differ when thinking mode is active.
"""

from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration

from ._hf_vlm import HuggingFaceVLM


class Qwen3VLInstance(HuggingFaceVLM):
    """Qwen3-VL-4B-Instruct (default). Pass model_id to use a different size."""

    MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
    MODEL_CLASS = Qwen3VLForConditionalGeneration

    def _build_message(self, image, prompt: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _prepare_single(self, image, prompt: str):
        messages = self._build_message(image, prompt)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    def _prepare_batch(self, images, prompts):
        self.processor.tokenizer.padding_side = "left"
        messages_list = [self._build_message(img, prompt) for img, prompt in zip(images, prompts)]
        return self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

    def _decode_outputs(self, inputs, generated_ids):
        trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated_ids)]
        raw_counts = [len(t) for t in trimmed]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # Visible tokens only — excludes <think>...</think> blocks
        visible_counts = [
            len(self.processor.tokenizer.encode(t, add_special_tokens=False)) for t in texts
        ]
        return texts, visible_counts, raw_counts
