# Multimodal Corruptions

Adversarial search against Vision-Language Models (VLMs) via simultaneous image and text corruptions using MOO.

## 1) Setup

Make sure conda is installed, then run:
```bash
bash 01_create_env.sh
conda activate mmm
```
> [!NOTE]  
> `flash-attn` compilation takes up to 1 hour!

**Dataset** — download the ImageNet ILSVRC 2017 DET validation split and place it under `dataset/2017/ILSVRC/`. All three components are required:

```
dataset/2017/ILSVRC/
├── Data/DET/val/             # validation images (.JPEG)
├── Annotations/DET/val/      # XML annotations
└── devkit/data/meta_det.mat  # synset-to-label mapping
```

## 2) Data Selection

Run once before any experiments. Samples 100 images per group (single-class solo, single-class multi-instance, multi-class) from the dataset:
```bash
bash 02_data_selection.sh
```

## 2.1) Generate Text Perturbation Mappings (Optional)

Generates `homophone_mapping.json` and `synonym_mapping.json` used by text perturbations. Requires [Ollama](https://ollama.com) running locally with a capable model pulled. Without these files the homophone and synonym perturbations are silently skipped.

```bash
bash 02_1_generate_mappings.sh                         # default: gpt-oss:120b
bash 02_1_generate_mappings.sh --model llama3:70b      # override model
```

Alternatively generate the files with any LLM of your choice — the expected format is:
```json
{"label": ["variant1", "variant2", ...], ...}
```

## 3) Run Experiments

Run all VLMs × all modes (multi, image, text) across N GPUs. With N > 1 jobs run simultaneously, one per GPU:
```bash
bash 03_run_all.sh <N_GPUS>   # e.g. bash 03_run_all.sh 2
```
---

## VLMs

The currently available VLMs are shown below, however most HF VLMs should be compatible.

| Key | Model |
|-----|-------|
| `qwen` | Qwen3-VL-4B-Instruct |
| `gemma` | Gemma-3-4b-it |
| `kimi` | Kimi-VL-A3B-Instruct |
| `hunyuan` | HunyuanVL |
