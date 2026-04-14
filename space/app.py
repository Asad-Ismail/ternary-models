"""
Ternary Model Playground — Interactive demo for ternary-quantized multimodal models.

Demonstrates VLM image understanding and text generation using models quantized
with ternary-quant (https://github.com/Asad-Ismail/ternary-quant).
"""

import gradio as gr
import torch
import os

# Lazy-load models to avoid startup OOM on free Spaces
_loaded_models = {}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_runtime():
    device = get_device()
    if device == "mps":
        return "metal"
    return "cached"


def load_vlm(model_id: str):
    if model_id in _loaded_models:
        return _loaded_models[model_id]

    from ternary_quant.inference import load_ternary_model

    model, processor = load_ternary_model(
        model_id,
        device="auto",
        runtime_mode=get_runtime(),
    )
    _loaded_models[model_id] = (model, processor)
    return model, processor


MODELS = {
    "Gemma 4-E4B (any-to-any multimodal, 8B)": "AsadIsmail/gemma-4-E4B-it-ternary",
    "Qwen2.5-VL-7B": "AsadIsmail/Qwen2.5-VL-7B-Instruct-ternary",
    "SmolVLM2-2.2B (compact)": "AsadIsmail/SmolVLM2-2.2B-Instruct-ternary",
}


def describe_image(image, prompt, model_name):
    if image is None:
        return "Please upload an image."

    model_id = MODELS.get(model_name)
    if not model_id:
        return f"Model {model_name} not found."

    try:
        model, processor = load_vlm(model_id)
    except Exception as e:
        return f"Failed to load model: {e}"

    if not prompt:
        prompt = "Describe this image in detail."

    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

        # Decode only the generated tokens (skip input)
        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated = outputs[0][input_len:]
        return processor.decode(generated, skip_special_tokens=True)
    except Exception as e:
        return f"Generation error: {e}"


def generate_text(prompt, model_name):
    model_id = MODELS.get(model_name)
    if not model_id:
        return f"Model {model_name} not found."

    try:
        model, processor = load_vlm(model_id)
    except Exception as e:
        return f"Failed to load model: {e}"

    try:
        inputs = processor(text=prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)

        input_len = inputs.get("input_ids", torch.tensor([])).shape[-1]
        generated = outputs[0][input_len:]
        return processor.decode(generated, skip_special_tokens=True)
    except Exception as e:
        return f"Generation error: {e}"


with gr.Blocks(title="Ternary Model Playground", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Ternary Model Playground

        Interactive demo for **ternary-quantized multimodal models** — VLMs that GGUF can't quantize.

        These models are quantized with [ternary-quant](https://github.com/Asad-Ismail/ternary-quant)
        using component-aware tritplane3 quantization, retaining 95-99% of original quality.

        **Collection**: [github.com/Asad-Ismail/ternary-models](https://github.com/Asad-Ismail/ternary-models)
        """
    )

    with gr.Tab("Image Understanding (VLM)"):
        with gr.Row():
            with gr.Column():
                vlm_model = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    label="Model",
                )
                vlm_image = gr.Image(type="pil", label="Upload Image")
                vlm_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Describe this image in detail.",
                    value="Describe this image in detail.",
                )
                vlm_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                vlm_output = gr.Textbox(label="Output", lines=10)

        vlm_btn.click(
            describe_image,
            inputs=[vlm_image, vlm_prompt, vlm_model],
            outputs=vlm_output,
        )

    with gr.Tab("Text Generation"):
        with gr.Row():
            with gr.Column():
                txt_model = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value=list(MODELS.keys())[0],
                    label="Model",
                )
                txt_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Write your prompt here...",
                    lines=3,
                )
                txt_btn = gr.Button("Generate", variant="primary")
            with gr.Column():
                txt_output = gr.Textbox(label="Output", lines=10)

        txt_btn.click(
            generate_text,
            inputs=[txt_prompt, txt_model],
            outputs=txt_output,
        )

    gr.Markdown(
        """
        ---
        **How it works**: Each model's weights are quantized to ternary values {-1, 0, +1} using
        progressive tritplane quantization. Vision encoders and text decoders are quantized
        independently with component-aware calibration. See the
        [ternary-quant paper](https://github.com/Asad-Ismail/ternary-quant/tree/main/paper)
        for details.
        """
    )

if __name__ == "__main__":
    demo.launch()
