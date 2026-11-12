"""Lazy neural backends: report and baseline do not import GPU libraries."""
from pathlib import Path
import os
import shutil
from .config import ROOT
from .artifacts import safe_path, fingerprint, file_hash, atomic_text
from .preprocessing import messages


def ollama_generator(c):
    from ollama import Client, ResponseError
    client = Client(host=c["model"]["host"])
    g = c["generation"]

    def generate(dialogue):
        try:
            response = client.chat(model=c["model"]["name"], messages=[
                {"role": "system", "content": g["system_prompt"]}, {"role": "user", "content": dialogue}],
                think=g["think"], options={"temperature": g["temperature"], "num_predict": g["max_new_tokens"], "num_ctx": g["num_ctx"]}, keep_alive=g["keep_alive"])
        except (ConnectionError, ResponseError) as error:
            raise RuntimeError(f"Ollama generation failed: {error}") from error
        return response.message.content.strip()
    return generate


def fast_language_model():
    cache = safe_path(ROOT / ".cache/unsloth")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = str(cache)
    # Import Unsloth before transformers/trl, as required by its patching.
    from unsloth import FastLanguageModel
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Install CUDA-enabled PyTorch in the active Poetry environment for LoRA.")
    return FastLanguageModel


def lora_generator(c):
    adapter = c["model"]["adapter_path"]
    if not adapter or not (Path(adapter) / "adapter_config.json").exists():
        raise ValueError("LoRA adapter missing. Run train.py or pass --adapter-path PATH.")
    # Unsloth may repair tokenizer metadata. Never let a backend write to input
    # adapters, particularly the protected notebook artifacts.
    source = Path(adapter)
    files = sorted(p for p in source.iterdir() if p.is_file())
    staged = safe_path(ROOT / ".cache/adapters" / fingerprint(str(source.resolve()), [(p.name, file_hash(p)) for p in files]))
    if not (staged / ".complete").exists():
        staged.mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, staged / file.name)
        atomic_text(staged / ".complete", "staged read-only input adapter\n")
    adapter = str(staged)
    FastLanguageModel = fast_language_model()
    import torch
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=adapter, max_seq_length=c["model"]["max_seq_length"], dtype=None, load_in_4bit=c["model"]["load_in_4bit"])
    FastLanguageModel.for_inference(model)
    g = c["generation"]

    def generate(dialogue):
        # A mapping works across newer tokenizer versions returning BatchEncoding.
        inputs = tokenizer.apply_chat_template(messages(dialogue, g["system_prompt"]), tokenize=True,
            add_generation_prompt=True, enable_thinking=g["think"], return_tensors="pt", return_dict=True).to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=g["max_new_tokens"], do_sample=g["do_sample"], use_cache=True,
                                     **({"temperature": g["temperature"]} if g["do_sample"] else {}))
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return generate
