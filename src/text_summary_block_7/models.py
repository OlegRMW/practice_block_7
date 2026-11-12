import re


def baseline_model(text):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    selected = [sentences[0] + sentences[-1]] if len(sentences) > 2 else sentences
    return re.sub(r"#Person\d+#:\s*", "", " ".join(selected)).strip()


def generator(config):
    kind = config["model"]["kind"]
    if kind == "baseline":
        return baseline_model
    from .neural import ollama_generator, lora_generator
    return ollama_generator(config) if kind == "qwen" else lora_generator(config)
