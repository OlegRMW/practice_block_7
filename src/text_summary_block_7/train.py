from pathlib import Path
import inspect
import math
import yaml
from .artifacts import safe_path, atomic_text
from .data import load_data
from .neural import fast_language_model
from .preprocessing import prepare_dataset


def train(c):
    if c["model"]["kind"] != "qwen_lora":
        raise ValueError("Only qwen_lora supports training")
    output = safe_path(c["paths"]["models_dir"], c["data"].values())
    if (output / "adapter_config.json").exists():
        raise ValueError(f"Adapter already exists: {output}; choose a new --output-dir")
    train_df, val_df = (load_data(c["data"][k], training=True) for k in ("train", "validation"))
    if set(train_df.id) & set(val_df.id):
        raise ValueError("Train and validation IDs overlap")
    settings = c["training"].copy()
    size = settings.pop("train_size")
    if size > len(train_df):
        raise ValueError(f"train_size={size} exceeds {len(train_df)} available rows")
    train_df = train_df.sample(n=size, random_state=c["seed"])
    val_size = settings.pop("validation_size")
    if val_size != -1:
        if val_size > len(val_df):
            raise ValueError("validation_size exceeds available validation rows")
        val_df = val_df.sample(n=val_size, random_state=c["seed"])
    FastLanguageModel = fast_language_model()
    import torch
    from trl import SFTConfig, SFTTrainer
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=c["model"]["name"], max_seq_length=c["model"]["max_seq_length"], dtype=None, load_in_4bit=c["model"]["load_in_4bit"])
    model = FastLanguageModel.get_peft_model(model, **c["lora"], random_state=c["seed"])
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Expected only LoRA adapter parameters to be trainable")
    datasets = [prepare_dataset(df, tokenizer, c["generation"]["system_prompt"]) for df in (train_df, val_df)]
    ratio = settings.pop("warmup_ratio")
    if "warmup_ratio" in inspect.signature(SFTConfig).parameters:
        settings["warmup_ratio"] = ratio
    else:
        total = settings["max_steps"] if settings["max_steps"] > 0 else math.ceil(math.ceil(len(train_df) / (settings["per_device_train_batch_size"] * settings["gradient_accumulation_steps"])) * settings["num_train_epochs"])
        settings["warmup_steps"] = math.ceil(total * ratio)
    args = SFTConfig(output_dir=str(output / "checkpoints"), **settings, max_length=c["model"]["max_seq_length"],
                     fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(), report_to="none", seed=c["seed"])
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=datasets[0], eval_dataset=datasets[1], args=args)
    result = trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    atomic_text(output / "training-run.yaml", yaml.safe_dump({"config": c, "metrics": result.metrics, "trainable_parameters": trainable}, allow_unicode=True, sort_keys=False))
    return output
