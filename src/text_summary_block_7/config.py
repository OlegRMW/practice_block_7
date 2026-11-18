"""Validated experiment configuration shared by the three CLIs."""
from copy import deepcopy
from pathlib import Path
import argparse
import math
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAMES = ("baseline", "qwen", "qwen_lora")
PATH_KEYS = {"train", "test", "validation", "models_dir", "results_dir", "adapter_path"}


def load_config(path, overrides=(), named=None):
    path = Path(path).resolve()
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML configuration: {path}") from error
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")
    config = deepcopy(config)
    if isinstance(config.get("judge"), dict):
        judge = config["judge"]
        if "model" in judge:
            raise ValueError("Replace judge.model with judge.models: gpt_oss: {model: openai/gpt-oss-120b}, glm_flash: {model: z-ai/glm-5.3-flash}")
        if not isinstance(judge.get("models"), dict):
            raise ValueError("judge.models must be a mapping")
        for params in judge["models"].values():
            if isinstance(params, dict):
                for key in ("timeout", "max_retries", "temperature", "max_tokens"):
                    params.setdefault(key, None)
    cli_paths = set()
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected --set key=value: {item}")
        # YAML 1.1 treats enum strings such as 'no' as booleans. CLI booleans
        # use the documented true/false spelling; retain these enum strings.
        try:
            parsed = value if value.lower() in {"no", "yes", "on", "off"} else yaml.safe_load(value)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML value for {key}") from error
        assign(config, key, parsed)
        if key.split(".")[-1] in PATH_KEYS:
            cli_paths.add(key)
    for key, value in (named or {}).items():
        if value is not None:
            assign(config, key, value)
            if key.split(".")[-1] in PATH_KEYS:
                cli_paths.add(key)
    _validate(config)
    for section in ("data", "paths", "model"):
        for key, value in config[section].items():
            if key in PATH_KEYS and value is not None:
                base = Path.cwd() if f"{section}.{key}" in cli_paths else path.parent
                config[section][key] = str((base / value).resolve())
    return config


def assign(config, key, value):
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            raise ValueError(f"Unknown configuration key: {key}")
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        raise ValueError(f"Unknown configuration key: {key}")
    node[parts[-1]] = value


def _validate(c):
    allowed = {
        "seed": None,
        "data": {"train", "test", "validation"},
        "paths": {"models_dir", "results_dir"},
        "model": {"kind", "name", "host", "adapter_path", "max_seq_length", "load_in_4bit"},
        "generation": {"system_prompt", "max_new_tokens", "temperature", "num_ctx", "keep_alive", "do_sample", "think"},
        "metrics": {"use_stemmer", "tokenize", "smooth_method"},
        "judge": {"base_url", "models", "api_key_env", "prompt", "timeout", "max_retries", "temperature", "max_tokens", "sample_size", "seed"},
        "resources": {"sample_interval_seconds"},
        "lora": {"r", "lora_alpha", "lora_dropout", "target_modules", "bias", "use_gradient_checkpointing"},
        "training": {"train_size", "num_train_epochs", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "learning_rate", "lr_scheduler_type", "warmup_ratio", "completion_only_loss", "optim", "weight_decay", "logging_steps", "eval_strategy", "save_strategy", "save_total_limit", "max_steps", "validation_size", "run_kind"},
    }
    if c.keys() != allowed.keys():
        raise ValueError(f"Invalid top-level keys; expected {', '.join(allowed)}")
    for section, keys in allowed.items():
        if keys is not None and (not isinstance(c[section], dict) or c[section].keys() != keys):
            raise ValueError(f"Invalid {section} keys; expected {', '.join(sorted(keys))}")
    if c["model"]["kind"] not in MODEL_NAMES:
        raise ValueError(f"model.kind must be one of {MODEL_NAMES}")
    for section in ("data", "paths"):
        for key, value in c[section].items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty path")
    for section, keys in {
        "model": ("name", "host"), "generation": ("system_prompt", "keep_alive"),
        "judge": ("base_url", "api_key_env", "prompt"),
    }.items():
        for key in keys:
            if not isinstance(c[section][key], str) or not c[section][key].strip():
                raise ValueError(f"{section}.{key} must be a nonempty string")
    if c["model"]["adapter_path"] is not None and not isinstance(c["model"]["adapter_path"], str):
        raise ValueError("model.adapter_path must be a path or null")
    for section, key in [("", "seed"), ("judge", "seed")]:
        value = c[section][key] if section else c[key]
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(f"{section}.{key} must be an integer in [0, 2**32)")
    positives = {"model": ["max_seq_length"], "generation": ["max_new_tokens", "num_ctx"],
                 "judge": ["sample_size", "max_tokens"], "lora": ["r", "lora_alpha"],
                 "training": ["train_size", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps", "logging_steps", "save_total_limit"]}
    for section, keys in positives.items():
        for key in keys:
            v = c[section][key]
            if type(v) is not int or v < 1:
                raise ValueError(f"{section}.{key} must be a positive integer")
    for section, key in [("model", "load_in_4bit"), ("generation", "do_sample"), ("generation", "think"), ("metrics", "use_stemmer"), ("training", "completion_only_loss")]:
        if type(c[section][key]) is not bool:
            raise ValueError(f"{section}.{key} must be boolean")
    for section, key, low, high in [
        ("training", "learning_rate", 0, 1), ("training", "num_train_epochs", 0, float("inf")),
        ("judge", "timeout", 0, float("inf")),
    ]:
        v = c[section][key]
        if type(v) not in (int, float) or not math.isfinite(v) or not low < v <= high:
            raise ValueError(f"Invalid {section}.{key}")
    for section, key in [("lora", "lora_dropout"), ("training", "warmup_ratio"), ("training", "weight_decay"), ("generation", "temperature"), ("judge", "temperature")]:
        v = c[section][key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError(f"Invalid {section}.{key}")
    if c["lora"]["lora_dropout"] >= 1 or c["training"]["warmup_ratio"] > 1:
        raise ValueError("Dropout must be < 1; warmup_ratio must be <= 1")
    for key in ("max_steps", "validation_size"):
        v = c["training"][key]
        if type(v) is not int or (v != -1 and v < 1):
            raise ValueError(f"training.{key} must be -1 or positive")
    if type(c["judge"]["max_retries"]) is not int or c["judge"]["max_retries"] < 0:
        raise ValueError("judge.max_retries must be nonnegative")
    if not isinstance(c["lora"]["target_modules"], list) or not c["lora"]["target_modules"] or any(not isinstance(x, str) or not x for x in c["lora"]["target_modules"]):
        raise ValueError("lora.target_modules must be a nonempty string list")
    enums = [("metrics", "tokenize", {"13a", "intl", "none"}), ("metrics", "smooth_method", {"exp", "floor", "add-k", "none"}),
             ("training", "eval_strategy", {"no", "epoch"}), ("training", "save_strategy", {"no", "epoch"}),
             ("training", "lr_scheduler_type", {"cosine", "linear", "constant"}), ("training", "optim", {"adamw_torch", "adamw_torch_fused"}), ("lora", "bias", {"none"})]
    enums.append(("training", "run_kind", {"full", "smoke", "limited"}))
    for section, key, values in enums:
        if not isinstance(c[section][key], str) or c[section][key] not in values:
            raise ValueError(f"{section}.{key} must be one of {values}")
    if c["training"]["completion_only_loss"] is not True:
        raise ValueError("This project trains on completion tokens only")
    if c["lora"]["use_gradient_checkpointing"] not in (True, False, "unsloth"):
        raise ValueError("lora.use_gradient_checkpointing must be boolean or 'unsloth'")
    interval = c["resources"]["sample_interval_seconds"]
    if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
        raise ValueError("resources.sample_interval_seconds must be positive")
    models = c["judge"]["models"]
    if not isinstance(models, dict) or not models:
        raise ValueError("judge.models must be a nonempty mapping")
    for name, params in models.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Judge aliases must contain lowercase letters, digits and underscores")
        if not isinstance(params, dict) or params.keys() != {"model", "timeout", "max_retries", "temperature", "max_tokens"}:
            raise ValueError(f"Invalid judge settings: {name}")
        if not isinstance(params["model"], str) or not params["model"].strip():
            raise ValueError(f"judge.models.{name}.model must be nonempty")
        for key in ("timeout", "max_retries", "temperature", "max_tokens"):
            value = params[key]
            if value is None:
                continue
            integer = key in {"max_retries", "max_tokens"}
            if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value < 0 or (key in {"timeout", "max_tokens"} and value == 0):
                raise ValueError(f"Invalid judge.models.{name}.{key}")
    if len({params["model"] for params in models.values()}) != len(models):
        raise ValueError("Judge model IDs must be distinct")


def select_judges(config, names=None):
    available = config["judge"]["models"]
    names = list(available) if names is None else list(dict.fromkeys(names))
    if not names or any(n not in available for n in names):
        raise ValueError(f"Select judges from: {', '.join(available)}")
    shared = {k: v for k, v in config["judge"].items() if k != "models"}
    return {name: {**shared, **{k: v for k, v in available[name].items() if v is not None}} for name in names}


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=ROOT / "configs/lora.yaml")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return p
