"""Lightweight resource storage: safe for report without GPU libraries."""
from pathlib import Path
import pandas as pd
import yaml
from .artifacts import file_hash, fingerprint, read_csv, write_csv


def adapter_info(path):
    if not path:
        return {"adapter_id": "", "adapter_mib": None}
    directory = Path(path)
    files = sorted(p for p in directory.glob('adapter*') if p.is_file() and p.suffix in {'.json', '.safetensors', '.bin'})
    if not files:
        return {"adapter_id": "", "adapter_mib": None}
    tokenizer_files = sorted(p for p in directory.glob("tokenizer*") if p.is_file())
    return {"adapter_id": fingerprint([(p.name, file_hash(p)) for p in files]),
            "tokenizer_id": fingerprint([(p.name, file_hash(p)) for p in tokenizer_files]),
            "adapter_mib": sum(p.stat().st_size for p in files) / 2**20}


def save_resources(rows, directory):
    write_csv(pd.DataFrame(rows), Path(directory) / "resources.csv")


def load_resources(directory):
    path = Path(directory) / "resources.csv"
    frame = read_csv(path) if path.exists() else pd.DataFrame()
    marker = Path(directory) / "run.yaml"
    if not frame.empty and marker.exists():
        manifest = yaml.safe_load(marker.read_text(encoding="utf-8"))
        if "run_id" not in frame or set(frame.run_id) != {manifest["run_id"]}:
            raise ValueError(f"Resource run ID mismatch: {directory}")
    return frame
