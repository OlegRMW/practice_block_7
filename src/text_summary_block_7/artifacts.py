"""Writes are restricted to derived artifacts, never source datasets/notebooks."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import tempfile
import uuid
import importlib.metadata
import pandas as pd
import yaml
from .config import ROOT, MODEL_NAMES

METRICS = ["bleu", "rouge1", "rouge2", "rougeL", "rougeLsum"]
CRITERIA = ["faithfulness", "coverage", "relevance", "coherence", "overall"]
JUDGE_COLUMNS = ["id", "model", "response_hash", *CRITERIA, "judge_model", "judge_hash", "recorded_at", "status", "error", "source", "run_id"]


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(path, inputs=()):
    p = Path(path).resolve()
    forbidden = [ROOT / "notebooks", ROOT / "data", ROOT / ".git", ROOT / "src", ROOT / "scripts", ROOT / "configs", ROOT / "tests"]
    if p == ROOT or any(p == x.resolve() or x.resolve() in p.parents for x in forbidden):
        raise ValueError(f"Protected output path: {p}")
    if any(p == Path(x).resolve() or p in Path(x).resolve().parents for x in inputs):
        raise ValueError(f"Output would overwrite an input: {p}")
    # Resolve existing symlinks/junctions before comparing with protected files.
    manifest = ROOT / ".verification/protected.json"
    if manifest.exists():
        for name in json.loads(manifest.read_text(encoding="utf-8")):
            source = (ROOT / name).resolve()
            if p == source or p in source.parents:
                raise ValueError(f"Protected original file: {source}")
    return p


def atomic_text(path, content, inputs=()):
    path = safe_path(path, inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(content)
        os.replace(tmp, path)
    finally:
        if Path(tmp).exists():
            Path(tmp).unlink()


def write_csv(df, path):
    atomic_text(path, df.to_csv(index=False))


def read_csv(path):
    return pd.read_csv(path, dtype={"id": str, "dialogue": str, "summary": str, "reference": str, "prediction": str}, keep_default_na=False)


def predictions(df, values, model, run_id=None):
    if len(df) != len(values) or any(not isinstance(x, str) for x in values):
        raise ValueError("Expected one string prediction per input")
    frame = df[["id", "dialogue", "summary"]].rename(columns={"summary": "reference"}).copy()
    frame["prediction"] = values
    frame["model"] = model
    frame["run_id"] = run_id or uuid.uuid4().hex
    frame["response_hash"] = [fingerprint(model, r.dialogue, r.reference, r.prediction) for r in frame.itertuples()]
    return frame


def save_run(pred, metrics, summary, config, provenance=None):
    root = safe_path(config["paths"]["results_dir"], config["data"].values())
    directory = safe_path(root / config["model"]["kind"])
    directory.mkdir(parents=True, exist_ok=True)
    # run.yaml is the commit marker: readers reject differing run IDs on interruption.
    for name, frame in [("predictions", pred), ("metrics", metrics), ("summary", summary)]:
        write_csv(frame, directory / f"{name}.csv")
    metadata = {"run_id": str(pred.run_id.iloc[0]), "rows": len(pred), "recorded_at": now(), "config": config,
                "test_sha256": file_hash(config["data"]["test"]),
                "metric_versions": {x: importlib.metadata.version(x) for x in ("sacrebleu", "rouge-score", "nltk")},
                "metric_details": {"sentence_effective_order": True, "corpus_effective_order": False, "rouge_aggregation": "mean F1", "rougeLsum_sentences": "nltk.sent_tokenize"},
                "provenance": provenance or "model inference"}
    atomic_text(directory / "run.yaml", yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False))
    rebuild_combined(root)
    return directory


def load_run(directory):
    directory = Path(directory)
    manifest = yaml.safe_load((directory / "run.yaml").read_text(encoding="utf-8"))
    pred, metrics, summary = [read_csv(directory / f"{x}.csv") for x in ("predictions", "metrics", "summary")]
    for frame in (pred, metrics, summary):
        if set(frame.run_id) != {manifest["run_id"]}:
            raise ValueError(f"Incomplete run publication: {directory}")
    if len(pred) != manifest["rows"] or len(metrics) != len(pred) or len(summary) != 1:
        raise ValueError(f"Inconsistent row counts: {directory}")
    if pred.id.duplicated().any() or metrics.id.duplicated().any() or list(pred.id) != list(metrics.id) or list(pred.response_hash) != list(metrics.response_hash):
        raise ValueError(f"Mismatched prediction/metric keys: {directory}")
    return pred, metrics, summary


def load_judges(directory):
    path = Path(directory) / "judge.csv"
    return read_csv(path) if path.exists() else pd.DataFrame(columns=JUDGE_COLUMNS)


def classified_judges(pred, judges):
    if judges.empty:
        return judges.copy(), judges.copy()
    keys = set(zip(pred.id, pred.model, pred.response_hash))
    matching = pd.Series([(str(r.id), r.model, r.response_hash) in keys and bool(r.response_hash) for r in judges.itertuples()], index=judges.index)
    return judges[matching].copy(), judges[~matching].copy()


def rebuild_combined(root):
    frames = []
    for model in MODEL_NAMES:
        directory = Path(root) / model
        if not (directory / "run.yaml").exists():
            continue
        pred, metrics, _ = load_run(directory)
        current, _ = classified_judges(pred, load_judges(directory))
        valid = current[current.status.eq("ok")].drop_duplicates(["id", "response_hash"], keep="last")
        frame = metrics.merge(valid[["id", "response_hash", *CRITERIA, "judge_model", "judge_hash", "source"]], on=["id", "response_hash"], how="left")
        frames.append(frame)
    if frames:
        write_csv(pd.concat(frames, ignore_index=True), Path(root) / "all_models_metrics.csv")
