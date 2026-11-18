"""One-time developer import of existing notebook CSVs (never modifies them)."""
from pathlib import Path
import pandas as pd
from .config import ROOT, load_config
from .data import load_data
from .metrics import compute_metrics
from .judge import parse_scores
from .artifacts import predictions, save_run, read_csv, write_csv, JUDGE_COLUMNS, CRITERIA, fingerprint, rebuild_combined, file_hash
import json


def bootstrap(results_dir=None):
    source = ROOT / "notebooks/qwen_lora_test_predictions.csv"
    old = read_csv(source)
    test = load_data(ROOT / "data/test.csv/test.csv")
    if not old[["id", "dialogue", "summary"]].equals(test[["id", "dialogue", "summary"]]):
        raise ValueError("Notebook predictions do not match original test IDs/text/order")
    judge_path = ROOT / "notebooks/routerai_judge_results.csv"
    judges = read_csv(judge_path)
    labels = {"baseline": "Baseline", "qwen": "Qwen Zero-shot", "qwen_lora": "Qwen LoRA"}
    configs = {"baseline": "baseline", "qwen": "qwen", "qwen_lora": "lora"}
    columns = {"baseline": "baseline_summary", "qwen": "qwen_summary", "qwen_lora": "qwen_lora_summary"}
    for kind, name in configs.items():
        c = load_config(ROOT / f"configs/{name}.yaml", named={"paths.results_dir": str(Path(results_dir).resolve())} if results_dir else None)
        # These are original notebook predictions, not a new 64-token inference.
        c["generation"]["max_new_tokens"] = 128 if kind == "qwen_lora" else 64
        pred = predictions(test, old[columns[kind]].tolist(), kind)
        metrics, summary = compute_metrics(pred, c["metrics"])
        directory = save_run(pred, metrics, summary, c, provenance={"predictions": str(source.relative_to(ROOT)), "sha256": file_hash(source)})
        imported = []
        for row in judges[judges.model.eq(labels[kind])].to_dict("records"):
            idx = int(row["index"])
            if not 0 <= idx < len(test):
                raise ValueError("Judge index out of range")
            scores = parse_scores(json.dumps({k: int(row[k]) for k in CRITERIA[:-1]}))
            if abs(float(row["overall"]) - scores["overall"]) > 1e-8:
                raise ValueError("Inconsistent imported overall")
            # Index matches the test, but source CSV does not record evaluated text.
            # Preserve as historical/unverified rather than manufacture a response hash.
            imported.append({"id": test.id.iloc[idx], "model": kind, "response_hash": "", **scores,
                             "judge_model": c["judge"]["models"]["gpt_oss"]["model"], "judge_hash": fingerprint("legacy", str(judge_path)),
                             "recorded_at": "", "status": "legacy_unverified", "error": "", "source": "notebooks/routerai_judge_results.csv (index only)", "run_id": "legacy"})
        write_csv(pd.DataFrame(imported, columns=JUDGE_COLUMNS), directory / "judge.csv")
        print(f"Prepared {kind}: {len(pred)} answers; {len(imported)} historical judge scores")
    rebuild_combined(c["paths"]["results_dir"])
