"""Immutable inference runs and a wide per-answer view of independent run scores."""
from pathlib import Path
import pandas as pd
import yaml
from .artifacts import METRICS, CRITERIA, load_run, load_judges, write_csv
from .config import select_judges
from .judge import matching_settings, settings_hash


def runs(directory):
    result = []
    for marker in (Path(directory) / "runs").glob("*/run.yaml"):
        manifest = yaml.safe_load(marker.read_text(encoding="utf-8"))
        pred, metrics, summary = load_run(marker.parent)
        result.append((manifest, pred, metrics, summary, marker.parent))
    return sorted(result, key=lambda x: x[0]["recorded_at"])


def run_judges(directory, manifest, pred):
    judges = load_judges(directory)
    subset = judges[judges.run_id.eq(manifest["run_id"])]
    keys = set(zip(pred.id, pred.response_hash))
    return subset[[tuple(x) in keys for x in zip(subset.id, subset.response_hash)]].copy() if not subset.empty else subset


def rebuild_wide(directory):
    records = {}
    judges = load_judges(directory)
    for manifest, pred, metrics, _, _ in runs(directory):
        prefix = "run_" + manifest["run_id"] + "__"
        by_id = metrics.set_index("id")
        planned = manifest.get("evaluation", {}).get("judges", {})
        for row in pred.itertuples():
            key = (manifest["test_sha256"], row.id)
            target = records.setdefault(key, {"dataset_hash": key[0], "id": row.id, "model": row.model})
            target[prefix + "response_hash"] = row.response_hash
            for metric in METRICS:
                target[prefix + metric] = by_id.loc[row.id, metric]
            for alias in manifest["config"]["judge"]["models"]:
                entries = judges[judges.run_id.eq(manifest["run_id"]) & judges.id.eq(row.id) & judges.response_hash.eq(row.response_hash) & judges.judge_alias.eq(alias)]
                ok = entries[entries.status.eq("ok")]
                for criterion in CRITERIA:
                    target[prefix + alias + "__" + criterion] = ok.iloc[-1][criterion] if not ok.empty else None
                target[prefix + alias + "__status"] = entries.iloc[-1].status if not entries.empty else ("not_evaluated" if alias in planned else "not_requested")
                target[prefix + alias + "__judge_model"] = manifest["config"]["judge"]["models"][alias]["model"]
    if records:
        write_csv(pd.DataFrame(records.values()), Path(directory) / "scores_by_run.csv")


def aggregate_judges(directory, history):
    """Equal run weight on the common dialogue set; incomplete runs excluded."""
    rows, incomplete = [], []
    groups = {}
    for manifest, pred, _, _, _ in history:
        subset = run_judges(directory, manifest, pred)
        evaluation = manifest.get("evaluation", {})
        expected = set(evaluation.get("sample_ids", []))
        for alias, settings in evaluation.get("judges", {}).items():
            recorded = select_judges(manifest["config"], [alias])[alias]
            if settings_hash(recorded) != settings["hash"] or recorded["model"] != settings["model"]:
                raise ValueError("Recorded judge configuration does not match the evaluation plan")
            selected = subset[subset.judge_alias.eq(alias) & matching_settings(subset, recorded)]
            latest = selected.drop_duplicates("id", keep="last")
            ok = selected[selected.status.eq("ok")].drop_duplicates("id", keep="last")
            errors = int(latest.status.ne("ok").sum())
            if not expected or set(ok.id) != expected or errors:
                incomplete.append({"model": pred.model.iloc[0], "run_id": manifest["run_id"], "judge": settings["model"], "ok": len(ok), "expected": len(expected), "errors": errors})
                continue
            key = (manifest["conditions_id"], alias, settings["model"], settings["hash"])
            groups.setdefault(key, []).append((manifest, ok))
    # Use the same IDs for all judges within each inference condition group.
    common = {}
    for (condition, _, _, _), entries in groups.items():
        for _, frame in entries:
            ids = set(frame.id)
            common[condition] = common.get(condition, ids) & ids
    for (condition, alias, judge_model, judge_hash), entries in groups.items():
        ids = common[condition]
        if not ids:
            incomplete.append({"model": entries[0][1].model.iloc[0], "run_id": "multiple", "judge": judge_model, "ok": 0, "expected": "common IDs", "errors": "no overlap"})
            continue
        means = [frame[frame.id.isin(ids)][CRITERIA].apply(pd.to_numeric).mean() for _, frame in entries]
        overrides = sum(int(frame[frame.id.isin(ids)].recovery_note.eq("max_tokens_override").sum()) for _, frame in entries)
        rows.append({"model": entries[0][1].model.iloc[0], "condition": condition, "judge": judge_model, "judge_alias": alias,
                     "judge_hash": judge_hash, "runs": len(entries), "N": len(ids), "budget_overrides": overrides,
                     **pd.DataFrame(means).mean().to_dict()})
    return pd.DataFrame(rows), pd.DataFrame(incomplete)
