"""Complete an existing run without regenerating or rescoring successful answers."""
from pathlib import Path
import re
import yaml
import pandas as pd
from .artifacts import load_run, load_judges, write_csv, safe_path, rebuild_combined
from .config import select_judges
from .history import rebuild_wide
from .judge import api_key, settings_hash, evaluate_judge, matching_settings


def retry_failed(config, run_id=None, names=None, response_attempts=3, recovery_max_tokens=None):
    directory = safe_path(Path(config["paths"]["results_dir"]) / config["model"]["kind"])
    if run_id is None:
        run_id = yaml.safe_load((directory / "run.yaml").read_text(encoding="utf-8"))["run_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    snapshot = safe_path(directory / "runs" / run_id)
    manifest = yaml.safe_load((snapshot / "run.yaml").read_text(encoding="utf-8"))
    pred, _, _ = load_run(snapshot)
    plan = manifest.get("evaluation", {})
    expected = set(plan.get("sample_ids", []))
    planned = plan.get("judges", {})
    names = list(planned) if names is None else list(dict.fromkeys(names))
    if not expected or not names or any(n not in planned for n in names):
        raise ValueError("Select judges originally scheduled for this run")
    if not expected <= set(pred.id):
        raise ValueError("Recorded judge sample does not match snapshot")
    selected = select_judges(manifest["config"], names)
    if recovery_max_tokens is not None and (type(recovery_max_tokens) is not int or any(recovery_max_tokens <= s["max_tokens"] for s in selected.values())):
        raise ValueError("Recovery max_tokens must exceed the recorded limit")
    remaining = 0
    try:
        for alias, settings in selected.items():
            if settings_hash(settings) != planned[alias]["hash"] or settings["model"] != planned[alias]["model"]:
                raise ValueError("Recorded judge configuration mismatch; refusing to mix scores")
            old = load_judges(directory)
            matching = old.run_id.eq(run_id) & matching_settings(old, settings)
            good = old[matching & old.status.eq("ok")]
            keys = set(zip(good.id, good.response_hash))
            sample = pred[pred.id.isin(expected)]
            missing = sample[[tuple(key) not in keys for key in zip(sample.id, sample.response_hash)]]
            # A stale failed refresh must not exclude an already valid score.
            valid_key = pd.Series([(row.id, row.response_hash) in keys for row in old.itertuples()], index=old.index, dtype=bool)
            stale = matching & old.status.ne("ok") & valid_key
            if stale.any():
                write_csv(old[~stale], directory / "judge.csv")
            print(f"{settings['model']}: {len(sample) - len(missing)}/{len(sample)} already valid; retrying {len(missing)}")
            if not missing.empty:
                api_key(settings)
                kwargs = {"recovery_max_tokens": recovery_max_tokens} if recovery_max_tokens is not None else {}
                remaining += evaluate_judge(pred, settings, directory, alias=alias, sample=missing, response_attempts=response_attempts, **kwargs)
    finally:
        all_scores = load_judges(directory)
        write_csv(all_scores[all_scores.run_id.eq(run_id)], snapshot / "judge.csv")
        rebuild_combined(directory.parent)
        rebuild_wide(directory)
    print(f"Recovery of run {run_id}: remaining failures in selected judges: {remaining}")
    return 1 if remaining else 0
