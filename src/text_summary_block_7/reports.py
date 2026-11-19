"""Compact CSV-only reports, including independent averages across runs."""
from pathlib import Path
import pandas as pd
import yaml
from .config import MODEL_NAMES
from .artifacts import METRICS, CRITERIA, load_run, load_judges
from .history import runs, aggregate_judges
from .resource_artifacts import load_resources


def table(console, title, frame, columns=None):
    from rich.table import Table
    t = Table(title=title, show_lines=False, expand=False)
    frame = frame if columns is None else frame.reindex(columns=columns)
    labels = {"model": "Model", "count": "N", "corpus_bleu": "BLEU corp", "bleu": "BLEU sent",
              "rouge1": "R-1", "rouge2": "R-2", "rougeL": "R-L", "rougeLsum": "R-Lsum",
              "faithfulness": "Faith", "coverage": "Cover", "relevance": "Relev", "coherence": "Coher",
              "overall": "Avg", "errors": "Err", "condition": "Cond", "runs": "Runs"}
    for col in frame.columns:
        t.add_column(labels.get(col, str(col)), overflow="fold")
    for values in frame.itertuples(index=False, name=None):
        t.add_row(*[("N/A" if pd.isna(v) or v == "" else f"{v:.3f}" if isinstance(v, float) else str(v)) for v in values])
    console.print(t)


def number(row, name):
    value = row.get(name)
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def resource_report(console, root, histories):
    times, ram, gpu = [], [], []
    for model, history in histories.items():
        latest = {}
        for entry in history:
            latest[entry[0]["conditions_id"]] = entry
        if not latest:
            times.append({"model": model})
        for condition, (manifest, _, _, _, directory) in latest.items():
            resources = load_resources(directory)
            if resources.empty:
                times.append({"model": model, "condition": condition[:6]})
                continue
            prep = resources[resources.phase.eq("preparation")]
            main = resources[resources.phase.eq("inference")]
            if main.empty:
                continue
            r = main.iloc[0]
            key = {"model": model, "condition": condition[:6]}
            times.append({**key, "limit": r.get("max_new_tokens"), "prep s": number(prep.iloc[0], "wall_seconds") if not prep.empty else None,
                          "load s": number(r, "model_load_seconds"), "work s": number(r, "execution_seconds"), "ans/s": number(r, "answers_per_second"),
                          "tok/s": number(r, "tokens_per_second")})
            def peak(name):
                return pd.to_numeric(resources[name], errors="coerce").max() if name in resources else None
            ram.append({**key, "Python MiB": peak("client_ram_peak_mib"), "server MiB": peak("server_ram_peak_mib"), "adapter MiB": number(r, "adapter_mib")})
            gpu.append({**key, "alloc MiB": peak("torch_allocated_peak_mib"), "reserve MiB": peak("torch_reserved_peak_mib"),
                        "process MiB": peak("gpu_process_peak_mib"), "resident MiB": peak("model_resident_vram_mib")})
            console.print(f"{model}/{condition[:6]}: client CPU={r.get('cpu', 'N/A')}; GPU={r.get('gpu', 'N/A')}; precision={r.get('precision', 'N/A')}; load={r.get('load_state', 'unknown')}")
            if r.get("gpu") == "unknown" and r.get("available_gpu"):
                console.print(f"Local GPU available (server placement unverified): {r['available_gpu']}")
            notes = sorted({str(v) for v in resources.get("notes", []) if str(v)})
            if notes:
                console.print("N/A details: " + "; ".join(notes))
    if times:
        table(console, "Inference resources - latest run per condition", pd.DataFrame(times), ["model", "condition", "limit", "prep s", "load s", "work s", "ans/s", "tok/s"])
        console.print("prep includes Torch loading; load is Ollama server loading. work excludes reported load.")
    if ram:
        table(console, "Sampled sum RSS peaks (preparation + inference)", pd.DataFrame(ram))
        table(console, "GPU memory: distinct methods, not interchangeable", pd.DataFrame(gpu))
    training = []
    for marker in (Path(root) / "training").glob("*/run.yaml"):
        resources = load_resources(marker.parent)
        main = resources[resources.phase.eq("training")] if not resources.empty else resources
        if main.empty:
            continue
        r = main.iloc[0]
        prep = resources[resources.phase.eq("preparation")]
        training.append({"run": str(r.run_id)[:8], "kind": r.get("run_kind"), "N": r.get("examples"), "steps": r.get("steps"),
                         "prep s": number(prep.iloc[0], "wall_seconds") if not prep.empty else None,
                         "work s": number(r, "wall_seconds"), "ex/s": number(r, "examples_per_second"),
                         "RAM MiB": pd.to_numeric(resources.client_ram_peak_mib, errors="coerce").max(), "VRAM MiB": pd.to_numeric(resources.torch_allocated_peak_mib, errors="coerce").max(),
                         "adapter MiB": number(r, "adapter_mib")})
        console.print(f"Training {str(r.run_id)[:8]}: CPU={r.get('cpu')}; GPU={r.get('gpu')}; adapter={str(r.get('adapter_id'))[:8]}")
    if training:
        table(console, "Training speed (smoke is NOT full training)", pd.DataFrame(training), ["run", "kind", "N", "steps", "prep s", "work s", "ex/s"])
        table(console, "Training memory and adapter size", pd.DataFrame(training), ["run", "RAM MiB", "VRAM MiB", "adapter MiB"])
        console.print("Training ex/s is the Trainer counter; adapter size excludes tokenizer and checkpoints.")
    else:
        console.print("Training resources: N/A (no recorded training runs).")
    console.print("Memory in MiB. Ollama resident memory is not a measured process peak; old resources are N/A.")


def report(root, models=None, console=None):
    from rich.console import Console
    console = console or Console()
    root = Path(root)
    models = list(MODEL_NAMES if models is None else models)
    if any(x not in MODEL_NAMES for x in models):
        raise ValueError(f"Unknown model; choose {MODEL_NAMES}")
    quality, judged, pending, legacy = [], [], [], []
    histories = {}
    for model in models:
        directory = root / model
        if not (directory / "run.yaml").exists():
            console.print(f"{model}: no results")
            continue
        history = runs(directory)
        histories[model] = history
        if history:
            groups = {}
            for entry in history:
                groups.setdefault(entry[0]["conditions_id"], []).append(entry)
            for condition, entries in groups.items():
                summaries = pd.concat([x[3] for x in entries])
                quality.append({"model": model, "condition": condition[:6], "runs": len(entries), "count": int(summaries["count"].iloc[0]),
                                "limit": entries[0][0]["config"]["generation"]["max_new_tokens"] if model != "baseline" else "N/A",
                                **summaries[["corpus_bleu", *METRICS]].mean().to_dict()})
            averages, incomplete = aggregate_judges(directory, history)
            if not averages.empty:
                judged.append(averages)
            if not incomplete.empty:
                pending.append(incomplete)
        else:
            _, _, summary = load_run(directory)
            marker = yaml.safe_load((directory / "run.yaml").read_text(encoding="utf-8"))
            limit = marker.get("config", {}).get("generation", {}).get("max_new_tokens", "unknown") if model != "baseline" else "N/A"
            quality.append({**summary.iloc[0].to_dict(), "condition": "legacy", "runs": 1, "limit": limit})
        known_runs = {x[0]["run_id"] for x in history}
        old = load_judges(directory)
        old = old[~old.run_id.isin(known_runs)]
        for (judge_model, judge_hash, status), group in old.groupby(["judge_model", "judge_hash", "status"]):
            if status not in {"ok", "legacy_unverified"}:
                continue
            legacy.append({"model": model, "judge": judge_model, "N": len(group), "status": status,
                           **{k: pd.to_numeric(group[k], errors="coerce").mean() for k in CRITERIA}})
    if not quality:
        raise ValueError("No model CSV results found. Run evaluate.py first.")
    frame = pd.DataFrame(quality)
    console.print("BLEU: 0-100; ROUGE F1: 0-1; judges: 1-5. Scores are means of compatible runs.")
    table(console, "BLEU - equal run weight", frame, ["model", "condition", "runs", "count", "limit", "corpus_bleu", "bleu"])
    table(console, "ROUGE - equal run weight", frame, ["model", "condition", *METRICS[1:]])
    if pd.to_numeric(frame["limit"], errors="coerce").nunique() > 1:
        console.print("Different output limits: these groups are not equivalent generation conditions.")
    console.print("LLM-as-judge: 1 — очень плохо; 2 — плохо; 3 — приемлемо; 4 — хорошо; 5 — отлично. Выше — лучше.")
    console.print("Faith=faithfulness; Cover=coverage; Relev=relevance; Coher=coherence; Avg=overall.")
    if judged:
        averages = pd.concat(judged, ignore_index=True)
        for row in averages[averages.budget_overrides.gt(0)].itertuples():
            console.print(f"Budget exception: {row.model} / {row.judge}: {row.budget_overrides} score(s) recovered with an explicitly extended max_tokens; actual settings hashes are preserved in judge.csv.")
        for (judge, judge_hash), group in averages.groupby(["judge", "judge_hash"]):
            display = group.assign(condition=group.condition.str[:6])
            table(console, f"{judge} / {judge_hash[:6]} - complete runs, common IDs", display,
                  ["model", "condition", "runs", "N", *CRITERIA])
        joint = []
        for (model, condition), group in averages.groupby(["model", "condition"]):
            if len(group) == 2 and set(group.judge_alias) == {"gpt_oss", "glm_flash"}:
                joint.append({"model": model, "condition": condition[:6], "N": int(group.N.min()),
                              **group[CRITERIA].mean().to_dict()})
        if joint:
            table(console, "Both judges - equal weight 50% / 50%", pd.DataFrame(joint))
        else:
            console.print("Joint judge mean: N/A (two comparable judges required).")
    else:
        console.print("Judge run averages: N/A (no complete scored runs).")
    if pending:
        incomplete = pd.concat(pending)
        for key in ("ok", "expected", "errors"):
            incomplete[key] = pd.to_numeric(incomplete[key], errors="coerce")
        summary = incomplete.groupby(["model", "judge"], as_index=False).agg(
            runs=("run_id", "count"), ok=("ok", lambda s: s.sum(min_count=1)),
            expected=("expected", lambda s: s.sum(min_count=1)), errors=("errors", lambda s: s.sum(min_count=1)))
        table(console, "Incomplete runs EXCLUDED from judge means", summary)
    if legacy:
        table(console, "Historical / legacy scores - NOT pooled into new run means", pd.DataFrame(legacy).drop(columns="judge"))
        for name in sorted({x["judge"] for x in legacy}):
            console.print("Historical judge: " + name)
    resource_report(console, root, histories)
    console.print(f"Individual answers, scores and run columns: {root.resolve()}")
    return frame
