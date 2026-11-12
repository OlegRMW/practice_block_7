"""Terminal reports from CSV only. No inference, judge or GPU imports."""
from pathlib import Path
import pandas as pd
from .config import MODEL_NAMES
from .artifacts import METRICS, CRITERIA, load_run, load_judges, classified_judges


def table(console, title, frame, columns=None):
    from rich.table import Table
    t = Table(title=title, show_lines=False, expand=False)
    frame = frame if columns is None else frame[columns]
    labels = {"model": "Model", "count": "N", "corpus_bleu": "BLEU corpus", "bleu": "BLEU sent.",
              "rouge1": "R-1", "rouge2": "R-2", "rougeL": "R-L", "rougeLsum": "R-Lsum",
              "faithfulness": "Faith", "coverage": "Cover", "relevance": "Relev", "coherence": "Coher", "overall": "Avg", "evaluated": "N judge", "errors": "Err", "judge_hash": "Judge ID"}
    for col in frame.columns:
        t.add_column(labels.get(col, str(col)), overflow="fold")
    for values in frame.itertuples(index=False, name=None):
        t.add_row(*[("—" if pd.isna(v) or v == "" else f"{v:.4f}" if isinstance(v, float) else str(v)) for v in values])
    console.print(t)


def report(root, models=None, console=None):
    from rich.console import Console
    console = console or Console()
    root = Path(root)
    models = list(MODEL_NAMES if models is None else models)
    if any(x not in MODEL_NAMES for x in models):
        raise ValueError(f"Unknown model; choose {MODEL_NAMES}")
    runs, summaries = [], []
    for model in models:
        directory = root / model
        if not (directory / "run.yaml").exists():
            console.print(f"{model}: no results")
            continue
        pred, metrics, summary = load_run(directory)
        current, history = classified_judges(pred, load_judges(directory))
        summaries.append(summary)
        runs.append((model, pred, metrics, current, history))
    if not runs:
        raise ValueError("No model CSV results found. Prepare results or run evaluate.py first.")
    console.print("[bold]BLEU: 0–100; ROUGE F1: 0–1; judge: 1–5[/bold]")
    table(console, "BLEU — full test", pd.concat(summaries), ["model", "count", "corpus_bleu", "bleu"])
    table(console, "ROUGE F1 — full test", pd.concat(summaries), ["model", "count", *METRICS[1:]])
    identities = [set(x[1].id) for x in runs]
    if any(ids != identities[0] for ids in identities):
        console.print("[yellow]Different input ID sets: scores are not directly comparable.[/yellow]")
    judge_summary = []
    for model, pred, metrics, current, history in runs:
        if current.empty:
            judge_summary.append({"model": model, "judge_model": "—", "evaluated": 0, "errors": 0, "judge_hash": "", **{k: None for k in CRITERIA}})
        else:
            for judge_hash, group in current.groupby("judge_hash", dropna=False):
                ok = group[group.status.eq("ok")]
                judge_summary.append({"model": model, "judge_model": group.judge_model.iloc[0], "evaluated": len(ok), "errors": int(group.status.ne("ok").sum()),
                                      **{k: pd.to_numeric(ok[k]).mean() for k in CRITERIA}, "judge_hash": str(judge_hash)[:8]})
    console.print("Faith = faithfulness; Cover = coverage; Relev = relevance; Coher = coherence; Avg = overall")
    for judge_hash, group in pd.DataFrame(judge_summary).groupby("judge_hash", dropna=False):
        table(console, f"LLM-as-judge — current answers (judge {judge_hash or 'not run'})", group, ["model", "evaluated", "errors", *CRITERIA])
    historical_summary = []
    for model, _, _, _, history in runs:
        if not history.empty:
            for (source, judge_model, judge_hash), group in history.groupby(["source", "judge_model", "judge_hash"]):
                ok = group[group.status.isin(["ok", "legacy_unverified"])]
                historical_summary.append({"model": model, "judge_hash": str(judge_hash)[:8], "evaluated": len(ok), **{k: pd.to_numeric(ok[k]).mean() for k in CRITERIA}})
                console.print(f"Historical source ({model}, {judge_model}, {str(judge_hash)[:8]}): {source}")
    if historical_summary:
        for judge_hash, group in pd.DataFrame(historical_summary).groupby("judge_hash"):
            table(console, f"LLM-as-judge — historical {judge_hash}, NOT current scores", group, ["model", "evaluated", *CRITERIA])
    console.print(f"Per-answer metrics are saved in CSV: {root.resolve()}")
    return pd.concat(summaries, ignore_index=True)
