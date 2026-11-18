"""Inference runs are immutable; API judges are strictly opt-in."""
from .data import load_data
from .artifacts import predictions, save_run, safe_path, rebuild_combined, load_judges, write_csv
from .models import generator
from .metrics import compute_metrics
from .config import select_judges
from .generation import GenerationResult
from .resources import ResourceMeter
from .resource_artifacts import adapter_info
from .history import rebuild_wide


def evaluate(c, with_judge=False, judges=None, response_attempts=3):
    safe_path(c["paths"]["results_dir"], c["data"].values())
    selected = select_judges(c) if judges is None else judges
    if with_judge:
        from .judge import api_key
        api_key(c["judge"])
    df = load_data(c["data"]["test"])
    kind = c["model"]["kind"]
    backend = {"baseline": "baseline", "qwen": "ollama", "qwen_lora": "torch"}[kind]
    interval = c["resources"]["sample_interval_seconds"]
    with ResourceMeter(backend, interval, c["model"]["host"]) as preparation:
        generate = generator(c)
        before = generate.info() if hasattr(generate, "info") else {}
    from tqdm import tqdm
    outputs = []
    with ResourceMeter(backend, interval, c["model"]["host"]) as execution:
        for text in tqdm(df.dialogue, desc=kind):
            value = generate(text)
            outputs.append(value if isinstance(value, GenerationResult) else GenerationResult(value))
        after = generate.info() if hasattr(generate, "info") else {}
    pred = predictions(df, [o.text for o in outputs], kind)
    adapter = adapter_info(c["model"]["adapter_path"]) if kind == "qwen_lora" else {}
    identity = {**adapter, **{k: after[k] for k in ("precision", "model_digest", "base_model_revision") if after.get(k) is not None}}
    base = {"run_id": pred.run_id.iloc[0], "operation": "inference", "model": kind, "backend": backend,
            "examples": len(df), "max_new_tokens": c["generation"]["max_new_tokens"] if kind != "baseline" else None,
            **adapter, "precision": after.get("precision", before.get("precision", "N/A")),
            "load_state": before.get("load_state", "not applicable"), "model_digest": after.get("model_digest", "")}
    prep_row = {**base, **preparation.row, "phase": "preparation"}
    main = {**base, **execution.row, "phase": "inference"}
    count = sum(o.generated_tokens for o in outputs) if all(o.generated_tokens is not None for o in outputs) else None
    load = 0.
    if backend == "ollama":
        loads = [o.stats.get("load_duration") for o in outputs]
        load = sum(loads) / 1e9 if all(v is not None for v in loads) else None
        durations = [o.stats.get("eval_duration") for o in outputs]
        decode = sum(durations) / 1e9 if all(v is not None for v in durations) else None
        main["decode_tokens_per_second"] = count / decode if count is not None and decode and decode > 0 else None
        main["model_resident_vram_mib"] = after.get("model_resident_vram_mib")
        main["server_loaded_models"] = after.get("server_loaded_models")
        if max(before.get("server_loaded_models", 0), after.get("server_loaded_models", 0)) > 1:
            main["notes"] += "; shared Ollama server: RAM/process VRAM includes other loaded models"
    seconds = execution.seconds - load if load is not None and execution.seconds >= load else None
    main.update(model_load_seconds=load if backend == "ollama" else None, execution_seconds=seconds, generated_tokens=count,
                answers_per_second=len(df) / seconds if seconds and seconds > 0 else None,
                tokens_per_second=count / seconds if count is not None and seconds and seconds > 0 else None,
                throughput_scope="client wall excluding reported model load; tokenization/network included",
                token_count_method="backend eval_count/output IDs including EOS; baseline N/A")
    metrics, summary = compute_metrics(pred, c["metrics"])
    evaluation = {}
    sample = None
    if with_judge:
        from .judging import evaluation_plan
        sample, evaluation = evaluation_plan(pred, c, selected)
    directory = save_run(pred, metrics, summary, c, resources=[prep_row, main], evaluation=evaluation, adapter=identity)
    errors = 0
    try:
        if with_judge:
            from .judging import evaluate_judges
            errors = evaluate_judges(pred, selected, directory, sample, response_attempts=response_attempts)
    finally:
        judged = load_judges(directory)
        write_csv(judged[judged.run_id.eq(pred.run_id.iloc[0])], directory / "runs" / pred.run_id.iloc[0] / "judge.csv")
        rebuild_combined(directory.parent)
        rebuild_wide(directory)
    print(f"{kind}: saved {len(pred)} test answers; run {pred.run_id.iloc[0]}")
    print(f"Results: {directory}")
    if with_judge:
        print(f"Judges: {', '.join(selected)}; {len(sample)} answers each; errors: {errors}")
    print(f'View summary: poetry run python scripts/report.py --results-dir "{directory.parent}"')
    return 1 if errors else 0
