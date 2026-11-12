from .data import load_data
from .artifacts import predictions, save_run, safe_path, rebuild_combined
from .models import generator
from .metrics import compute_metrics


def evaluate(c, with_judge=False):
    safe_path(c["paths"]["results_dir"], c["data"].values())
    df = load_data(c["data"]["test"])
    generate = generator(c)
    from tqdm import tqdm
    values = [generate(text) for text in tqdm(df.dialogue, desc=c["model"]["kind"])]
    pred = predictions(df, values, c["model"]["kind"])
    metrics, summary = compute_metrics(pred, c["metrics"])
    directory = save_run(pred, metrics, summary, c)
    errors = 0
    if with_judge:
        from .judge import evaluate_judge
        errors = evaluate_judge(pred, c["judge"], directory)
        rebuild_combined(directory.parent)
    print(f"{c['model']['kind']}: saved {len(pred)} test predictions and BLEU/ROUGE metrics to {directory}")
    if with_judge:
        print(f"LLM-as-judge: {min(len(pred), c['judge']['sample_size'])} answers requested; errors: {errors}")
    print(f'View summary: poetry run python scripts/report.py --results-dir "{directory.parent}"')
    return 1 if errors else 0
