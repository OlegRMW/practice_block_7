"""Read-only dataset operations."""
import pandas as pd


def load_data(path, training=False):
    df = pd.read_csv(path, dtype={"id": str, "dialogue": str, "summary": str}, keep_default_na=False)
    required = {"id", "dialogue", "summary"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(df.columns))}")
    if training:
        df = df[df.dialogue.str.strip().ne("") & df.summary.str.strip().ne("")].copy()
    if df.empty or df.id.str.strip().eq("").any() or df.id.duplicated().any():
        raise ValueError(f"Dataset must contain unique nonempty IDs: {path}")
    if df.dialogue.str.strip().eq("").any() or df.summary.str.strip().eq("").any():
        raise ValueError(f"Empty dialogue/reference in {path}")
    return df.reset_index(drop=True)


def judge_sample(df, size=100, seed=42):
    return df.sample(n=min(size, len(df)), random_state=seed)
