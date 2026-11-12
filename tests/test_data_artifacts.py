import tempfile
from pathlib import Path
import unittest
import pandas as pd
from text_summary_block_7.artifacts import ROOT, safe_path, predictions, classified_judges, JUDGE_COLUMNS, load_run, save_run
from text_summary_block_7.config import load_config
from text_summary_block_7.data import judge_sample, load_data
from text_summary_block_7.metrics import compute_metrics


class DataTests(unittest.TestCase):
    def test_source_paths_are_protected(self):
        for path in [ROOT, ROOT / "notebooks/new.csv", ROOT / "data/test.csv/test.csv", ROOT / "notebooks/models/new"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                safe_path(path)

    def test_judge_sample_is_shared_reproducible_and_bounded(self):
        df = pd.DataFrame({"id": range(1500)})
        a = judge_sample(df)
        self.assertEqual(len(a), 100)
        self.assertEqual(list(a.id), list(judge_sample(df).id))
        self.assertEqual(len(judge_sample(df.head(3))), 3)

    def test_changed_answer_does_not_inherit_old_score(self):
        df = pd.DataFrame({"id": ["a"], "dialogue": ["Hello."], "summary": ["A greeting."]})
        old = predictions(df, ["Hi."], "baseline")
        new = predictions(df, ["Goodbye."], "baseline")
        judges = pd.DataFrame([{**{k: "" for k in JUDGE_COLUMNS}, "id": "a", "model": "baseline", "response_hash": old.response_hash.iloc[0], "status": "ok"}])
        current, history = classified_judges(new, judges)
        self.assertTrue(current.empty)
        self.assertEqual(len(history), 1)

    def test_partial_publication_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as directory:
            c = load_config(ROOT / "configs/baseline.yaml", named={"paths.results_dir": directory})
            df = pd.DataFrame({"id": ["001"], "dialogue": ["Hello."], "summary": ["A greeting."]})
            pred = predictions(df, ["A greeting."], "baseline")
            metrics, summary = compute_metrics(pred, c["metrics"])
            dest = save_run(pred, metrics, summary, c)
            self.assertEqual(load_run(dest)[0].id.iloc[0], "001")
            metrics.run_id = "wrong"
            metrics.to_csv(dest / "metrics.csv", index=False)
            with self.assertRaises(ValueError):
                load_run(dest)

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "input.csv"
            pd.DataFrame({"id": ["a", "a"], "dialogue": ["x", "y"], "summary": ["x", "y"]}).to_csv(p, index=False)
            with self.assertRaises(ValueError):
                load_data(p)
