import io
from pathlib import Path
import tempfile
import unittest
import pandas as pd
from rich.console import Console
from text_summary_block_7.artifacts import ROOT, predictions, save_run, load_judges, write_csv, JUDGE_COLUMNS, CRITERIA, read_csv, rebuild_combined
from text_summary_block_7.config import load_config, select_judges
from text_summary_block_7.metrics import compute_metrics
from text_summary_block_7.history import runs, rebuild_wide, aggregate_judges
from text_summary_block_7.judging import evaluation_plan
from text_summary_block_7.reports import report


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / ".verification")
        self.directory = Path(self.temp.name)
        self.df = pd.DataFrame({"id": ["a", "b"], "dialogue": ["Hi.", "Hello."], "summary": ["A greeting.", "A greeting."]})
        path = self.directory / "test.csv"
        self.df.to_csv(path, index=False)
        self.config = load_config(ROOT / "configs/baseline.yaml", named={"data.test": str(path), "paths.results_dir": str(self.directory / "results")})

    def tearDown(self):
        self.temp.cleanup()

    def add(self, run_id, scores, partial=False, changed=False):
        c = self.config
        pred = predictions(self.df, ["Hi." if changed else "A greeting."] * 2, "baseline", run_id)
        selected = select_judges(c, list(scores)) if scores else {}
        _, plan = evaluation_plan(pred, c, selected)
        metric, summary = compute_metrics(pred, c["metrics"])
        d = save_run(pred, metric, summary, c, evaluation=plan if scores else {})
        old = load_judges(d)
        entries = []
        for alias, score in scores.items():
            for row in pred.itertuples():
                if partial and row.id == "b":
                    continue
                entries.append({"id": row.id, "model": "baseline", "response_hash": row.response_hash,
                                **{k: score for k in CRITERIA}, "judge_model": plan["judges"][alias]["model"], "judge_hash": plan["judges"][alias]["hash"],
                                "recorded_at": run_id, "status": "ok", "error": "", "source": "test", "run_id": run_id, "judge_alias": alias})
        write_csv(pd.concat([old, pd.DataFrame(entries, columns=JUDGE_COLUMNS)], ignore_index=True), d / "judge.csv")
        rebuild_wide(d)
        rebuild_combined(d.parent)
        return d

    def test_runs_have_independent_columns_and_equal_judge_weights(self):
        self.add("r1", {"gpt_oss": 2, "glm_flash": 4})
        self.add("r2", {"gpt_oss": 4, "glm_flash": 2}, changed=True)
        self.add("r3", {"gpt_oss": 5})
        self.add("r4", {"gpt_oss": 1}, partial=True)
        d = self.add("r5", {})
        wide = read_csv(d / "scores_by_run.csv")
        self.assertEqual(wide["run_r1__gpt_oss__faithfulness"].tolist(), [2, 2])
        self.assertEqual(wide["run_r2__glm_flash__overall"].tolist(), [2, 2])
        self.assertTrue(wide["run_r5__gpt_oss__overall"].eq("").all())
        self.assertEqual(len(runs(d)), 5)
        averaged, incomplete = aggregate_judges(d, runs(d))
        gpt = averaged[averaged.judge_alias.eq("gpt_oss")].iloc[0]
        glm = averaged[averaged.judge_alias.eq("glm_flash")].iloc[0]
        self.assertEqual(gpt.runs, 3)
        self.assertAlmostEqual(gpt.overall, 11/3)
        self.assertEqual(glm.runs, 2)
        self.assertEqual(glm.overall, 3)
        self.assertAlmostEqual(averaged.overall.mean(), 10/3)
        self.assertEqual(len(incomplete), 1)
        output = io.StringIO()
        report(d.parent, ["baseline"], Console(file=output, width=100))
        self.assertIn("50% / 50%", output.getvalue())
        self.assertIn("Incomplete", output.getvalue())

    def test_combined_keeps_both_judges_and_changed_limits_separate(self):
        d = self.add("r1", {"gpt_oss": 2, "glm_flash": 4})
        combined = read_csv(d.parent / "all_models_metrics.csv")
        self.assertEqual(len(combined), 4)
        self.assertEqual(combined.judge_model.nunique(), 2)
        self.config["generation"]["max_new_tokens"] = 128
        self.add("r2", {"gpt_oss": 5, "glm_flash": 5})
        averaged, _ = aggregate_judges(d, runs(d))
        self.assertEqual(averaged.condition.nunique(), 2)
        self.assertEqual(len(averaged), 4)

    def test_duplicate_run_cannot_overwrite_snapshot(self):
        self.add("r1", {})

    def test_yaml_key_order_does_not_split_equivalent_conditions(self):
        self.add("r1", {"gpt_oss": 2, "glm_flash": 4})
        self.config["generation"] = dict(reversed(list(self.config["generation"].items())))
        d = self.add("r2", {"gpt_oss": 4, "glm_flash": 2})
        averaged, _ = aggregate_judges(d, runs(d))
        self.assertEqual(averaged.condition.nunique(), 1)
        self.assertTrue(averaged.runs.eq(2).all())
        with self.assertRaises(ValueError):
            self.add("r1", {})
