from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import pandas as pd
from text_summary_block_7.artifacts import ROOT, CRITERIA, JUDGE_COLUMNS, predictions, save_run, write_csv, load_judges, read_csv
from text_summary_block_7.config import load_config, select_judges
from text_summary_block_7.judging import evaluation_plan
from text_summary_block_7.metrics import compute_metrics
from text_summary_block_7.history import aggregate_judges, runs
from text_summary_block_7.recovery import retry_failed


class RecoveryTests(unittest.TestCase):
    def test_only_missing_score_is_requested_and_successes_are_unchanged(self):
        self._assert_recovery(None)

    def test_extended_budget_preserves_actual_hash_and_completes_original_run(self):
        self._assert_recovery(4000)

    def _assert_recovery(self, limit):
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as tmp:
            root = Path(tmp)
            df = pd.DataFrame({"id": ["a", "b"], "dialogue": ["Hi.", "Hello."], "summary": ["A greeting."] * 2})
            source = root / "test.csv"
            df.to_csv(source, index=False)
            config = load_config(ROOT / "configs/baseline.yaml", named={"data.test": str(source), "paths.results_dir": str(root / "results")})
            pred = predictions(df, ["A greeting."] * 2, "baseline", "recovery_test")
            selected = select_judges(config)
            _, plan = evaluation_plan(pred, config, selected)
            metrics, summary = compute_metrics(pred, config["metrics"])
            directory = save_run(pred, metrics, summary, config, evaluation=plan)
            entries = []
            for alias in selected:
                for row in pred.itertuples():
                    bad = alias == "glm_flash" and row.id == "b"
                    entries.append({"id": row.id, "model": row.model, "response_hash": row.response_hash,
                                    **{k: None if bad else 4 for k in CRITERIA}, "judge_model": selected[alias]["model"],
                                    "judge_hash": plan["judges"][alias]["hash"], "recorded_at": "original", "status": "error" if bad else "ok",
                                    "error": "ValueError" if bad else "", "source": "test", "run_id": "recovery_test", "judge_alias": alias})
            write_csv(pd.DataFrame(entries, columns=JUDGE_COLUMNS), directory / "judge.csv")
            original = load_judges(directory)
            success = original[original.status.eq("ok")].copy()
            create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"faithfulness":5,"coverage":4,"relevance":3,"coherence":2}'), finish_reason="stop")]))
            client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
            with patch("text_summary_block_7.recovery.api_key", return_value="test"), patch("text_summary_block_7.judge.create_client", return_value=client):
                self.assertEqual(retry_failed(config, "recovery_test", recovery_max_tokens=limit), 0)
            self.assertEqual(create.call_count, 1)
            self.assertEqual(create.call_args.kwargs["model"], "z-ai/glm-5.3-flash")
            self.assertEqual(create.call_args.kwargs["max_tokens"], limit or 1000)
            result = load_judges(directory)
            from text_summary_block_7.judge import settings_hash, matching_settings
            repaired = result[result.id.eq("b") & result.judge_alias.eq("glm_flash")].iloc[0]
            self.assertEqual(repaired.judge_hash, settings_hash({**selected["glm_flash"], "max_tokens": limit or 1000}))
            if limit:
                self.assertNotEqual(repaired.judge_hash, repaired.planned_judge_hash)
                self.assertEqual(repaired.planned_judge_hash, plan["judges"]["glm_flash"]["hash"])
                self.assertEqual(repaired.recovery_note, "max_tokens_override")
                tampered = result.loc[[repaired.name]].copy()
                tampered["request_max_tokens"] = 2000
                self.assertFalse(matching_settings(tampered, selected["glm_flash"]).any())
                tampered["request_max_tokens"] = 1000
                tampered["judge_hash"] = plan["judges"]["glm_flash"]["hash"]
                self.assertFalse(matching_settings(tampered, selected["glm_flash"]).any())
            self.assertEqual(len(result[result.status.eq("ok")]), 4)
            for row in success.to_dict("records"):
                match = result[result.id.eq(row["id"]) & result.judge_model.eq(row["judge_model"])].iloc[0]
                for key in [*CRITERIA, "recorded_at", "response_hash"]:
                    self.assertEqual(float(match[key]), float(row[key])) if key in CRITERIA else self.assertEqual(match[key], row[key])
            averages, incomplete = aggregate_judges(directory, runs(directory))
            self.assertEqual(len(averages), 2)
            self.assertEqual(averages.budget_overrides.sum(), int(limit is not None))
            self.assertTrue(incomplete.empty)
            self.assertEqual(len(runs(directory)), 1)
            wide = read_csv(directory / "scores_by_run.csv")
            self.assertEqual(wide["run_recovery_test__glm_flash__faithfulness"].tolist(), [4, 5])
            with patch("text_summary_block_7.judge.create_client", side_effect=AssertionError("Must not rescore")):
                self.assertEqual(retry_failed(config, "recovery_test"), 0)
