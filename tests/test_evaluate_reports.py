import io
from contextlib import redirect_stdout
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from rich.console import Console
from text_summary_block_7.config import ROOT, load_config
from text_summary_block_7.artifacts import load_judges, write_csv, JUDGE_COLUMNS
from text_summary_block_7.evaluate import evaluate
from text_summary_block_7.reports import report


class EvaluateTests(unittest.TestCase):
    def test_missing_key_is_detected_before_inference(self):
        c = load_config(ROOT / "configs/baseline.yaml")
        with patch("text_summary_block_7.judge.api_key", side_effect=ValueError("Missing key")), patch("text_summary_block_7.evaluate.generator") as generate:
            with self.assertRaises(ValueError):
                evaluate(c, with_judge=True)
            generate.assert_not_called()

    def test_no_judge_without_flag_and_history_survives(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as directory:
            path = Path(directory)
            dataset = path / "test.csv"
            answer_id = "answer_should_stay_in_csv"
            pd.DataFrame({"id": [answer_id], "dialogue": ["Alice booked a train."], "summary": ["A train was booked."]}).to_csv(dataset, index=False)
            c = load_config(ROOT / "configs/baseline.yaml", named={"data.test": str(dataset), "paths.results_dir": str(path / "results")})
            output = io.StringIO()
            with patch("text_summary_block_7.judge.create_client", side_effect=AssertionError("Unexpected API")), patch("text_summary_block_7.reports.report") as report_mock, redirect_stdout(output):
                self.assertEqual(evaluate(c), 0)
                judge_path = path / "results/baseline/judge.csv"
                write_csv(pd.DataFrame([{**{k: "" for k in JUDGE_COLUMNS}, "id": answer_id, "model": "baseline", "status": "legacy_unverified", "overall": 4}], columns=JUDGE_COLUMNS), judge_path)
                before = judge_path.read_bytes()
                self.assertEqual(evaluate(c), 0)
                self.assertEqual(judge_path.read_bytes(), before)
                report_mock.assert_not_called()
            self.assertLessEqual(len(output.getvalue().splitlines()), 8)
            stream = io.StringIO()
            report(path / "results", ["baseline"], Console(file=stream, width=160, force_terminal=False))
            self.assertIn("historical", stream.getvalue().lower())
            self.assertNotIn("per-answer BLEU", stream.getvalue())
            self.assertNotIn(answer_id, stream.getvalue())
            self.assertLess(len(stream.getvalue().splitlines()), 100)

    def test_report_does_not_import_neural_or_api_libraries(self):
        code = "import sys; from text_summary_block_7.reports import report; assert not any(x in sys.modules for x in ['torch','unsloth','openai','ollama','pynvml','psutil']); print('ok')"
        result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
