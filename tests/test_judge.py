import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from text_summary_block_7.judge import parse_scores
from text_summary_block_7.judge import evaluate_judge
from text_summary_block_7.config import ROOT, load_config
from text_summary_block_7.artifacts import predictions, load_judges


class JudgeTests(unittest.TestCase):
    def test_only_100_calls_and_error_has_no_fake_score(self):
        class Completions:
            calls = 0

            def create(self, **kwargs):
                self.calls += 1
                text = 'invalid' if self.calls == 1 else '{"faithfulness":5,"coverage":4,"relevance":3,"coherence":2}'
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
        completions = Completions()
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions), close=lambda: None)
        df = pd.DataFrame({"id": [str(x) for x in range(150)], "dialogue": ["Hello."] * 150, "summary": ["A greeting."] * 150})
        pred = predictions(df, ["Hi."] * 150, "baseline")
        settings = load_config(ROOT / "configs/baseline.yaml")["judge"]
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as directory, patch("text_summary_block_7.judge.create_client", return_value=client):
            errors = evaluate_judge(pred, settings, Path(directory))
            self.assertEqual(errors, 1)
            self.assertEqual(completions.calls, 100)
            judges = load_judges(directory)
            self.assertEqual(len(judges), 100)
            self.assertEqual(judges[judges.status.eq("error")].overall.iloc[0], "")

            settings = {**settings, "sample_size": 1}
            # The first sampled row failed initially; refresh it successfully,
            # then fail again and ensure its last good score survives.
            self.assertEqual(evaluate_judge(pred, settings, Path(directory)), 0)
            completions.calls = 0
            self.assertEqual(evaluate_judge(pred, settings, Path(directory)), 1)
            refreshed = load_judges(directory)
            sample_id = pred.sample(n=1, random_state=42).id.iloc[0]
            self.assertEqual(len(refreshed[refreshed.id.eq(sample_id) & refreshed.status.eq("ok")]), 1)

    def test_scores_and_markdown(self):
        scores = parse_scores('```json\n{"faithfulness":5,"coverage":4,"relevance":3,"coherence":2}\n```')
        self.assertEqual(scores["overall"], 3.5)

    def test_invalid_scores(self):
        for content in ["", "[]", '{"faithfulness":true}', '{"faithfulness":6,"coverage":4,"relevance":3,"coherence":2}', '{"faithfulness":1.5,"coverage":4,"relevance":3,"coherence":2}']:
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_scores(content)
