import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
from text_summary_block_7.judge import parse_scores
from text_summary_block_7.judge import evaluate_judge
from text_summary_block_7.config import ROOT, load_config, select_judges
from text_summary_block_7.artifacts import predictions, load_judges
from text_summary_block_7.judging import evaluate_judges, evaluation_plan


class JudgeTests(unittest.TestCase):
    def test_one_provider_failure_does_not_stop_second_judge(self):
        config = load_config(ROOT / "configs/baseline.yaml")
        selected = select_judges(config)
        df = pd.DataFrame({"id": [str(x) for x in range(150)], "dialogue": ["Hello."] * 150, "summary": ["A greeting."] * 150})
        pred = predictions(df, ["Hi."] * 150, "baseline")
        sample, plan = evaluation_plan(pred, config, selected)
        seen = []
        def evaluate_one(pred, settings, directory, alias, sample, response_attempts):
            seen.append((alias, sample.id.tolist(), settings["prompt"]))
            return len(sample) if alias == "gpt_oss" else 0
        with patch("text_summary_block_7.judging.evaluate_judge", side_effect=evaluate_one):
            errors = evaluate_judges(pred, selected, Path("unused"), sample)
        self.assertEqual(errors, 100)
        self.assertEqual([x[0] for x in seen], ["gpt_oss", "glm_flash"])
        self.assertEqual(seen[0][1:], seen[1][1:])
        self.assertEqual(len(plan["sample_ids"]), 100)

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
        settings = select_judges(load_config(ROOT / "configs/baseline.yaml"))["gpt_oss"]
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as directory, patch("text_summary_block_7.judge.create_client", return_value=client):
            errors = evaluate_judge(pred, settings, Path(directory), response_attempts=1)
            self.assertEqual(errors, 1)
            self.assertEqual(completions.calls, 100)
            judges = load_judges(directory)
            self.assertEqual(len(judges), 100)
            self.assertEqual(judges[judges.status.eq("error")].overall.iloc[0], "")

            settings = {**settings, "sample_size": 1}
            # The first sampled row failed initially; refresh it successfully,
            # then fail again and ensure its last good score survives.
            self.assertEqual(evaluate_judge(pred, settings, Path(directory), response_attempts=1), 0)
            completions.calls = 0
            self.assertEqual(evaluate_judge(pred, settings, Path(directory), response_attempts=1), 1)

    def test_invalid_response_is_retried_with_safe_diagnostics(self):
        replies = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="length")], usage=SimpleNamespace(completion_tokens=1000)),
                   SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"faithfulness":5,"coverage":4,"relevance":3,"coherence":2}'), finish_reason="stop")])]
        from unittest.mock import Mock
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(side_effect=replies))), close=lambda: None)
        df = pd.DataFrame({"id": ["x"], "dialogue": ["Hello."], "summary": ["A greeting."]})
        pred = predictions(df, ["Hi."], "baseline")
        settings = select_judges(load_config(ROOT / "configs/baseline.yaml"))["gpt_oss"]
        with tempfile.TemporaryDirectory(dir=ROOT / ".verification") as directory, patch("text_summary_block_7.judge.create_client", return_value=client):
            self.assertEqual(evaluate_judge(pred, settings, Path(directory)), 0)
            score = load_judges(directory).iloc[0]
            self.assertEqual(score.status, "ok")
            self.assertEqual(score.attempts, 2)
            self.assertEqual(score.error_detail, "empty_response")
            self.assertEqual(client.chat.completions.create.call_count, 2)
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
