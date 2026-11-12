import unittest
import pandas as pd
from text_summary_block_7.artifacts import predictions
from text_summary_block_7.metrics import compute_metrics


class MetricsTests(unittest.TestCase):
    def test_perfect_and_empty_prediction(self):
        text = "Alice has booked a train ticket for tomorrow."
        df = pd.DataFrame({"id": ["a", "b"], "dialogue": [text, text], "summary": [text, text]})
        pred = predictions(df, [text, ""], "baseline")
        metrics, summary = compute_metrics(pred, {"use_stemmer": False, "tokenize": "13a", "smooth_method": "exp"})
        self.assertAlmostEqual(metrics.bleu.iloc[0], 100)
        self.assertEqual(metrics.rouge1.iloc[0], 1)
        self.assertEqual(metrics.bleu.iloc[1], 0)
        self.assertEqual(metrics.rouge1.iloc[1], 0)
        self.assertNotAlmostEqual(summary.corpus_bleu.iloc[0], summary.bleu.iloc[0])
