from pathlib import Path
import tempfile
import unittest
from text_summary_block_7.config import ROOT, load_config


class ConfigTests(unittest.TestCase):
    def test_override_priority_and_original_unchanged(self):
        path = ROOT / "configs/lora.yaml"
        before = path.read_bytes()
        c = load_config(path, ["training.learning_rate=0.0003", "lora.r=32"], {"training.learning_rate": 0.0002})
        self.assertEqual(c["training"]["learning_rate"], .0002)
        self.assertEqual(c["lora"]["r"], 32)
        self.assertEqual(before, path.read_bytes())

    def test_bad_parameters_fail_before_model_loading(self):
        for override in ["judge.enabled=true", "seed=true", "training.learning_rate=-1", "lora.r=2.5", "generation.think=1"]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load_config(ROOT / "configs/lora.yaml", [override])

    def test_paths_keep_their_origin(self):
        c = load_config(ROOT / "configs/lora.yaml", ["data.test=local.csv"])
        self.assertEqual(Path(c["data"]["test"]), Path.cwd() / "local.csv")
        self.assertEqual(Path(c["data"]["train"]), ROOT / "data/train.csv/train.csv")

    def test_no_strategy_is_a_string_not_yaml_boolean(self):
        c = load_config(ROOT / "configs/lora.yaml", ["training.eval_strategy=no"])
        self.assertEqual(c["training"]["eval_strategy"], "no")
