from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_summary_block_7.config import parser, load_config


def main(argv=None):
    p = parser("Predict all test answers and evaluate; judge API is explicitly opt-in")
    p.add_argument("--test")
    p.add_argument("--adapter-path")
    p.add_argument("--output-dir")
    p.add_argument("--llm-as-judge", action="store_true", help="Call RouterAI for 100 test dialogues (default)")
    p.add_argument("--judge-sample-size", type=int)
    a = p.parse_args(argv)
    try:
        c = load_config(a.config, a.set, {"data.test": a.test, "model.adapter_path": a.adapter_path, "paths.results_dir": a.output_dir, "judge.sample_size": a.judge_sample_size})
        from text_summary_block_7.evaluate import evaluate
        return evaluate(c, with_judge=a.llm_as_judge)
    except (OSError, ValueError, RuntimeError, ImportError, LookupError) as e:
        p.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
