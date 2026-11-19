from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_summary_block_7.config import parser, load_config, select_judges


def main(argv=None):
    p = parser("Predict all test answers and evaluate; judge API is explicitly opt-in")
    p.add_argument("--test")
    p.add_argument("--adapter-path")
    p.add_argument("--output-dir")
    p.add_argument("--llm-as-judge", action="store_true", help="Call both configured judges on 100 dialogues by default")
    p.add_argument("--judge-sample-size", type=int)
    p.add_argument("--judges", nargs="+", metavar="JUDGE", help="gpt_oss and/or glm_flash; requires --llm-as-judge")
    p.add_argument("--retry-failed", action="store_true", help="Complete missing judge scores of an existing run; no inference")
    p.add_argument("--run-id", help="Run to repair; defaults to the latest run of the selected model")
    p.add_argument("--judge-attempts", type=int, default=3, help="Attempts for empty/invalid judge responses (default: 3)")
    p.add_argument("--recovery-max-tokens", type=int, help="Explicitly extend the token budget for missing scores only; requires --retry-failed")
    a = p.parse_args(argv)
    try:
        c = load_config(a.config, a.set, {"data.test": a.test, "model.adapter_path": a.adapter_path, "paths.results_dir": a.output_dir, "judge.sample_size": a.judge_sample_size})
        if a.judges and not a.llm_as_judge:
            raise ValueError("--judges requires --llm-as-judge")
        if a.judge_attempts < 1:
            raise ValueError("--judge-attempts must be positive")
        if a.run_id and not a.retry_failed:
            raise ValueError("--run-id requires --retry-failed")
        if a.recovery_max_tokens is not None and not a.retry_failed:
            raise ValueError("--recovery-max-tokens requires --retry-failed")
        if a.retry_failed:
            if not a.llm_as_judge:
                raise ValueError("--retry-failed requires --llm-as-judge")
            if a.test or a.adapter_path or a.judge_sample_size or a.set:
                raise ValueError("Recovery uses the recorded snapshot/settings; overrides are not allowed")
            from text_summary_block_7.recovery import retry_failed
            return retry_failed(c, a.run_id, a.judges, a.judge_attempts, a.recovery_max_tokens)
        selected = select_judges(c, a.judges)
        from text_summary_block_7.evaluate import evaluate
        return evaluate(c, with_judge=a.llm_as_judge, judges=selected, response_attempts=a.judge_attempts)
    except (OSError, ValueError, RuntimeError, ImportError, LookupError) as e:
        p.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
