from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_summary_block_7.config import ROOT, MODEL_NAMES


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare saved model CSV metrics; no model/API calls")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    a = p.parse_args(argv)
    try:
        from text_summary_block_7.reports import report
        report(a.results_dir, a.models)
        return 0
    except (OSError, ValueError, KeyError, ImportError) as e:
        p.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
