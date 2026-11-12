from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from text_summary_block_7.config import parser, load_config


def main(argv=None):
    p = parser("Fine-tune Qwen with a LoRA adapter")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--train-size", type=int)
    p.add_argument("--output-dir")
    a = p.parse_args(argv)
    try:
        c = load_config(a.config, a.set, {"training.learning_rate": a.learning_rate, "training.num_train_epochs": a.epochs, "training.train_size": a.train_size, "paths.models_dir": a.output_dir})
        from text_summary_block_7.train import train
        print(f"Saved adapter: {train(c)}")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, LookupError) as e:
        p.exit(2, f"error: {e}\n")


if __name__ == "__main__":
    raise SystemExit(main())
