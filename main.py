from src.experiment_pipeline import ChurnExperimentPipeline
from src.utils import configure_logging


def main() -> None:
    configure_logging()
    results = ChurnExperimentPipeline().run()
    display_cols = [c for c in ["feature_set", "model", "validation_accuracy", "test_accuracy", "test_roc_auc", "test_pr_auc", "test_f1", "threshold"] if c in results.columns]
    print(results[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
