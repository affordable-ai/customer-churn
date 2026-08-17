from src.experiment_pipeline import ChurnExperimentPipeline
from src.utils import configure_logging


def main() -> None:
    configure_logging()
    results = ChurnExperimentPipeline().run()
    print(results[["feature_set", "model", "validation_pr_auc", "validation_f1", "test_pr_auc", "test_f1", "threshold"]].to_string(index=False))


if __name__ == "__main__":
    main()
