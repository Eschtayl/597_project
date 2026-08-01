"""Standalone entry point for the original Phase 1 preprocessing workflow."""
from pathlib import Path

try:
    from .helpers import feature_cleaner, load_csv, log_and_scale, shuffle_and_separate
except ImportError:  # Preserve ``python phase_1/PreprocessingPipeline.py``.
    from helpers import feature_cleaner, load_csv, log_and_scale, shuffle_and_separate


def main(path=None):
    """Preprocess a sampled packet dataset and print basic integrity checks."""
    root = Path(__file__).resolve().parents[1]
    nested = root / "flow_and_packet" / "packet_based"
    data_path = (
        Path(path)
        if path
        else (nested if nested.is_dir() else root / "packet_based")
    )

    df_combined_sampled = load_csv(str(data_path))
    df_features, labels = shuffle_and_separate(df_combined_sampled)
    df_features_clean, _ = feature_cleaner(df_features)
    df_preprocessed, _ = log_and_scale(df_features_clean, labels)

    print(f"Final preprocessed shape: {df_preprocessed.shape}")
    print(f"Total NaNs in final dataset: {df_preprocessed.isna().sum().sum()}")


if __name__ == "__main__":
    main()
