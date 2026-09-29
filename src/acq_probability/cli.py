from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from acq_probability.config import load_config
from acq_probability.features import engineer_features
from acq_probability.labels import build_labels
from acq_probability.metrics import enrich_predictions

DEFAULT_CONFIG = Path("config/model.toml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acq-model",
        description="Train and backtest a point-in-time acquisition candidate ranking model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Generate synthetic data and run the full pipeline")
    demo.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    demo.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    demo.add_argument("--n-companies", type=int, default=650)

    train = subparsers.add_parser("train", help="Train and backtest from a point-in-time CSV")
    train.add_argument("--input", type=Path, required=True)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    train.add_argument("--as-of", help="Assert complete outcome coverage through YYYY-MM-DD")

    label = subparsers.add_parser(
        "label", help="Build mature announcement labels from an event register"
    )
    label.add_argument("--input", type=Path, required=True, help="Unlabeled company-date panel")
    label.add_argument(
        "--events", type=Path, required=True, help="Qualifying target announcements CSV"
    )
    label.add_argument("--as-of", required=True, help="Complete event coverage through YYYY-MM-DD")
    label.add_argument("--output", type=Path, default=Path("data/processed/company_panel.csv"))

    score = subparsers.add_parser("score", help="Score one or more new company cross-sections")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, default=Path("outputs/scored_candidates.csv"))
    return parser


def _print_result(output_dir: Path, metrics: dict[str, float | int]) -> None:
    print(f"Artifacts: {output_dir.resolve()}")
    print(
        f"Top {metrics.get('top_fraction', 0.1):.0%} result: "
        f"{metrics['top_decile_acquisition_rate']:.2%} acquired vs "
        f"{metrics['base_rate']:.2%} base rate = "
        f"{metrics['top_decile_lift']:.2f}x lift"
    )


def _score(input_path: Path, model_path: Path, output_path: Path) -> None:
    features = engineer_features(pd.read_csv(input_path), require_target=False)
    bundle = joblib.load(model_path)
    if bundle.get("schema_version") != 3:
        raise ValueError("This model bundle needs rebuilding; rerun demo or train for schema v3")
    if (
        "earliest_score_date" in bundle
        and (features["observation_date"] < pd.Timestamp(bundle["earliest_score_date"])).any()
    ):
        raise ValueError("Scoring dates precede this model's calibration-label availability")
    missing = sorted(set(bundle["features"]) - set(features.columns))
    if missing:
        raise ValueError(f"Scoring data is missing engineered features: {', '.join(missing)}")
    probabilities = bundle["model"].predict_proba(features[bundle["features"]])[:, 1]
    scored = features[["company_id", "company_name", "observation_date", "sector"]].copy()
    scored["predicted_probability"] = probabilities
    scored = enrich_predictions(scored, top_fraction=bundle["top_fraction"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False, date_format="%Y-%m-%d")
    print(f"Scored {len(scored):,} observations: {output_path.resolve()}")
    if bundle.get("data_source") == "synthetic":
        print("Warning: this model was trained on synthetic data; scores are demo-only.")


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in ["input", "config", "model", "events"]:
        path = getattr(args, name, None)
        if path is not None and not path.is_file():
            parser.error(
                f"File not found: {path}. Run from the repository root. "
                "For a self-contained example use: acq-model demo"
            )
    try:
        _run(args)
    except ValueError as exc:
        parser.error(str(exc))


def _run(args: argparse.Namespace) -> None:
    if args.command == "demo":
        from acq_probability.pipeline import run_demo

        config = load_config(args.config)
        result = run_demo(config, args.output_dir, n_companies=args.n_companies)
        _print_result(args.output_dir, result.test_metrics)
    elif args.command == "train":
        from acq_probability.pipeline import run_pipeline

        config = load_config(args.config)
        panel = pd.read_csv(args.input)
        result = run_pipeline(panel, config, args.output_dir, as_of=args.as_of)
        _print_result(args.output_dir, result.test_metrics)
    elif args.command == "score":
        _score(args.input, args.model, args.output)
    elif args.command == "label":
        panel, audit = build_labels(
            pd.read_csv(args.input), pd.read_csv(args.events), as_of=args.as_of
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(args.output, index=False, date_format="%Y-%m-%d")
        print(f"Labeled panel: {args.output.resolve()}")
        for key, value in audit.items():
            print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()
