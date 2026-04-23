from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ensure_directories, load_config
from src.data.split_dataset import split_manifest
from src.data.verify_annotations import verify_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ALPR dataset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_directories(config)

    raw_manifest = Path(config["paths"]["raw_manifest"])
    processed_dir = Path(config["paths"]["processed_dir"])
    artifacts_dir = Path(config["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    clean_df, errors = verify_manifest(
        manifest_path=raw_manifest,
        columns=config["data"]["csv_columns"],
        strict_plate_pattern=False,
    )

    clean_manifest = processed_dir / "clean_manifest.csv"
    processed_dir.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(clean_manifest, index=False)

    train_csv, val_csv, test_csv = split_manifest(
        manifest_path=clean_manifest,
        output_dir=processed_dir,
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
        test_ratio=float(config["data"]["test_ratio"]),
        seed=int(config["seed"]),
    )

    if errors:
        error_file = artifacts_dir / "annotation_errors.txt"
        error_file.write_text("\n".join(errors), encoding="utf-8")
        print(f"[prepare_data] {len(errors)} rows were removed. See: {error_file}")

    print("[prepare_data] Completed")
    print(f"- clean manifest: {clean_manifest}")
    print(f"- train split: {train_csv}")
    print(f"- val split: {val_csv}")
    print(f"- test split: {test_csv}")


if __name__ == "__main__":
    main()
