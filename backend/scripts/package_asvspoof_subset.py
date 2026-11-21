"""Package a balanced ASVspoof sample and its audio files into a ZIP archive."""

import argparse
import csv
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from scripts.sample_balanced_manifest import create_balanced_manifest
else:
    from sample_balanced_manifest import create_balanced_manifest

project_root = Path(__file__).resolve().parents[1]
REQUIRED_COLUMNS = {"audio_path", "label", "split"}


def package_subset(
    manifest_path: Path,
    dataset_root: Path,
    output_path: Path,
    samples_per_class: int = 100,
    seed: int = 42,
) -> tuple[int, int]:
    manifest_path = Path(manifest_path)
    dataset_root = Path(dataset_root).resolve()
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Output archive already exists: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary_directory:
        sampled_manifest = Path(temporary_directory) / "manifest.csv"
        create_balanced_manifest(
            manifest_path,
            sampled_manifest,
            samples_per_class=samples_per_class,
            seed=seed,
        )

        with sampled_manifest.open(newline="", encoding="utf-8") as manifest:
            reader = csv.DictReader(manifest)
            missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(
                    "Manifest is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )
            rows = list(reader)

        audio_files: list[tuple[Path, str]] = []
        for row in rows:
            relative_path = Path(row["audio_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Audio path must be relative to the dataset root: "
                    f"{row['audio_path']!r}"
                )
            audio_path = (dataset_root / relative_path).resolve()
            try:
                archive_path = audio_path.relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(
                    f"Audio path must stay inside the dataset root: "
                    f"{row['audio_path']!r}"
                ) from error
            if not audio_path.is_file():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            audio_files.append((audio_path, archive_path.as_posix()))

        temporary_archive: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".zip",
                dir=output_path.parent,
                delete=False,
            ) as temporary_file:
                temporary_archive = Path(temporary_file.name)

            with zipfile.ZipFile(
                temporary_archive,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=1,
            ) as archive:
                archive.write(sampled_manifest, "manifest.csv")
                for audio_path, archive_path in audio_files:
                    archive.write(audio_path, archive_path)

            if output_path.exists():
                raise FileExistsError(f"Output archive already exists: {output_path}")
            temporary_archive.replace(output_path)
        finally:
            if temporary_archive is not None and temporary_archive.exists():
                temporary_archive.unlink()

    return len(rows), output_path.stat().st_size


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a balanced subset ZIP with its audio and a trainable manifest."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "asvspoof2019_la.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "asvspoof" / "LA" / "LA",
        help="Directory containing the split audio folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "asvspoof_subset_100_seed42.zip",
    )
    parser.add_argument(
        "--samples-per-class",
        type=_positive_int,
        default=100,
        help="Samples per label in each train, dev, and eval split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sample_count, archive_size = package_subset(
        args.manifest,
        args.dataset_root,
        args.output,
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )
    print(f"Subset archive written to {args.output}")
    print(f"Audio files: {sample_count}")
    print(f"Archive size: {archive_size / (1024 * 1024):.1f} MiB")
    print("After extraction, use manifest.csv and the extraction folder as dataset root.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
