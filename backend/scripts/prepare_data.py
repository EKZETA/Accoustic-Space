"""Apply a room impulse response and export Librosa features to CSV."""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.signal import fftconvolve

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.services.audio_processor import AudioProcessor

FIELDNAMES = [
    "utterance_id",
    "split",
    "label",
    "duration_seconds",
    "sample_rate",
    "mean_spectral_centroid_hz",
    "mean_spectral_rolloff_hz",
    "mean_rms_energy",
    "reverb_decay_proxy_seconds",
    "spectrogram_frequency_bins",
    "spectrogram_time_frames",
    "spectrogram_values_db",
]
REQUIRED_MANIFEST_COLUMNS = {"audio_path", "utterance_id"}


def load_and_apply_rir(
    audio_path: Path,
    rir_path: Path,
    sample_rate: int,
) -> np.ndarray:
    audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    rir, _ = librosa.load(rir_path, sr=sample_rate, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Audio is empty or contains invalid samples: {audio_path}")
    if rir.size == 0 or not np.isfinite(rir).all():
        raise ValueError(f"RIR is empty or contains invalid samples: {rir_path}")

    rir_peak = float(np.max(np.abs(rir)))
    if rir_peak == 0.0:
        raise ValueError(f"RIR contains only silence: {rir_path}")

    return fftconvolve(audio, rir / rir_peak, mode="full")


def _write_feature_row(
    writer: csv.DictWriter,
    row: dict[str, str],
    dataset_root: Path,
    rir_path: Path,
    processor: AudioProcessor,
) -> None:
    audio_path = dataset_root / row["audio_path"]
    augmented_audio = load_and_apply_rir(
        audio_path,
        rir_path,
        processor.target_sr,
    )
    features = processor.extract_features_from_audio(
        augmented_audio,
        processor.target_sr,
    )
    metadata = features["metadata"]
    spectrogram = features["spectrogram"]
    acoustic_features = features["acoustic_features"]

    writer.writerow(
        {
            "utterance_id": row["utterance_id"],
            "split": row.get("split", ""),
            "label": row.get("label", ""),
            "duration_seconds": metadata["duration_seconds"],
            "sample_rate": metadata["sample_rate"],
            "mean_spectral_centroid_hz": acoustic_features[
                "mean_spectral_centroid_hz"
            ],
            "mean_spectral_rolloff_hz": acoustic_features[
                "mean_spectral_rolloff_hz"
            ],
            "mean_rms_energy": acoustic_features["mean_rms_energy"],
            "reverb_decay_proxy_seconds": acoustic_features[
                "reverb_decay_proxy_seconds"
            ],
            "spectrogram_frequency_bins": spectrogram["frequency_bins"],
            "spectrogram_time_frames": spectrogram["time_frames"],
            "spectrogram_values_db": json.dumps(
                spectrogram["values_db"],
                separators=(",", ":"),
            ),
        }
    )


def generate_rir_features_from_manifest(
    manifest_csv: Path,
    dataset_root: Path,
    rir_file: Path,
    output_csv: Path,
    split: str | None = None,
    limit: int | None = None,
) -> dict[str, int]:
    """Write one row per successfully processed manifest entry."""
    manifest_csv = Path(manifest_csv)
    dataset_root = Path(dataset_root)
    rir_file = Path(rir_file)
    output_csv = Path(output_csv)

    if not manifest_csv.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
    if not rir_file.is_file():
        raise FileNotFoundError(f"RIR file not found: {rir_file}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    processor = AudioProcessor(target_sr=16_000)
    counts = {"processed": 0, "failed": 0}
    temporary_path: Path | None = None

    try:
        with manifest_csv.open(newline="", encoding="utf-8") as manifest:
            reader = csv.DictReader(manifest)
            columns = set(reader.fieldnames or [])
            missing_columns = REQUIRED_MANIFEST_COLUMNS - columns
            if missing_columns:
                raise ValueError(
                    "Manifest is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )

            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=output_csv.parent,
                prefix=f".{output_csv.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                writer = csv.DictWriter(temporary_file, fieldnames=FIELDNAMES)
                writer.writeheader()

                selected = 0
                for row_number, row in enumerate(reader, start=2):
                    if split is not None and row.get("split") != split:
                        continue
                    if limit is not None and selected >= limit:
                        break
                    selected += 1

                    try:
                        _write_feature_row(
                            writer,
                            row,
                            dataset_root,
                            rir_file,
                            processor,
                        )
                        counts["processed"] += 1
                    except (OSError, ValueError, RuntimeError) as error:
                        print(
                            f"Failed manifest row {row_number} "
                            f"({row.get('utterance_id', 'unknown')}): {error}",
                            file=sys.stderr,
                        )
                        counts["failed"] += 1

        if counts["processed"] == 0:
            raise ValueError(
                "No audio entries were processed. Check the manifest, split, "
                "and dataset root."
            )
        temporary_path.replace(output_csv)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return counts


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an RIR to manifest audio, extract Librosa features, "
            "and save them to CSV."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root / "data" / "manifests" / "asvspoof2019_la.csv",
        help="Input CSV manifest.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "asvspoof" / "LA" / "LA",
        help="Root directory used to resolve manifest audio_path values.",
    )
    parser.add_argument(
        "--rir",
        type=Path,
        default=(
            project_root
            / "data"
            / "rir"
            / "RIRS_NOISES"
            / "simulated_rirs"
            / "smallroom"
            / "Room001"
            / "Room001-00001.wav"
        ),
        help="Room impulse response WAV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "rir_extracted_features.csv",
        help="Destination CSV file.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "eval"),
        help="Process only this manifest split (default: all splits).",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="Optional maximum number of selected audio entries to process.",
    )
    args = parser.parse_args()

    counts = generate_rir_features_from_manifest(
        args.manifest,
        args.dataset_root,
        args.rir,
        args.output,
        split=args.split,
        limit=args.limit,
    )
    print(f"Feature CSV written to {args.output}")
    print(f"processed: {counts['processed']}; failed: {counts['failed']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
