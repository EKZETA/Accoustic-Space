"""Create a CSV manifest from the official ASVspoof 2019 LA protocols."""

import argparse
import csv
from pathlib import Path

SPLITS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
SPLIT_AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}
FIELDNAMES = [
    "split",
    "speaker_id",
    "utterance_id",
    "attack_id",
    "label",
    "audio_path",
]


def create_manifest(dataset_root: Path, output_path: Path) -> dict[str, int]:
    protocol_dir = dataset_root / "ASVspoof2019_LA_cm_protocols"
    entries: list[dict[str, str]] = []
    counts = {split: 0 for split in SPLITS}
    missing_audio = 0

    for split, protocol_name in SPLITS.items():
        protocol_path = protocol_dir / protocol_name
        if not protocol_path.is_file():
            raise FileNotFoundError(f"ASVspoof protocol not found: {protocol_path}")

        audio_dir = SPLIT_AUDIO_DIRS[split]
        with protocol_path.open(encoding="utf-8") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(
                        f"Expected 5 fields in {protocol_path} at line {line_number}."
                    )

                speaker_id, utterance_id, _, attack_id, label = fields
                if label not in {"bonafide", "spoof"}:
                    raise ValueError(
                        f"Unknown label {label!r} in {protocol_path} "
                        f"at line {line_number}."
                    )

                relative_audio_path = (
                    Path(audio_dir) / "flac" / f"{utterance_id}.flac"
                )
                if not (dataset_root / relative_audio_path).is_file():
                    missing_audio += 1

                entries.append(
                    {
                        "split": split,
                        "speaker_id": speaker_id,
                        "utterance_id": utterance_id,
                        "attack_id": "" if attack_id == "-" else attack_id,
                        "label": label,
                        "audio_path": relative_audio_path.as_posix(),
                    }
                )
                counts[split] += 1

    if missing_audio:
        raise FileNotFoundError(
            f"{missing_audio} protocol entries have no corresponding FLAC file "
            f"under {dataset_root}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as manifest:
        writer = csv.DictWriter(manifest, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(entries)

    return counts


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a labeled ASVspoof 2019 LA CSV manifest."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "data" / "asvspoof" / "LA" / "LA",
        help="Directory containing ASVspoof2019_LA_cm_protocols and split folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "manifests" / "asvspoof2019_la.csv",
        help="Destination CSV path (defaults to backend/data/manifests).",
    )
    args = parser.parse_args()

    counts = create_manifest(args.dataset_root, args.output)
    print(f"Manifest written to {args.output}")
    for split, count in counts.items():
        print(f"{split}: {count} samples")
    print(f"total: {sum(counts.values())} samples")


if __name__ == "__main__":
    main()
