import csv
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_asvspoof_subset import package_subset


class ASVspoofSubsetPackageTests(unittest.TestCase):
    def create_dataset(self, root: Path, malicious_path: bool = False) -> Path:
        dataset_root = root / "dataset"
        manifest_path = root / "manifest.csv"
        rows = []
        for split in ("train", "dev", "eval"):
            for label in ("bonafide", "spoof"):
                for index in range(2):
                    relative_path = (
                        f"{split}/{label}-{index}.flac"
                        if not (malicious_path and split == "train" and index == 0)
                        else "../outside.flac"
                    )
                    rows.append(
                        {
                            "split": split,
                            "label": label,
                            "audio_path": relative_path,
                        }
                    )
                    if ".." not in relative_path:
                        audio_path = dataset_root / relative_path
                        audio_path.parent.mkdir(parents=True, exist_ok=True)
                        audio_path.write_bytes(b"test audio")

        with manifest_path.open("w", newline="", encoding="utf-8") as manifest:
            writer = csv.DictWriter(
                manifest,
                fieldnames=["split", "label", "audio_path"],
            )
            writer.writeheader()
            writer.writerows(rows)
        return manifest_path

    def test_packages_only_balanced_audio_and_a_usable_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self.create_dataset(root)
            archive_path = root / "subset.zip"

            file_count, _ = package_subset(
                manifest_path,
                root / "dataset",
                archive_path,
                samples_per_class=2,
            )

            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                rows = list(
                    csv.DictReader(
                        archive.read("manifest.csv").decode("utf-8").splitlines()
                    )
                )
            self.assertEqual(file_count, 12)
            self.assertEqual(len(names), 13)
            self.assertEqual(len(rows), 12)
            self.assertTrue(all(name != "dataset" for name in names))
            self.assertTrue(
                all((root / "dataset" / row["audio_path"]).is_file() for row in rows)
            )

    def test_rejects_audio_paths_outside_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self.create_dataset(root, malicious_path=True)

            with self.assertRaisesRegex(ValueError, "must be relative"):
                package_subset(
                    manifest_path,
                    root / "dataset",
                    root / "subset.zip",
                    samples_per_class=2,
                )

            self.assertFalse((root / "subset.zip").exists())

    def test_does_not_overwrite_an_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = self.create_dataset(root)
            archive_path = root / "subset.zip"
            archive_path.write_bytes(b"keep this file")

            with self.assertRaises(FileExistsError):
                package_subset(
                    manifest_path,
                    root / "dataset",
                    archive_path,
                    samples_per_class=2,
                )

            self.assertEqual(archive_path.read_bytes(), b"keep this file")


if __name__ == "__main__":
    unittest.main()
