from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_INPUT = Path("data/hdnet/merged_train.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split merged_train.npz into train and validation subsets. "
            "By default, each Subject is split internally by the train ratio."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input npz file. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: same directory as input file.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Training split ratio. Default: 0.9",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for shuffling. Default: 42",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Disable random shuffling within each Subject.",
    )
    parser.add_argument(
        "--compressed",
        action="store_true",
        help="Use np.savez_compressed for outputs. This is slower but saves disk space.",
    )
    return parser.parse_args()


def validate_arrays(data: np.lib.npyio.NpzFile) -> int:
    if not data.files:
        raise ValueError("Input npz file does not contain any arrays.")

    sample_count = data[data.files[0]].shape[0]
    invalid = [
        (key, data[key].shape)
        for key in data.files
        if data[key].shape[0] != sample_count
    ]
    if invalid:
        details = ", ".join(f"{key}: {shape}" for key, shape in invalid)
        raise ValueError(
            "All arrays must have the same first dimension. Mismatched arrays: "
            f"{details}"
        )
    return sample_count


def make_per_subject_split_indices(
    subjects: np.ndarray,
    train_ratio: float,
    seed: int,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if subjects.ndim != 1:
        raise ValueError("Subject array must be one-dimensional.")

    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []

    unique_subjects = np.unique(subjects)
    for subject in unique_subjects:
        subject_indices = np.flatnonzero(subjects == subject)
        if shuffle:
            subject_indices = subject_indices.copy()
            rng.shuffle(subject_indices)

        train_count = int(len(subject_indices) * train_ratio)
        if len(subject_indices) > 1:
            train_count = min(max(train_count, 1), len(subject_indices) - 1)

        train_parts.append(subject_indices[:train_count])
        val_parts.append(subject_indices[train_count:])

    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)

    if shuffle:
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

    return train_indices, val_indices


def save_subset(
    source: np.lib.npyio.NpzFile,
    indices: np.ndarray,
    output_path: Path,
    compressed: bool,
) -> None:
    arrays = {}
    for key in source.files:
        print(f"  slicing {key}: {len(indices)} samples")
        arrays[key] = source[key][indices]

    saver = np.savez_compressed if compressed else np.savez
    saver(output_path, **arrays)


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    output_dir = (args.output_dir or input_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.npz"
    val_path = output_dir / "val.npz"
    indices_path = output_dir / "split_indices.npz"

    print(f"Loading: {input_path}")
    with np.load(input_path, allow_pickle=False) as data:
        if "Subject" not in data.files:
            raise KeyError("Input npz file must contain a 'Subject' array.")

        sample_count = validate_arrays(data)
        train_indices, val_indices = make_per_subject_split_indices(
            subjects=data["Subject"],
            train_ratio=args.train_ratio,
            seed=args.seed,
            shuffle=not args.no_shuffle,
        )

        print(f"Total samples: {sample_count}")
        print(f"Total subjects: {len(np.unique(data['Subject']))}")
        print(f"Train samples: {len(train_indices)}")
        print(f"Validation samples: {len(val_indices)}")
        print("Split mode: per Subject")
        print(f"Shuffle within each Subject: {not args.no_shuffle}")
        print(f"Seed: {args.seed}")

        np.savez(
            indices_path,
            train_indices=train_indices,
            val_indices=val_indices,
            seed=np.array(args.seed),
            train_ratio=np.array(args.train_ratio),
            shuffled=np.array(not args.no_shuffle),
            split_mode=np.array("per_subject"),
        )
        print(f"Saved indices: {indices_path}")

        print(f"Saving train split: {train_path}")
        save_subset(data, train_indices, train_path, args.compressed)

        print(f"Saving validation split: {val_path}")
        save_subset(data, val_indices, val_path, args.compressed)

    print("Done.")


if __name__ == "__main__":
    main()

