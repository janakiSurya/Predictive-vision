import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name):
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def prepare_events(frames, input_channels=2):
    frames = torch.as_tensor(frames)
    if frames.ndim != 5:
        raise ValueError(f"Expected 5D event frames, got shape {tuple(frames.shape)}.")

    if frames.shape[1] == input_channels:
        return frames.float()
    if frames.shape[2] == input_channels:
        return frames.permute(0, 2, 1, 3, 4).float()

    raise ValueError(
        "Could not infer event channel dimension. Expected either "
        f"shape [B, {input_channels}, T, H, W] or [B, T, {input_channels}, H, W], "
        f"got {tuple(frames.shape)}."
    )


def active_density(tensor, eps=0.0):
    return (tensor.abs() > eps).float().mean().item()


def build_ncaltech101_dataset(dataset_cfg):
    try:
        import tonic
    except ImportError as exc:
        raise RuntimeError(
            "The tonic package is required to load N-Caltech101. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    def normalize_label(target):
        if isinstance(target, bytes):
            return target.decode("utf-8")
        return target

    temp_dataset = tonic.datasets.NCALTECH101(save_to=dataset_cfg["data_dir"])
    unique_classes = sorted(list(set([normalize_label(target) for _, target in temp_dataset])))
    class_map = {cls_name: index for index, cls_name in enumerate(unique_classes)}

    def safe_label_map(target):
        return class_map[normalize_label(target)]

    dataset = tonic.datasets.NCALTECH101(
        save_to=dataset_cfg["data_dir"],
        target_transform=safe_label_map,
    )
    sensor_size = tuple(dataset_cfg.get("sensor_size", [304, 240, 2]))
    n_time_bins = int(dataset_cfg.get("n_time_bins", 10))
    to_frame = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_time_bins)
    cached_dataset = tonic.DiskCachedDataset(
        dataset,
        transform=to_frame,
        cache_path=dataset_cfg["cache_dir"],
    )
    return cached_dataset, len(class_map)


def make_split_indices(length, fractions, seed):
    if len(fractions) != 3:
        raise ValueError("dataset.split must contain [train, val, test] fractions.")
    fractions = np.asarray(fractions, dtype=np.float64)
    fractions = fractions / fractions.sum()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(length).tolist()
    train_end = int(length * fractions[0])
    val_end = train_end + int(length * fractions[1])
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def load_or_create_splits(length, dataset_cfg, split_path):
    split_path = Path(split_path)
    if split_path.exists():
        with split_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload["dataset_length"] != length:
            raise ValueError(
                f"Split file length {payload['dataset_length']} does not match dataset length {length}."
            )
        return payload["indices"]

    split_path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(dataset_cfg.get("split_seed", dataset_cfg.get("seed", 2026)))
    fractions = dataset_cfg.get("split", [0.7, 0.15, 0.15])
    indices = make_split_indices(length, fractions, seed)
    payload = {
        "dataset_length": length,
        "seed": seed,
        "fractions": fractions,
        "indices": indices,
    }
    with split_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    return indices


def build_data_loaders(config, split_path):
    dataset_cfg = config["dataset"]
    dataset, num_classes = build_ncaltech101_dataset(dataset_cfg)
    indices = load_or_create_splits(len(dataset), dataset_cfg, split_path)

    batch_size = int(dataset_cfg.get("batch_size", 8))
    eval_batch_size = int(dataset_cfg.get("eval_batch_size", batch_size))
    num_workers = int(dataset_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        Subset(dataset, indices["train"]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=torch.utils.data.dataloader.default_collate,
    )
    val_loader = DataLoader(
        Subset(dataset, indices["val"]),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=torch.utils.data.dataloader.default_collate,
    )
    test_loader = DataLoader(
        Subset(dataset, indices["test"]),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=torch.utils.data.dataloader.default_collate,
    )
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "num_classes": num_classes,
    }
