from pathlib import Path

import matplotlib.pyplot as plt
import torch

from predictive_vision.checkpoints import load_checkpoint
from predictive_vision.data import build_data_loaders, prepare_events, select_device
from predictive_vision.training import build_models


def _project_to_image(volume, sample_idx=0):
    sample = volume[sample_idx].detach().cpu()
    return sample.abs().sum(dim=0).sum(dim=0)


def plot_qualitative_raster(raw, expected, sparse, output_path, sample_idx=0):
    raw_2d = _project_to_image(raw, sample_idx=sample_idx)
    expected_2d = _project_to_image(expected, sample_idx=sample_idx)
    sparse_2d = _project_to_image(sparse, sample_idx=sample_idx)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Brain-Inspired Event Predictive Coding: Informative Surprise",
        fontsize=16,
        fontweight="bold",
    )
    axes[0].imshow(raw_2d, cmap="hot", interpolation="nearest")
    axes[0].set_title("1. Raw DVS Events")
    axes[0].axis("off")
    axes[1].imshow(expected_2d, cmap="hot", interpolation="nearest")
    axes[1].set_title("2. Memory-Aware Expectation")
    axes[1].axis("off")
    axes[2].imshow(sparse_2d, cmap="hot", interpolation="nearest")
    axes[2].set_title("3. Regime-Routed Surprise")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def _density(tensor, eps=0.0):
    return (tensor.abs() > eps).float().mean().item()


def plot_quantitative_density(raw, metrics, output_path):
    densities = [
        _density(raw),
        _density(metrics["cleaned"]),
        _density(metrics["residual_mask"]),
        _density(metrics["mask"]),
    ]
    labels = [
        "Raw Input\nO(k)",
        "Noise Gate\nLayer 1",
        "Residual\nLayer 3",
        "Sparse Output\nLayer 4",
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(labels, densities, color=["#d1495b", "#edae49", "#00798c", "#30638e"])
    ax.set_title("Progressive Signal Sparsification", fontsize=14, fontweight="bold")
    ax.set_ylabel("Active Voxel Ratio", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.002,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def run_plotting(run, checkpoint, split="val", sample_index=0):
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    config = checkpoint_payload["config"]
    device = select_device(config.get("training", {}).get("device", "auto"))
    run_dir = Path(run)
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_data_loaders(config, split_path=run_dir / "splits" / "split_indices.json")
    pipeline, baseline_classifier, sparse_classifier = build_models(
        config,
        loaders["num_classes"],
        device,
    )
    load_checkpoint(checkpoint, pipeline, baseline_classifier, sparse_classifier, map_location=device)
    pipeline.eval()

    frames, _ = next(iter(loaders[split]))
    raw_events = prepare_events(
        frames,
        input_channels=int(config.get("model", {}).get("input_channels", 2)),
    ).to(device)
    with torch.no_grad():
        metrics = pipeline(raw_events)

    plot_qualitative_raster(
        raw_events,
        metrics["expected"],
        metrics["output"],
        figure_dir / "figure_2_raster_plot.png",
        sample_idx=sample_index,
    )
    plot_quantitative_density(raw_events, metrics, figure_dir / "figure_3_density_reduction.png")
    print(f"Saved figures to {figure_dir}")
    return figure_dir
