import torch
import torch.nn as nn

from predictive_vision.checkpoints import load_checkpoint
from predictive_vision.config import deep_update, load_config, save_json, save_yaml
from predictive_vision.data import prepare_events
from predictive_vision.data import build_data_loaders, select_device, set_seed
from predictive_vision.metrics import empty_metric_accumulator, finalize_metrics, update_metric_accumulator


def evaluate_models(pipeline, baseline_classifier, sparse_classifier, loader, device, config):
    model_cfg = config.get("model", {})
    input_channels = int(model_cfg.get("input_channels", 2))
    criterion = nn.CrossEntropyLoss()

    pipeline.eval()
    baseline_classifier.eval()
    sparse_classifier.eval()

    acc = empty_metric_accumulator()
    raw_loss_total = 0.0
    sparse_loss_total = 0.0
    with torch.no_grad():
        for frames, labels in loader:
            raw_events = prepare_events(frames, input_channels=input_channels).to(device)
            labels = labels.to(device).long()

            pipeline_out = pipeline(raw_events)
            raw_logits = baseline_classifier(raw_events)
            sparse_logits = sparse_classifier(pipeline_out["output"])

            raw_loss_total += criterion(raw_logits, labels).item()
            sparse_loss_total += criterion(sparse_logits, labels).item()
            update_metric_accumulator(
                acc,
                raw_events,
                pipeline_out,
                raw_logits,
                sparse_logits,
                labels,
            )

    metrics = finalize_metrics(acc)
    batches = max(acc["batches"], 1)
    metrics["raw_loss"] = raw_loss_total / batches
    metrics["sparse_loss"] = sparse_loss_total / batches
    return metrics


def run_evaluation(config_path, checkpoint_path):
    from predictive_vision.training import build_models

    eval_config = load_config(config_path)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
    train_config = checkpoint_payload.get("config", {})
    config = deep_update(train_config, eval_config)

    seed = int(config.get("training", {}).get("seed", checkpoint_payload.get("seed", 2026)))
    set_seed(seed)
    device = select_device(
        config.get("evaluation", {}).get(
            "device",
            config.get("training", {}).get("device", "auto"),
        )
    )

    from pathlib import Path

    run_dir = Path(checkpoint_payload.get("run_dir", Path(checkpoint_path).parents[1]))
    split_path = Path(config.get("dataset", {}).get("split_path") or (run_dir / "splits" / "split_indices.json"))
    output_dir = Path(config.get("evaluation", {}).get("output_dir") or (run_dir / "eval"))
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_data_loaders(config, split_path=split_path)
    pipeline, baseline_classifier, sparse_classifier = build_models(
        checkpoint_payload.get("config", config),
        loaders["num_classes"],
        device,
    )
    load_checkpoint(
        checkpoint_path,
        pipeline,
        baseline_classifier,
        sparse_classifier,
        map_location=device,
    )

    split = config.get("evaluation", {}).get("split", "test")
    if split not in loaders:
        raise ValueError(f"Unknown evaluation split '{split}'. Choose train, val, or test.")

    metrics = evaluate_models(
        pipeline,
        baseline_classifier,
        sparse_classifier,
        loaders[split],
        device,
        checkpoint_payload.get("config", config),
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "split": split,
        "seed": seed,
        "architecture_version": checkpoint_payload.get("architecture_version"),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "checkpoint_global_step": checkpoint_payload.get("global_step"),
        "metrics": metrics,
    }
    save_json(summary, output_dir / f"{split}_summary.json")
    save_yaml(config, output_dir / "resolved_eval_config.yaml")

    print(
        f"Evaluation complete on {split}: "
        f"raw_acc={metrics['raw_accuracy']:.2f}% "
        f"sparse_acc={metrics['sparse_accuracy']:.2f}% "
        f"reduction={metrics['data_reduction']:.2f}%"
    )
    print(f"Saved summary to {output_dir / f'{split}_summary.json'}")
    return summary
