import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from predictive_vision.checkpoints import checkpoint_payload, load_checkpoint, save_checkpoint
from predictive_vision.config import config_hash, deep_update, load_config, save_json, save_yaml
from predictive_vision.data import build_data_loaders, prepare_events, select_device, set_seed
from predictive_vision.evaluation import evaluate_models
from predictive_vision.metrics import empty_metric_accumulator, finalize_metrics, update_metric_accumulator
from predictive_vision.model import FullPredictiveVisionPipeline


class EventClassifier(nn.Module):
    def __init__(self, input_channels=2, num_classes=101):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.reshape(x.size(0), -1))


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
        handle.write("\n")


def build_models(config, num_classes, device):
    model_cfg = dict(config.get("model", {}))
    for section in ("memory", "plasticity", "regime_objective", "sparse_routing"):
        if section in config and section not in model_cfg:
            model_cfg[section] = config[section]
    input_channels = int(model_cfg.get("input_channels", 2))
    pipeline = FullPredictiveVisionPipeline(**model_cfg).to(device)
    baseline_classifier = EventClassifier(input_channels=input_channels, num_classes=num_classes).to(device)
    sparse_classifier = EventClassifier(input_channels=input_channels, num_classes=num_classes).to(device)
    return pipeline, baseline_classifier, sparse_classifier


def build_optimizers(config, pipeline, baseline_classifier, sparse_classifier):
    lr_cfg = config.get("training", {}).get("learning_rates", {})
    return {
        "pipeline": torch.optim.Adam(pipeline.parameters(), lr=float(lr_cfg.get("pipeline", 1e-3))),
        "baseline": torch.optim.Adam(baseline_classifier.parameters(), lr=float(lr_cfg.get("baseline", 1e-3))),
        "sparse": torch.optim.Adam(sparse_classifier.parameters(), lr=float(lr_cfg.get("sparse", 1e-3))),
    }


def build_schedulers(config, optimizers):
    scheduler_cfg = config.get("training", {}).get("scheduler", {"type": "none"})
    scheduler_type = scheduler_cfg.get("type", "none")
    if scheduler_type == "none":
        return {}
    if scheduler_type != "step_lr":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    step_size = int(scheduler_cfg.get("step_size", 3))
    gamma = float(scheduler_cfg.get("gamma", 0.5))
    return {
        name: torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        for name, optimizer in optimizers.items()
    }


def compute_pipeline_losses(pipeline_out, sparse_logits, labels, config):
    loss_cfg = config.get("training", {}).get("losses", {})
    regime_cfg = config.get("regime_objective", {})
    reconstruction_weight = float(loss_cfg.get("reconstruction_weight", 1.0))
    sparse_weight = float(loss_cfg.get("sparse_classification_weight", 1.0))
    entropy_weight = float(loss_cfg.get("regime_entropy_weight", 0.0))
    regime_weight = float(regime_cfg.get("weight", 0.0)) if regime_cfg.get("enabled", True) else 0.0
    temporal_weight = float(regime_cfg.get("temporal_consistency_weight", 0.0))

    reconstruction = F.mse_loss(
        pipeline_out["expected"],
        pipeline_out["cleaned"],
        reduction="none",
    )
    reconstruction = (reconstruction * pipeline_out["beta"]).mean()
    sparse_classification = F.cross_entropy(sparse_logits, labels)

    regime_probs = torch.clamp(pipeline_out["regime_probs"], min=1e-8)
    regime_entropy = -(regime_probs * regime_probs.log()).sum(dim=-1).mean()
    regime_logits = pipeline_out["regime_logits"]
    regime_labels = pipeline_out["regime_pseudo_labels"]
    regime_pseudo = F.cross_entropy(
        regime_logits.reshape(-1, regime_logits.shape[-1]),
        regime_labels.reshape(-1),
    )
    if regime_probs.shape[1] > 1:
        temporal_consistency = (regime_probs[:, 1:] - regime_probs[:, :-1]).square().mean()
    else:
        temporal_consistency = regime_probs.new_tensor(0.0)

    total = (
        (reconstruction_weight * reconstruction)
        + (sparse_weight * sparse_classification)
        + (entropy_weight * regime_entropy)
        + (regime_weight * regime_pseudo)
        + (temporal_weight * temporal_consistency)
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "sparse_classification": sparse_classification,
        "regime_entropy": regime_entropy,
        "regime_pseudo": regime_pseudo,
        "temporal_consistency": temporal_consistency,
    }


def best_score(metrics, config):
    experiment_cfg = config.get("experiment", {})
    min_reduction = float(experiment_cfg.get("min_data_reduction", 50.0))
    sparse_accuracy = float(metrics.get("sparse_accuracy", 0.0))
    data_reduction = float(metrics.get("data_reduction", 0.0))
    if data_reduction >= min_reduction:
        return sparse_accuracy
    return sparse_accuracy - (min_reduction - data_reduction)


def make_run_dir(config, resume_payload=None):
    if resume_payload is not None and resume_payload.get("run_dir"):
        return Path(resume_payload["run_dir"])

    experiment_cfg = config.get("experiment", {})
    output_dir = Path(experiment_cfg.get("output_dir", "experiments"))
    run_id = experiment_cfg.get("run_id")
    if not run_id:
        name = experiment_cfg.get("name", "paper_run")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_id = f"{timestamp}_{name}_{config_hash(config)}"
    return output_dir / run_id


def train_one_epoch(
    pipeline,
    baseline_classifier,
    sparse_classifier,
    train_loader,
    optimizers,
    device,
    config,
    global_step,
):
    model_cfg = config.get("model", {})
    input_channels = int(model_cfg.get("input_channels", 2))
    criterion = nn.CrossEntropyLoss()
    grad_clip = config.get("training", {}).get("grad_clip_norm")

    pipeline.train()
    baseline_classifier.train()
    sparse_classifier.train()

    acc = empty_metric_accumulator()
    loss_sums = {
        "raw_classification": 0.0,
        "total": 0.0,
        "reconstruction": 0.0,
        "sparse_classification": 0.0,
        "regime_entropy": 0.0,
        "regime_pseudo": 0.0,
        "temporal_consistency": 0.0,
    }

    for frames, labels in train_loader:
        raw_events = prepare_events(frames, input_channels=input_channels).to(device)
        labels = labels.to(device).long()

        optimizers["baseline"].zero_grad()
        raw_logits = baseline_classifier(raw_events)
        raw_loss = criterion(raw_logits, labels)
        raw_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(baseline_classifier.parameters(), float(grad_clip))
        optimizers["baseline"].step()

        optimizers["pipeline"].zero_grad()
        optimizers["sparse"].zero_grad()
        pipeline_out = pipeline(raw_events)
        sparse_logits = sparse_classifier(pipeline_out["output"])
        losses = compute_pipeline_losses(pipeline_out, sparse_logits, labels, config)
        losses["total"].backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(pipeline.parameters(), float(grad_clip))
            torch.nn.utils.clip_grad_norm_(sparse_classifier.parameters(), float(grad_clip))
        optimizers["pipeline"].step()
        optimizers["sparse"].step()
        pipeline.update_memory_from_output(pipeline_out)

        loss_sums["raw_classification"] += raw_loss.item()
        for key in ("total", "reconstruction", "sparse_classification", "regime_entropy", "regime_pseudo", "temporal_consistency"):
            loss_sums[key] += losses[key].item()

        with torch.no_grad():
            update_metric_accumulator(acc, raw_events, pipeline_out, raw_logits, sparse_logits, labels)
        global_step += 1

    batches = max(acc["batches"], 1)
    metrics = finalize_metrics(acc)
    for key, value in loss_sums.items():
        metrics[f"{key}_loss"] = value / batches
    return metrics, global_step


def run_training(config_path, resume=None):
    config = load_config(config_path)
    resume_payload = None
    if resume:
        resume_payload = torch.load(resume, map_location="cpu")
        config = deep_update(resume_payload.get("config", {}), config)

    seed = int(config.get("training", {}).get("seed", 2026))
    set_seed(seed)
    device = select_device(config.get("training", {}).get("device", "auto"))
    run_dir = make_run_dir(config, resume_payload=resume_payload)
    checkpoint_dir = run_dir / "checkpoints"
    split_path = run_dir / "splits" / "split_indices.json"
    metrics_path = run_dir / "metrics.jsonl"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "resolved_config.yaml")
    save_json(config, run_dir / "resolved_config.json")

    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")
    loaders = build_data_loaders(config, split_path=split_path)
    pipeline, baseline_classifier, sparse_classifier = build_models(config, loaders["num_classes"], device)
    optimizers = build_optimizers(config, pipeline, baseline_classifier, sparse_classifier)
    schedulers = build_schedulers(config, optimizers)

    start_epoch = 0
    global_step = 0
    current_best_score = float("-inf")
    best_metrics = None
    if resume:
        loaded = load_checkpoint(
            resume,
            pipeline,
            baseline_classifier,
            sparse_classifier,
            optimizers=optimizers,
            schedulers=schedulers,
            map_location=device,
        )
        start_epoch = int(loaded["epoch"]) + 1
        global_step = int(loaded.get("global_step", 0))
        current_best_score = float(loaded.get("best_score", float("-inf")))
        best_metrics = loaded.get("metrics")
        print(f"Resumed from epoch {start_epoch} at global step {global_step}.")

    training_cfg = config.get("training", {})
    experiment_cfg = config.get("experiment", {})
    epochs = int(training_cfg.get("epochs", 5))
    validate_every = int(experiment_cfg.get("validate_every_epochs", 1))
    save_every = int(experiment_cfg.get("save_every_epochs", 1))
    cfg_hash = config_hash(config)
    last_validation = None

    for epoch in range(start_epoch, epochs):
        train_metrics, global_step = train_one_epoch(
            pipeline,
            baseline_classifier,
            sparse_classifier,
            loaders["train"],
            optimizers,
            device,
            config,
            global_step,
        )

        for scheduler in schedulers.values():
            scheduler.step()

        append_jsonl(metrics_path, {"epoch": epoch, "global_step": global_step, "phase": "train", **train_metrics})
        print(
            f"Epoch {epoch + 1}/{epochs} train: "
            f"raw_acc={train_metrics['raw_accuracy']:.2f}% "
            f"sparse_acc={train_metrics['sparse_accuracy']:.2f}% "
            f"reduction={train_metrics['data_reduction']:.2f}%"
        )

        should_validate = ((epoch + 1) % validate_every == 0) or (epoch + 1 == epochs)
        if should_validate:
            val_metrics = evaluate_models(pipeline, baseline_classifier, sparse_classifier, loaders["val"], device, config)
            last_validation = val_metrics
            append_jsonl(metrics_path, {"epoch": epoch, "global_step": global_step, "phase": "val", **val_metrics})
            score = best_score(val_metrics, config)
            print(
                f"Epoch {epoch + 1}/{epochs} val: "
                f"raw_acc={val_metrics['raw_accuracy']:.2f}% "
                f"sparse_acc={val_metrics['sparse_accuracy']:.2f}% "
                f"reduction={val_metrics['data_reduction']:.2f}% "
                f"score={score:.2f}"
            )
            if score > current_best_score:
                current_best_score = score
                best_metrics = val_metrics
                payload = checkpoint_payload(
                    pipeline,
                    baseline_classifier,
                    sparse_classifier,
                    optimizers,
                    schedulers,
                    config,
                    epoch,
                    global_step,
                    seed,
                    cfg_hash,
                    val_metrics,
                    run_dir,
                    current_best_score,
                )
                save_checkpoint(payload, checkpoint_dir / "best.pt")

        payload = checkpoint_payload(
            pipeline,
            baseline_classifier,
            sparse_classifier,
            optimizers,
            schedulers,
            config,
            epoch,
            global_step,
            seed,
            cfg_hash,
            last_validation or train_metrics,
            run_dir,
            current_best_score,
        )
        save_checkpoint(payload, checkpoint_dir / "last.pt")
        if save_every > 0 and ((epoch + 1) % save_every == 0):
            save_checkpoint(payload, checkpoint_dir / f"epoch_{epoch + 1:03d}.pt")

        summary = {
            "run_dir": str(run_dir),
            "architecture_version": payload["architecture_version"],
            "config_hash": cfg_hash,
            "seed": seed,
            "epoch": epoch,
            "global_step": global_step,
            "best_score": current_best_score,
            "best_metrics": best_metrics,
            "last_validation": last_validation,
            "last_train": train_metrics,
        }
        save_json(summary, run_dir / "summary.json")

    print(f"Training complete. Best checkpoint: {checkpoint_dir / 'best.pt'}")
    return run_dir
