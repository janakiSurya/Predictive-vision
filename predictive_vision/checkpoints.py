import subprocess
from pathlib import Path

import torch

from predictive_vision.model import ARCHITECTURE_VERSION


def get_git_commit():
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return "unknown"


def checkpoint_payload(
    pipeline,
    baseline_classifier,
    sparse_classifier,
    optimizers,
    schedulers,
    config,
    epoch,
    global_step,
    seed,
    config_hash,
    metrics,
    run_dir,
    best_score,
):
    return {
        "architecture_version": ARCHITECTURE_VERSION,
        "epoch": epoch,
        "global_step": global_step,
        "seed": seed,
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "metrics": metrics,
        "best_score": best_score,
        "run_dir": str(run_dir),
        "config": config,
        "pipeline_state": pipeline.state_dict(),
        "baseline_classifier_state": baseline_classifier.state_dict(),
        "sparse_classifier_state": sparse_classifier.state_dict(),
        "optimizer_states": {name: opt.state_dict() for name, opt in optimizers.items()},
        "scheduler_states": {name: sch.state_dict() for name, sch in schedulers.items()},
    }


def save_checkpoint(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    checkpoint,
    pipeline,
    baseline_classifier,
    sparse_classifier,
    optimizers=None,
    schedulers=None,
    map_location="cpu",
):
    payload = torch.load(checkpoint, map_location=map_location)
    pipeline.load_state_dict(payload["pipeline_state"])
    baseline_classifier.load_state_dict(payload["baseline_classifier_state"])
    sparse_classifier.load_state_dict(payload["sparse_classifier_state"])

    if optimizers is not None:
        for name, optimizer in optimizers.items():
            if name in payload.get("optimizer_states", {}):
                optimizer.load_state_dict(payload["optimizer_states"][name])
    if schedulers is not None:
        for name, scheduler in schedulers.items():
            if name in payload.get("scheduler_states", {}):
                scheduler.load_state_dict(payload["scheduler_states"][name])
    return payload
