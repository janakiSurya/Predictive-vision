from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from predictive_vision.checkpoints import checkpoint_payload, load_checkpoint, save_checkpoint
from predictive_vision.data import make_split_indices, select_device
from predictive_vision.training import (
    build_models,
    build_optimizers,
    build_schedulers,
    train_one_epoch,
)


def tiny_config():
    return {
        "model": {
            "input_channels": 2,
            "hidden_channels": 4,
            "num_regimes": 4,
            "noise_threshold": 1.0,
            "error_threshold": 0.05,
            "initial_alpha": 0.8,
            "use_noise_gate": True,
            "use_predictor": True,
            "use_plasticity": True,
            "use_regime_gate": True,
            "hard_regime_gate": False,
        },
        "memory": {
            "enabled": True,
            "stable_rate": 0.01,
            "adaptive_rate": 0.10,
            "episodic_size": 8,
            "high_surprise_threshold": 0.01,
            "low_compressibility_threshold": 0.35,
            "update_during_training": True,
        },
        "plasticity": {
            "temporal_window": 3,
            "conf_weight": 1.0,
            "pers_weight": 1.0,
            "ctx_weight": 1.0,
            "comp_weight": 1.0,
            "bias": 2.5,
            "scale": 2.0,
            "use_compressibility": True,
        },
        "regime_objective": {
            "enabled": True,
            "weight": 0.20,
            "temporal_consistency_weight": 0.02,
            "pseudo_labeler": {},
        },
        "sparse_routing": {
            "enabled": True,
            "regime_thresholds": [0.45, 0.35, 0.20, 0.60],
            "gate_temperature": 12.0,
        },
        "training": {
            "learning_rates": {"pipeline": 0.001, "baseline": 0.001, "sparse": 0.001},
            "losses": {
                "reconstruction_weight": 1.0,
                "sparse_classification_weight": 1.0,
                "regime_entropy_weight": 0.01,
            },
            "scheduler": {"type": "none"},
        },
    }


def synthetic_loader():
    frames = torch.zeros(4, 5, 2, 12, 12)
    frames[:, 1:4, 0, 4:8, 4:8] = 1.0
    frames[:, 2:5, 1, 5:9, 5:9] = 1.0
    labels = torch.tensor([0, 1, 2, 1])
    return DataLoader(TensorDataset(frames, labels), batch_size=2)


def test_fixed_split_reproducibility():
    first = make_split_indices(20, [0.7, 0.15, 0.15], seed=2026)
    second = make_split_indices(20, [0.7, 0.15, 0.15], seed=2026)
    assert first == second


def test_checkpoint_save_load_and_memory_state(tmp_path):
    cfg = tiny_config()
    device = select_device("cpu")
    pipeline, baseline, sparse = build_models(cfg, 3, device)
    optimizers = build_optimizers(cfg, pipeline, baseline, sparse)
    schedulers = build_schedulers(cfg, optimizers)
    payload = checkpoint_payload(
        pipeline,
        baseline,
        sparse,
        optimizers,
        schedulers,
        cfg,
        epoch=0,
        global_step=1,
        seed=2026,
        config_hash="abc",
        metrics={"sparse_accuracy": 0.0},
        run_dir=tmp_path,
        best_score=0.0,
    )
    path = Path(tmp_path) / "test.pt"
    save_checkpoint(payload, path)
    loaded_pipeline, loaded_baseline, loaded_sparse = build_models(cfg, 3, device)
    loaded = load_checkpoint(path, loaded_pipeline, loaded_baseline, loaded_sparse, map_location=device)
    assert loaded["architecture_version"] == "predictive_vision_v3"
    assert torch.allclose(pipeline.memory.stable_memory, loaded_pipeline.memory.stable_memory)


def test_first_batch_pseudo_labels_are_reproducible():
    cfg = tiny_config()
    device = select_device("cpu")
    frames, _ = next(iter(synthetic_loader()))
    pipeline_a, _, _ = build_models(cfg, 3, device)
    pipeline_b, _, _ = build_models(cfg, 3, device)
    out_a = pipeline_a(frames.permute(0, 2, 1, 3, 4).float())
    out_b = pipeline_b(frames.permute(0, 2, 1, 3, 4).float())
    assert torch.equal(out_a["regime_pseudo_labels"], out_b["regime_pseudo_labels"])


def test_synthetic_training_creates_metrics():
    cfg = tiny_config()
    device = select_device("cpu")
    pipeline, baseline, sparse = build_models(cfg, 3, device)
    optimizers = build_optimizers(cfg, pipeline, baseline, sparse)
    metrics, global_step = train_one_epoch(
        pipeline,
        baseline,
        sparse,
        synthetic_loader(),
        optimizers,
        device,
        cfg,
        global_step=0,
    )
    assert global_step == 2
    assert "gamma_comp_mean" in metrics
    assert "regime_pseudo_loss" in metrics
    assert metrics["samples"] == 4
