import torch

from predictive_vision.model import (
    FullPredictiveVisionPipeline,
    LearnedPredictor,
    NoiseGate,
    RegimePseudoLabeler,
    SparseOutput,
)


def test_polarity_channels_are_preserved():
    model = FullPredictiveVisionPipeline(hidden_channels=4, noise_threshold=1.0)
    x = torch.zeros(1, 2, 4, 6, 6)
    x[0, 0, 1, 2, 2] = 1.0
    x[0, 1, 1, 2, 2] = 1.0
    out = model(x)
    assert out["cleaned"].shape[1] == 2
    assert out["cleaned"][0, :, 1, 2, 2].sum().item() == 2.0


def test_negative_event_cluster_survives_support_gate():
    gate = NoiseGate(threshold=2.0)
    x = torch.zeros(1, 2, 3, 5, 5)
    x[0, 1, 1, 2, 2] = -1.0
    x[0, 1, 1, 2, 3] = -1.0
    x[0, 1, 1, 3, 2] = -1.0
    cleaned, _ = gate(x)
    assert int((cleaned != 0).sum().item()) == 3


def test_learned_predictor_is_causal():
    model = LearnedPredictor(input_channels=2, hidden_channels=4)
    x1 = torch.zeros(1, 2, 8, 6, 6)
    x2 = x1.clone()
    x2[:, :, 6] = 1.0
    y1 = model(x1, torch.zeros(1, 4, 8, 6, 6))
    y2 = model(x2, torch.zeros(1, 4, 8, 6, 6))
    assert torch.allclose(y1[:, :, :6], y2[:, :, :6])


def test_zero_residual_has_low_beta():
    model = FullPredictiveVisionPipeline(hidden_channels=4, noise_threshold=1.0)
    x = torch.zeros(1, 2, 5, 8, 8)
    out = model(x)
    assert out["beta"].mean().item() < 0.05


def test_coherent_persistent_residual_has_higher_beta_than_noise():
    model = FullPredictiveVisionPipeline(hidden_channels=4, noise_threshold=1.0, error_threshold=0.05)
    coherent = torch.zeros(1, 2, 5, 8, 8)
    coherent[:, :, 1:5, 3:6, 3:6] = 1.0
    noisy = torch.zeros_like(coherent)
    noisy[0, 0, 1, 1, 1] = 1.0
    noisy[0, 1, 3, 6, 6] = 1.0
    coherent_out = model(coherent)
    noisy_out = model(noisy)
    assert coherent_out["beta"].mean().item() > noisy_out["beta"].mean().item()
    assert coherent_out["plasticity_factors"]["gamma_comp"].mean().item() > noisy_out["plasticity_factors"]["gamma_comp"].mean().item()


def test_memory_updates_train_only_and_adaptive_is_faster():
    model = FullPredictiveVisionPipeline(hidden_channels=4, noise_threshold=1.0)
    x = torch.zeros(1, 2, 5, 8, 8)
    x[:, :, 1:5, 3:6, 3:6] = 1.0

    model.train()
    out = model(x)
    before_stable = model.memory.stable_memory.clone()
    before_adaptive = model.memory.adaptive_memory.clone()
    model.update_memory_from_output(out)
    stable_delta = (model.memory.stable_memory - before_stable).abs().sum()
    adaptive_delta = (model.memory.adaptive_memory - before_adaptive).abs().sum()
    assert adaptive_delta > stable_delta

    model.eval()
    out = model(x)
    stable_eval = model.memory.stable_memory.clone()
    model.update_memory_from_output(out)
    assert torch.allclose(model.memory.stable_memory, stable_eval)


def test_regime_pseudo_labels_for_synthetic_cases():
    labeler = RegimePseudoLabeler()
    residual_density = torch.tensor([[0.0, 0.04, 0.20, 0.05]])
    persistence = torch.tensor([[0.0, 0.60, 0.30, 0.20]])
    compressibility = torch.tensor([[0.0, 0.80, 0.80, 0.10]])
    transition_score = torch.tensor([[0.0, 0.01, 0.20, 0.01]])
    labels = labeler(residual_density, persistence, compressibility, transition_score)
    assert labels.tolist()[0] == [0, 1, 2, 3]


def test_anomaly_routing_preserves_more_than_stable_routing():
    router = SparseOutput(regime_thresholds=[0.45, 0.35, 0.20, 0.60], hard_gate=True)
    residual = torch.ones(1, 2, 2, 4, 4)
    mask = torch.ones_like(residual)
    compressibility = torch.ones_like(residual)

    stable_probs = torch.tensor([[[0.90, 0.05, 0.03, 0.02], [0.90, 0.05, 0.03, 0.02]]])
    anomaly_probs = torch.tensor([[[0.05, 0.05, 0.85, 0.05], [0.05, 0.05, 0.85, 0.05]]])
    stable_ids = stable_probs.argmax(dim=-1)
    anomaly_ids = anomaly_probs.argmax(dim=-1)

    stable_out, stable_mask, _ = router(residual, mask, stable_probs, stable_ids, compressibility)
    anomaly_out, anomaly_mask, _ = router(residual, mask, anomaly_probs, anomaly_ids, compressibility)
    assert anomaly_mask.mean().item() >= stable_mask.mean().item()
    assert anomaly_out.abs().mean().item() >= stable_out.abs().mean().item()
