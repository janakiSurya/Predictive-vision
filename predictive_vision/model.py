import math

import torch
import torch.nn as nn
import torch.nn.functional as F


ARCHITECTURE_VERSION = "predictive_vision_v3"


def _causal_temporal_average(x, window):
    padding = window - 1
    padded = F.pad(x, (0, 0, 0, 0, padding, 0))
    return F.avg_pool3d(padded, kernel_size=(window, 1, 1), stride=1)


def _local_average(x, kernel_size=3):
    padding = kernel_size // 2
    return F.avg_pool3d(x, kernel_size=(1, kernel_size, kernel_size), stride=1, padding=(0, padding, padding))


class NoiseGate(nn.Module):
    """Support-based spatiotemporal gate that preserves event polarity channels."""

    def __init__(self, threshold=3.0, kernel_size=3):
        super().__init__()
        self.threshold = threshold
        self.kernel_size = kernel_size
        kernel = torch.ones(1, 1, kernel_size, kernel_size, kernel_size)
        self.register_buffer("support_kernel", kernel)

    def forward(self, x):
        channels = x.shape[1]
        padding = self.kernel_size // 2
        activity = (x.abs() > 0).float()
        kernel = self.support_kernel.repeat(channels, 1, 1, 1, 1)
        local_support = F.conv3d(activity, kernel, padding=padding, groups=channels)
        mask = (local_support >= self.threshold).float()
        return x * mask, mask


class PhysicsPredictor(nn.Module):
    """Causal temporal persistence with a small local propagation prior."""

    def __init__(self, input_channels=2, spatial_blur=0.15):
        super().__init__()
        self.input_channels = input_channels
        self.spatial_blur = spatial_blur
        kernel = torch.ones(1, 1, 1, 3, 3) / 9.0
        self.register_buffer("spatial_kernel", kernel)

    def forward(self, x):
        previous = torch.zeros_like(x)
        previous[:, :, 1:, :, :] = x[:, :, :-1, :, :]
        kernel = self.spatial_kernel.repeat(self.input_channels, 1, 1, 1, 1)
        propagated = F.conv3d(previous, kernel, padding=(0, 1, 1), groups=self.input_channels)
        return ((1.0 - self.spatial_blur) * previous) + (self.spatial_blur * propagated)


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.temporal_padding = kernel_size[0] - 1
        self.spatial_padding = kernel_size[1] // 2
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

    def forward(self, x):
        x = F.pad(
            x,
            (
                self.spatial_padding,
                self.spatial_padding,
                self.spatial_padding,
                self.spatial_padding,
                self.temporal_padding,
                0,
            ),
        )
        return self.conv(x)


class LearnedPredictor(nn.Module):
    def __init__(self, input_channels=2, context_channels=4, hidden_channels=16):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv3d(input_channels + context_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
            CausalConv3d(hidden_channels, input_channels, kernel_size=3),
            nn.ReLU(),
        )

    def forward(self, x, memory_context):
        return self.net(torch.cat([x, memory_context], dim=1))


class HybridPredictor(nn.Module):
    def __init__(self, input_channels=2, hidden_channels=16, initial_alpha=0.9):
        super().__init__()
        self.physics_model = PhysicsPredictor(input_channels=input_channels)
        self.learned_model = LearnedPredictor(
            input_channels=input_channels,
            context_channels=input_channels * 2,
            hidden_channels=hidden_channels,
        )
        initial_alpha = min(max(initial_alpha, 1e-4), 1.0 - 1e-4)
        self.alpha_logit = nn.Parameter(torch.tensor([math.log(initial_alpha / (1.0 - initial_alpha))]))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(self, x, memory_context):
        alpha = self.alpha.view(1, 1, 1, 1, 1)
        pred_phys = self.physics_model(x)
        pred_learn = self.learned_model(x, memory_context)
        return (alpha * pred_phys) + ((1.0 - alpha) * pred_learn)


class ComplementaryMemoryStores(nn.Module):
    """Lightweight stable/adaptive/episodic stores inspired by complementary learning systems."""

    def __init__(
        self,
        input_channels=2,
        enabled=True,
        stable_rate=0.01,
        adaptive_rate=0.10,
        episodic_size=64,
        high_surprise_threshold=0.05,
        low_compressibility_threshold=0.35,
    ):
        super().__init__()
        self.enabled = enabled
        self.stable_rate = stable_rate
        self.adaptive_rate = adaptive_rate
        self.episodic_size = episodic_size
        self.high_surprise_threshold = high_surprise_threshold
        self.low_compressibility_threshold = low_compressibility_threshold
        self.feature_dim = (input_channels * 2) + 4
        self.register_buffer("stable_memory", torch.zeros(1, input_channels, 1, 1, 1))
        self.register_buffer("adaptive_memory", torch.zeros(1, input_channels, 1, 1, 1))
        self.register_buffer("episodic_buffer", torch.zeros(episodic_size, self.feature_dim))
        self.register_buffer("episodic_ptr", torch.zeros((), dtype=torch.long))
        self.register_buffer("episodic_count", torch.zeros((), dtype=torch.long))
        self.last_episodic_write = 0.0

    def context_like(self, x):
        stable = self.stable_memory.expand(x.shape[0], -1, x.shape[2], x.shape[3], x.shape[4])
        adaptive = self.adaptive_memory.expand_as(stable)
        if not self.enabled:
            return torch.zeros_like(stable), torch.zeros_like(adaptive)
        return stable, adaptive

    def mismatch(self, x):
        stable, adaptive = self.context_like(x)
        return ((x - stable).abs() + (x - adaptive).abs()) * 0.5

    @torch.no_grad()
    def update(self, cleaned, residual, beta, plasticity_factors):
        self.last_episodic_write = 0.0
        if not self.enabled:
            return

        summary = cleaned.detach().mean(dim=(0, 2, 3, 4), keepdim=False).view(1, -1, 1, 1, 1)
        update_strength = beta.detach().mean().clamp(0.0, 1.0)
        stable_rate = self.stable_rate * update_strength
        adaptive_rate = self.adaptive_rate * update_strength
        self.stable_memory.mul_(1.0 - stable_rate).add_(summary * stable_rate)
        self.adaptive_memory.mul_(1.0 - adaptive_rate).add_(summary * adaptive_rate)

        surprise = residual.detach().abs().mean()
        compressibility = plasticity_factors["gamma_comp"].detach().mean()
        should_buffer = (
            surprise > self.high_surprise_threshold
            and compressibility < self.low_compressibility_threshold
            and self.episodic_size > 0
        )
        if should_buffer:
            cleaned_summary = cleaned.detach().abs().mean(dim=(0, 2, 3, 4))
            residual_summary = residual.detach().abs().mean(dim=(0, 2, 3, 4))
            factor_summary = torch.stack(
                [
                    plasticity_factors["gamma_conf"].detach().mean(),
                    plasticity_factors["gamma_pers"].detach().mean(),
                    plasticity_factors["gamma_ctx"].detach().mean(),
                    plasticity_factors["gamma_comp"].detach().mean(),
                ]
            )
            vector = torch.cat([cleaned_summary, residual_summary, factor_summary])
            ptr = int(self.episodic_ptr.item())
            self.episodic_buffer[ptr].copy_(vector)
            self.episodic_ptr.copy_((self.episodic_ptr + 1) % self.episodic_size)
            self.episodic_count.copy_(torch.clamp(self.episodic_count + 1, max=self.episodic_size))
            self.last_episodic_write = 1.0

    def stats(self):
        return {
            "stable_memory_norm": self.stable_memory.norm().detach(),
            "adaptive_memory_norm": self.adaptive_memory.norm().detach(),
            "episodic_count": self.episodic_count.detach().float(),
            "episodic_rate": torch.tensor(
                self.last_episodic_write,
                device=self.stable_memory.device,
                dtype=self.stable_memory.dtype,
            ),
        }


class ErrorExtractor(nn.Module):
    def __init__(self, error_threshold=0.2):
        super().__init__()
        self.error_threshold = error_threshold

    def forward(self, observed, predicted):
        residual = observed - predicted
        residual_mask = (residual.abs() >= self.error_threshold).float()
        thresholded_residual = residual * residual_mask
        return residual, thresholded_residual, residual_mask


class PlasticityController(nn.Module):
    def __init__(
        self,
        temporal_window=3,
        conf_weight=1.0,
        pers_weight=1.0,
        ctx_weight=1.0,
        comp_weight=1.0,
        bias=2.5,
        scale=2.0,
        use_compressibility=True,
    ):
        super().__init__()
        self.temporal_window = temporal_window
        self.conf_weight = conf_weight
        self.pers_weight = pers_weight
        self.ctx_weight = ctx_weight
        self.comp_weight = comp_weight
        self.bias = bias
        self.scale = scale
        self.use_compressibility = use_compressibility

    def forward(self, residual, noise_mask, regime_confidence):
        magnitude = residual.abs()
        activity = (magnitude > 0).float()

        gamma_conf = noise_mask.detach().clamp(0.0, 1.0)
        gamma_pers = _causal_temporal_average(activity, self.temporal_window).clamp(0.0, 1.0)
        gamma_ctx = regime_confidence.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(residual)

        local_mean = _local_average(magnitude)
        local_second = _local_average(magnitude.square())
        local_var = (local_second - local_mean.square()).clamp_min(0.0)
        gamma_comp = local_mean / (local_mean + local_var.sqrt() + 1e-6)
        gamma_comp = gamma_comp * activity
        if not self.use_compressibility:
            gamma_comp = activity

        drive = (
            (self.conf_weight * gamma_conf)
            + (self.pers_weight * gamma_pers)
            + (self.ctx_weight * gamma_ctx)
            + (self.comp_weight * gamma_comp)
        )
        beta = torch.sigmoid((drive - self.bias) * self.scale) * activity
        factors = {
            "gamma_conf": gamma_conf,
            "gamma_pers": gamma_pers,
            "gamma_ctx": gamma_ctx,
            "gamma_comp": gamma_comp,
        }
        return beta, factors


class RegimePseudoLabeler(nn.Module):
    STABLE = 0
    COHERENT_MOTION = 1
    ANOMALY = 2
    NOISE = 3

    def __init__(
        self,
        stable_threshold=0.02,
        coherent_threshold=0.45,
        anomaly_threshold=0.12,
        noise_comp_threshold=0.25,
    ):
        super().__init__()
        self.stable_threshold = stable_threshold
        self.coherent_threshold = coherent_threshold
        self.anomaly_threshold = anomaly_threshold
        self.noise_comp_threshold = noise_comp_threshold

    def forward(self, residual_density, persistence, compressibility, transition_score):
        labels = torch.full_like(residual_density, self.STABLE, dtype=torch.long)
        coherent = (
            (residual_density >= self.stable_threshold)
            & (persistence >= self.coherent_threshold)
            & (compressibility >= self.noise_comp_threshold)
        )
        noisy = (
            (residual_density >= self.stable_threshold)
            & (compressibility < self.noise_comp_threshold)
        )
        anomaly = (
            (residual_density >= self.anomaly_threshold)
            | (transition_score >= self.anomaly_threshold)
        ) & (compressibility >= self.noise_comp_threshold)
        labels[coherent] = self.COHERENT_MOTION
        labels[noisy] = self.NOISE
        labels[anomaly] = self.ANOMALY
        return labels


class RegimeClassifier(nn.Module):
    def __init__(self, input_channels=2, hidden_size=16, num_regimes=4, pseudo_label_cfg=None):
        super().__init__()
        self.num_regimes = num_regimes
        self.feature_dim = 10
        self.rnn = nn.GRU(input_size=self.feature_dim, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_regimes)
        self.pseudo_labeler = RegimePseudoLabeler(**(pseudo_label_cfg or {}))

    def _time_features(self, cleaned, residual, factors, memory_mismatch):
        batch, channels, time, _, _ = residual.shape
        residual_mag = residual.abs()
        cleaned_mag = cleaned.abs()
        mismatch_mag = memory_mismatch.abs()

        residual_density = (residual_mag > 0).float().mean(dim=(1, 3, 4))
        cleaned_density = (cleaned_mag > 0).float().mean(dim=(1, 3, 4))
        residual_energy = residual_mag.mean(dim=(1, 3, 4))
        memory_mismatch_energy = mismatch_mag.mean(dim=(1, 3, 4))
        persistence = factors["gamma_pers"].mean(dim=(1, 3, 4))
        compressibility = factors["gamma_comp"].mean(dim=(1, 3, 4))
        confidence = factors["gamma_conf"].mean(dim=(1, 3, 4))
        context = factors["gamma_ctx"].mean(dim=(1, 3, 4))

        temporal_change = torch.zeros(batch, time, device=residual.device, dtype=residual.dtype)
        temporal_change[:, 1:] = (
            residual_mag[:, :, 1:] - residual_mag[:, :, :-1]
        ).abs().mean(dim=(1, 3, 4))
        transition_score = (temporal_change + memory_mismatch_energy) * 0.5

        features = torch.stack(
            [
                residual_density,
                cleaned_density,
                residual_energy,
                memory_mismatch_energy,
                persistence,
                compressibility,
                confidence,
                context,
                temporal_change,
                transition_score,
            ],
            dim=-1,
        )
        pseudo_labels = self.pseudo_labeler(
            residual_density,
            persistence,
            compressibility,
            transition_score,
        )
        return features, pseudo_labels, transition_score

    def forward(self, cleaned, residual, factors, memory_mismatch):
        features, pseudo_labels, transition_score = self._time_features(
            cleaned,
            residual,
            factors,
            memory_mismatch,
        )
        rnn_out, _ = self.rnn(features)
        logits = self.classifier(rnn_out)
        probs = F.softmax(logits, dim=-1)
        confidence, regime_id = torch.max(probs, dim=-1)
        context_gate = (confidence * (1.0 - probs[..., RegimePseudoLabeler.NOISE])).clamp(0.0, 1.0)
        return {
            "logits": logits,
            "probs": probs,
            "confidence": confidence,
            "regime_id": regime_id,
            "pseudo_labels": pseudo_labels,
            "transition_score": transition_score,
            "context_gate": context_gate,
            "features": features,
        }


class SparseOutput(nn.Module):
    def __init__(
        self,
        regime_thresholds=None,
        gate_temperature=12.0,
        hard_gate=False,
        use_contextual_routing=True,
    ):
        super().__init__()
        thresholds = regime_thresholds or [0.45, 0.35, 0.20, 0.60]
        self.register_buffer("regime_thresholds", torch.tensor(thresholds, dtype=torch.float32))
        self.gate_temperature = gate_temperature
        self.hard_gate = hard_gate
        self.use_contextual_routing = use_contextual_routing

    def forward(self, thresholded_error, residual_mask, regime_probs, regime_id, compressibility):
        batch, _, time, _, _ = thresholded_error.shape
        thresholds = self.regime_thresholds.to(thresholded_error.device)
        selected_threshold = thresholds[regime_id].view(batch, time)
        anomaly_prob = regime_probs[..., RegimePseudoLabeler.ANOMALY]
        noise_prob = regime_probs[..., RegimePseudoLabeler.NOISE]
        comp = compressibility.mean(dim=(1, 3, 4)).clamp(0.0, 1.0)

        if self.use_contextual_routing:
            routing_score = (0.60 * anomaly_prob) + (0.30 * comp) + (0.10 * (1.0 - noise_prob))
        else:
            routing_score = regime_probs.max(dim=-1).values

        hard_gate = (routing_score >= selected_threshold).float()
        soft_gate = torch.sigmoid((routing_score - selected_threshold) * self.gate_temperature)
        if self.hard_gate:
            gate = hard_gate
        else:
            gate = hard_gate + soft_gate - soft_gate.detach()

        gate = gate.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        hard_gate = hard_gate.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        output = thresholded_error * gate
        final_mask = residual_mask * hard_gate
        return output, final_mask, soft_gate


class FullPredictiveVisionPipeline(nn.Module):
    def __init__(
        self,
        input_channels=2,
        hidden_channels=16,
        num_regimes=4,
        noise_threshold=2.0,
        error_threshold=0.2,
        initial_alpha=0.8,
        use_noise_gate=True,
        use_predictor=True,
        use_plasticity=True,
        use_regime_gate=True,
        hard_regime_gate=False,
        memory=None,
        plasticity=None,
        regime_objective=None,
        sparse_routing=None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.use_noise_gate = use_noise_gate
        self.use_predictor = use_predictor
        self.use_plasticity = use_plasticity
        self.use_regime_gate = use_regime_gate
        self.update_memory = bool((memory or {}).get("update_during_training", True))

        memory_cfg = dict(memory or {})
        plasticity_cfg = plasticity or {}
        regime_cfg = regime_objective or {}
        routing_cfg = sparse_routing or {}
        memory_cfg.pop("update_during_training", None)

        self.noise_gate = NoiseGate(threshold=noise_threshold)
        self.memory = ComplementaryMemoryStores(input_channels=input_channels, **memory_cfg)
        self.predictor = HybridPredictor(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            initial_alpha=initial_alpha,
        )
        self.error_extractor = ErrorExtractor(error_threshold=error_threshold)
        self.plasticity_ctrl = PlasticityController(**plasticity_cfg)
        self.regime_classifier = RegimeClassifier(
            input_channels=input_channels,
            hidden_size=hidden_channels,
            num_regimes=num_regimes,
            pseudo_label_cfg=regime_cfg.get("pseudo_labeler", {}),
        )
        self.sparse_output = SparseOutput(
            hard_gate=hard_regime_gate,
            use_contextual_routing=routing_cfg.get("enabled", True),
            regime_thresholds=routing_cfg.get("regime_thresholds"),
            gate_temperature=routing_cfg.get("gate_temperature", 12.0),
        )

    def forward(self, raw_events):
        if raw_events.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {raw_events.shape[1]}."
            )

        if self.use_noise_gate:
            cleaned_events, noise_mask = self.noise_gate(raw_events)
        else:
            cleaned_events = raw_events
            noise_mask = torch.ones_like(raw_events)

        stable_ctx, adaptive_ctx = self.memory.context_like(cleaned_events)
        memory_context = torch.cat([stable_ctx, adaptive_ctx], dim=1)
        memory_mismatch = self.memory.mismatch(cleaned_events)

        if self.use_predictor:
            expected_events = self.predictor(cleaned_events, memory_context)
        else:
            expected_events = torch.zeros_like(cleaned_events)

        raw_residual, thresholded_residual, residual_mask = self.error_extractor(
            cleaned_events,
            expected_events,
        )

        bootstrap_confidence = torch.full(
            (raw_events.shape[0], raw_events.shape[2]),
            1.0 / self.regime_classifier.num_regimes,
            device=raw_events.device,
            dtype=raw_events.dtype,
        )
        preliminary_beta, preliminary_factors = self.plasticity_ctrl(
            thresholded_residual,
            noise_mask,
            bootstrap_confidence,
        )
        regime = self.regime_classifier(
            cleaned_events,
            thresholded_residual,
            preliminary_factors,
            memory_mismatch,
        )

        if self.use_plasticity:
            beta_gate, plasticity_factors = self.plasticity_ctrl(
                thresholded_residual,
                noise_mask,
                regime["confidence"],
            )
        else:
            beta_gate = torch.ones_like(thresholded_residual)
            plasticity_factors = preliminary_factors

        if self.use_regime_gate:
            final_sparse_signal, final_mask, confidence_gate = self.sparse_output(
                thresholded_residual,
                residual_mask,
                regime["probs"],
                regime["regime_id"],
                plasticity_factors["gamma_comp"],
            )
        else:
            final_sparse_signal = thresholded_residual
            final_mask = residual_mask
            confidence_gate = torch.ones_like(regime["confidence"])

        memory_stats = self.memory.stats()
        return {
            "output": final_sparse_signal,
            "residual": thresholded_residual,
            "raw_residual": raw_residual,
            "mask": final_mask,
            "residual_mask": residual_mask,
            "noise_mask": noise_mask,
            "beta": beta_gate,
            "plasticity_factors": plasticity_factors,
            "regime_logits": regime["logits"],
            "regime_probs": regime["probs"],
            "regime": regime["probs"],
            "regime_confidence": regime["confidence"],
            "regime_id": regime["regime_id"],
            "regime_pseudo_labels": regime["pseudo_labels"],
            "regime_transition_score": regime["transition_score"],
            "context_gate": regime["context_gate"],
            "confidence_gate": confidence_gate,
            "memory_mismatch": memory_mismatch,
            "memory_stats": memory_stats,
            "cleaned": cleaned_events,
            "expected": expected_events,
            "alpha": self.predictor.alpha.detach(),
        }

    def update_memory_from_output(self, pipeline_out):
        if self.training and self.update_memory:
            self.memory.update(
                pipeline_out["cleaned"],
                pipeline_out["residual"],
                pipeline_out["beta"],
                pipeline_out["plasticity_factors"],
            )
