import torch

from predictive_vision.data import active_density


def _scalar(value):
    if torch.is_tensor(value):
        return value.detach().float().mean().item()
    return float(value)


def empty_metric_accumulator():
    return {
        "samples": 0,
        "raw_correct": 0,
        "sparse_correct": 0,
        "raw_density": 0.0,
        "cleaned_density": 0.0,
        "residual_density": 0.0,
        "sparse_density": 0.0,
        "output_density": 0.0,
        "beta_mean": 0.0,
        "beta_min": 0.0,
        "beta_max": 0.0,
        "regime_confidence": 0.0,
        "regime_entropy": 0.0,
        "transition_rate": 0.0,
        "compressible_surprise_density": 0.0,
        "contextual_routing_density": 0.0,
        "alpha": 0.0,
        "gamma_conf_mean": 0.0,
        "gamma_pers_mean": 0.0,
        "gamma_ctx_mean": 0.0,
        "gamma_comp_mean": 0.0,
        "stable_memory_norm": 0.0,
        "adaptive_memory_norm": 0.0,
        "episodic_rate": 0.0,
        "episodic_count": 0.0,
        "batches": 0,
    }


def update_metric_accumulator(acc, raw_events, pipeline_out, raw_logits, sparse_logits, labels):
    batch_size = labels.size(0)
    factors = pipeline_out["plasticity_factors"]
    memory_stats = pipeline_out["memory_stats"]
    regime_probs = torch.clamp(pipeline_out["regime_probs"], min=1e-8)
    regime_entropy = -(regime_probs * regime_probs.log()).sum(dim=-1).mean()
    transition_rate = (pipeline_out["regime_transition_score"] > 0.12).float().mean()
    compressible_surprise = (
        (pipeline_out["residual_mask"] > 0)
        & (factors["gamma_comp"] > 0.5)
    ).float().mean()

    acc["samples"] += batch_size
    acc["raw_correct"] += (raw_logits.argmax(dim=1) == labels).sum().item()
    acc["sparse_correct"] += (sparse_logits.argmax(dim=1) == labels).sum().item()
    acc["raw_density"] += active_density(raw_events)
    acc["cleaned_density"] += active_density(pipeline_out["cleaned"])
    acc["residual_density"] += active_density(pipeline_out["residual_mask"])
    acc["sparse_density"] += active_density(pipeline_out["mask"])
    acc["output_density"] += active_density(pipeline_out["output"], eps=1e-8)
    acc["beta_mean"] += pipeline_out["beta"].mean().item()
    acc["beta_min"] += pipeline_out["beta"].min().item()
    acc["beta_max"] += pipeline_out["beta"].max().item()
    acc["regime_confidence"] += pipeline_out["regime_confidence"].mean().item()
    acc["regime_entropy"] += regime_entropy.item()
    acc["transition_rate"] += transition_rate.item()
    acc["compressible_surprise_density"] += compressible_surprise.item()
    acc["contextual_routing_density"] += active_density(pipeline_out["mask"])
    acc["alpha"] += pipeline_out["alpha"].mean().item()
    acc["gamma_conf_mean"] += factors["gamma_conf"].mean().item()
    acc["gamma_pers_mean"] += factors["gamma_pers"].mean().item()
    acc["gamma_ctx_mean"] += factors["gamma_ctx"].mean().item()
    acc["gamma_comp_mean"] += factors["gamma_comp"].mean().item()
    acc["stable_memory_norm"] += _scalar(memory_stats["stable_memory_norm"])
    acc["adaptive_memory_norm"] += _scalar(memory_stats["adaptive_memory_norm"])
    acc["episodic_rate"] += _scalar(memory_stats["episodic_rate"])
    acc["episodic_count"] += _scalar(memory_stats["episodic_count"])
    acc["batches"] += 1


def finalize_metrics(acc, prefix=None):
    batches = max(acc["batches"], 1)
    samples = max(acc["samples"], 1)
    raw_density = acc["raw_density"] / batches
    sparse_density = acc["sparse_density"] / batches
    metrics = {
        "samples": acc["samples"],
        "raw_accuracy": (acc["raw_correct"] / samples) * 100.0,
        "sparse_accuracy": (acc["sparse_correct"] / samples) * 100.0,
        "raw_density": raw_density,
        "cleaned_density": acc["cleaned_density"] / batches,
        "residual_density": acc["residual_density"] / batches,
        "sparse_density": sparse_density,
        "output_density": acc["output_density"] / batches,
        "data_reduction": (1.0 - (sparse_density / (raw_density + 1e-9))) * 100.0,
        "beta_mean": acc["beta_mean"] / batches,
        "beta_min": acc["beta_min"] / batches,
        "beta_max": acc["beta_max"] / batches,
        "regime_confidence": acc["regime_confidence"] / batches,
        "regime_entropy": acc["regime_entropy"] / batches,
        "transition_rate": acc["transition_rate"] / batches,
        "compressible_surprise_density": acc["compressible_surprise_density"] / batches,
        "contextual_routing_density": acc["contextual_routing_density"] / batches,
        "alpha": acc["alpha"] / batches,
        "gamma_conf_mean": acc["gamma_conf_mean"] / batches,
        "gamma_pers_mean": acc["gamma_pers_mean"] / batches,
        "gamma_ctx_mean": acc["gamma_ctx_mean"] / batches,
        "gamma_comp_mean": acc["gamma_comp_mean"] / batches,
        "stable_memory_norm": acc["stable_memory_norm"] / batches,
        "adaptive_memory_norm": acc["adaptive_memory_norm"] / batches,
        "episodic_rate": acc["episodic_rate"] / batches,
        "episodic_count": acc["episodic_count"] / batches,
    }
    if prefix is None:
        return metrics
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
