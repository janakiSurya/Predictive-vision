# Predictive Vision

Brain-inspired, event-driven predictive vision research code built with PyTorch and Tonic. The project studies a predictive-coding-style idea on N-Caltech101: preserve structured surprise while suppressing sensor noise, predictable event activity, and unstructured residuals.

The current architecture is `predictive_vision_v3`.

## Documentation

- [What We Built](docs/what_we_built.md) explains the project journey, architecture, module layout, and what each major piece does.
- [Running The Project](docs/running_the_project.md) is the step-by-step runbook for setup, tests, training, checkpoints, evaluation, figures, ablations, and paper outputs.
- [Paper-Grade Method Notes](docs/paper_grade_method.md) explains which neuroscience claims are implemented directly and which are practical approximations.

## What V3 Implements

- Polarity-preserving event input as `[batch, polarity_channel, time, height, width]`.
- Support-based noise gating over event activity instead of signed event sums.
- Causal physics and learned prediction so future time bins do not leak backward.
- Named plasticity factors: `gamma_conf`, `gamma_pers`, `gamma_ctx`, `gamma_comp`, and final `beta`.
- Complementary memory stores: slow stable memory, faster adaptive memory, and an episodic buffer for high-surprise low-compressibility events.
- Memory-aware prediction using stable and adaptive context internally.
- Self-supervised regime pseudo-labels for stable, coherent motion, anomaly/transition, and noisy/unstructured states.
- Regime-conditioned sparse routing so anomaly-like residuals are preserved more than stable or noisy residuals.
- Config-driven training, evaluation, checkpoints, ablations, and paper result collection.

## Layout

- `predictive_vision/` - package code for model, data, metrics, training, evaluation, checkpoints, plotting, and reproduction
- `configs/train_paper.yaml` - default paper-grade training config
- `configs/eval_paper.yaml` - checkpoint evaluation config
- `configs/ablation_*.yaml` - ablation configs
- `train.py` - thin training CLI
- `eval.py` - thin evaluation CLI
- `plot_results.py` - thin checkpoint-backed plotting CLI
- `reproduce_paper.py` - collects run summaries into paper tables
- `tests/` - synthetic tests that do not require N-Caltech101

Generated `data/`, `cache/`, `experiments/`, and `paper_outputs/` directories are ignored by git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first dataset run uses Tonic to prepare N-Caltech101 under `./data` and transformed samples under `./cache`.

## Train

```bash
python train.py --config configs/train_paper.yaml
```

Each run creates an experiment folder under `experiments/` with resolved configs, deterministic split indices, metrics, summaries, and checkpoints.

## Evaluate

```bash
python eval.py --config configs/eval_paper.yaml --checkpoint experiments/<run_id>/checkpoints/best.pt
```

Evaluation loads the checkpoint without retraining and reuses the run's saved split indices.

## Generate Figures

```bash
python plot_results.py --run experiments/<run_id> --checkpoint experiments/<run_id>/checkpoints/best.pt
```

Figures are written to `experiments/<run_id>/figures/`.

## Reproduce Paper Tables

```bash
python reproduce_paper.py --runs experiments/<run_id> experiments/<ablation_run_id>
```

Outputs are written to `paper_outputs/`:

- `paper_results.json`
- `density_table.csv`
- `ablation_table.csv`

## Ablations

```bash
python train.py --config configs/ablation_no_memory.yaml
python train.py --config configs/ablation_no_compressibility.yaml
python train.py --config configs/ablation_no_regime_objective.yaml
python train.py --config configs/ablation_no_contextual_routing.yaml
python train.py --config configs/ablation_no_plasticity.yaml
```

## Tests

```bash
pytest
```

The test suite uses synthetic event tensors, so it can verify architecture behavior without downloading N-Caltech101.
