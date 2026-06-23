# Predictive Vision

Prototype code for a brain-inspired, event-driven predictive vision pipeline built with PyTorch and [Tonic](https://github.com/neuromorphs/tonic). The project uses the `N-Caltech101` event-camera dataset to test a predictive-coding-style idea: reduce dense event streams into a sparser "informative surprise" signal while preserving enough structure for downstream recognition.

## What This Project Does

The core model, `FullPredictiveVisionPipeline`, processes binned event frames in several stages:

1. `NoiseGate`
   Removes isolated low-support events with a fixed 3D neighborhood filter.
2. `HybridPredictor`
   Blends a hand-coded temporal prediction rule with a learned 3D convolutional predictor.
3. `ErrorExtractor`
   Keeps only residuals whose magnitude exceeds a threshold.
4. `PlasticityController`
   Computes a beta gate from temporal persistence and spatial coherence.
5. `RegimeClassifier`
   Classifies temporal error patterns into latent regimes with a GRU.
6. `SparseOutput`
   Emits only confident, thresholded residual activity.

The intended outcome is a sparse representation that highlights unexpected or informative changes instead of replaying the full event stream.

The design is brain-inspired in the sense that it separates predictable sensory input from residual surprise, modulates learning pressure with a plasticity-style gate, and compresses perception toward sparse, behaviorally relevant signals.

## Repository Layout

- `model.py` - core predictive vision pipeline modules
- `run_mvp.py` - short verification/training loop for checking the pipeline end-to-end
- `benchmark.py` - empirical benchmark comparing raw events vs sparse pipeline output on classification
- `plot_results.py` - qualitative and quantitative visualization generation
- `requirements.txt` - minimal Python dependencies
- `figure_2_raster_plot.png` - saved example raster visualization
- `figure_3_density_reduction.png` - saved density reduction chart

## Requirements

- Python 3.10+ recommended
- PyTorch
- torchvision
- tonic
- matplotlib

`matplotlib` is used by `plot_results.py` but is not currently listed in `requirements.txt`.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib
```

## Dataset

All scripts use `tonic.datasets.NCALTECH101(save_to='./data')`.

On first run, Tonic will download or prepare the dataset under `./data`. The scripts then:

- map string labels to integer class IDs
- convert event streams to frame tensors with `ToFrame`
- use a fixed sensor size of `(304, 240, 2)`
- bin events into `10` temporal slices
- cache transformed samples under `./cache`

If you change transform settings, clear `./cache` before rerunning. `benchmark.py` already removes the cache at startup to avoid stale shape/label issues.

## Input Format

Tonic returns frames in roughly this layout:

```text
[batch, time, polarity, height, width]
```

The scripts convert them into the model format:

```text
[batch, channel, time, height, width]
```

They also sum the two polarity channels before inference/training:

```python
frames = frames.sum(dim=2, keepdim=True).permute(0, 2, 1, 3, 4).float()
```

## Running The Project

### 1. Quick pipeline verification

Runs a short optimization loop over a few batches and prints density and training diagnostics:

```bash
python run_mvp.py
```

This script stops after 3 batches and reports:

- raw input event density
- denoised density
- final sparse output density
- hybrid blend alpha
- plasticity gate beta
- modulated reconstruction loss

### 2. Benchmark the sparse representation

Trains:

- a baseline classifier on raw event volumes
- the predictive pipeline jointly with
- a classifier on the pipeline's sparse output

Run:

```bash
python benchmark.py
```

For each epoch, it prints validation metrics including:

- density reduction from raw to sparse events
- baseline classification accuracy
- sparse classification accuracy

This is the main script to use if you want to test whether sparsification preserves downstream task utility.

### 3. Generate figures

Creates two PNG outputs from one sample:

- `figure_2_raster_plot.png`
- `figure_3_density_reduction.png`

Run:

```bash
python plot_results.py
```

## Model Notes

- `HybridPredictor` includes a learnable blend parameter `alpha` between the physics-inspired predictor and the learned predictor.
- The learned predictor uses `ReLU` as its output activation, allowing predictions above `1.0`.
- Reconstruction-style training is modulated by the `beta` gate from `PlasticityController`.
- The final sparse output is filtered using regime confidence from `RegimeClassifier`.

## Current Limitations

- No command-line interface or config system yet
- No saved checkpoints or experiment logging
- No test suite
- `matplotlib` is missing from `requirements.txt`
- Device handling is only present in `benchmark.py`; `run_mvp.py` and `plot_results.py` run on default device
- Data download/caching behavior depends on Tonic and local dataset availability

## Suggested Next Steps

- Add `matplotlib` to `requirements.txt`
- Add a proper training/evaluation config
- Save model checkpoints and benchmark summaries
- Add unit tests for tensor shapes and pipeline stage outputs
- Add CLI flags for cache paths, batch size, epochs, and thresholds
