# Running The Project

This is the practical step-by-step guide for running the `predictive_vision_v3` project.

Use this when you want to train a model, save checkpoints, evaluate a checkpoint, generate figures, run ablations, and collect paper-style outputs.

## Step 0: Start In The Project Folder

From the terminal, go to the project root:

```bash
cd "/Volumes/My stuff/Predictive_vision"
```

## Step 1: Create A Virtual Environment

Create a Python environment:

```bash
python -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

If your machine uses `python3` instead of `python`, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 2: Install Dependencies

Install the project requirements:

```bash
pip install -r requirements.txt
```

The important packages are:

- `torch`
- `torchvision`
- `tonic`
- `matplotlib`
- `numpy`
- `pyyaml`
- `h5py`
- `pytest`

## Step 3: Run The Synthetic Tests

Before training on the real dataset, run:

```bash
pytest
```

These tests use synthetic event tensors. They do not require N-Caltech101.

Expected result:

```text
12 passed
```

This verifies the core model behavior: polarity preservation, causal prediction, memory updates, plasticity factors, regime pseudo-labels, sparse routing, checkpoint save/load, and reproducibility helpers.

## Step 4: Train The Main Paper Model

Run:

```bash
python train.py --config configs/train_paper.yaml
```

What happens:

- Tonic prepares N-Caltech101 under `./data`
- transformed event frames are cached under `./cache`
- a deterministic train/val/test split is saved
- the v3 predictive vision model trains
- a raw baseline classifier trains
- a sparse-output classifier trains
- metrics are written after training and validation
- checkpoints are saved

The first run may take longer because the dataset and cache need to be prepared.

## Step 5: Find The Saved Model

Training creates a new run folder under:

```text
experiments/
```

The folder name looks like:

```text
experiments/<timestamp>_paper_grade_v3_<config_hash>/
```

Inside that run folder, the main files are:

```text
resolved_config.yaml
resolved_config.json
metrics.jsonl
summary.json
splits/split_indices.json
checkpoints/last.pt
checkpoints/best.pt
```

Use `best.pt` for paper evaluation unless you specifically need the latest checkpoint.

## Step 6: Resume Training If Needed

To resume from the latest checkpoint:

```bash
python train.py --config configs/train_paper.yaml --resume experiments/<run_id>/checkpoints/last.pt
```

Use this if training stopped early or you want to continue the same run.

The resume path restores:

- model weights
- memory buffers
- classifier weights
- optimizer states
- scheduler states
- epoch
- global step
- best score

## Step 7: Evaluate The Saved Model

Evaluate the best checkpoint:

```bash
python eval.py --config configs/eval_paper.yaml --checkpoint experiments/<run_id>/checkpoints/best.pt
```

This does not retrain the model.

It loads the checkpoint and evaluates the configured split, usually `test`.

Expected output folder:

```text
experiments/<run_id>/eval/
```

Expected files:

```text
test_summary.json
resolved_eval_config.yaml
```

The summary includes metrics such as:

- raw accuracy
- sparse accuracy
- raw density
- sparse density
- data reduction
- beta statistics
- plasticity factor means
- regime entropy
- transition rate
- memory norms
- episodic rate

## Step 8: Generate Figures

Generate checkpoint-backed figures:

```bash
python plot_results.py --run experiments/<run_id> --checkpoint experiments/<run_id>/checkpoints/best.pt
```

Expected output folder:

```text
experiments/<run_id>/figures/
```

Expected files:

```text
figure_2_raster_plot.png
figure_3_density_reduction.png
```

These are generated from the trained checkpoint, not from an untrained one-off script.

## Step 9: Run Ablations

Ablations help test whether the neuroscience-heavy parts matter.

Run each ablation separately:

```bash
python train.py --config configs/ablation_no_memory.yaml
python train.py --config configs/ablation_no_compressibility.yaml
python train.py --config configs/ablation_no_regime_objective.yaml
python train.py --config configs/ablation_no_contextual_routing.yaml
python train.py --config configs/ablation_no_plasticity.yaml
```

Each ablation creates its own folder under `experiments/`.

Recommended order:

1. Train the main model with `configs/train_paper.yaml`
2. Train `ablation_no_memory`
3. Train `ablation_no_plasticity`
4. Train `ablation_no_compressibility`
5. Train `ablation_no_regime_objective`
6. Train `ablation_no_contextual_routing`

Then evaluate each run with `eval.py`.

## Step 10: Collect Paper Outputs

After training and evaluating the main model plus ablations, collect paper tables:

```bash
python reproduce_paper.py --runs experiments/<main_run_id> experiments/<ablation_run_id>
```

You can pass multiple run folders:

```bash
python reproduce_paper.py --runs \
  experiments/<main_run_id> \
  experiments/<no_memory_run_id> \
  experiments/<no_plasticity_run_id> \
  experiments/<no_compressibility_run_id>
```

Expected output folder:

```text
paper_outputs/
```

Expected files:

```text
paper_results.json
density_table.csv
ablation_table.csv
```

These files are the easiest way to move from experiment outputs to paper tables.

## Step 11: Understand Generated Folders

These folders are generated and ignored by git:

```text
data/
cache/
experiments/
paper_outputs/
venv/ or .venv/
__pycache__/
```

Do not delete `data/` or `cache/` unless you intentionally want to redownload or rebuild dataset artifacts.

## Step 12: Common Commands

Run tests:

```bash
pytest
```

Train main model:

```bash
python train.py --config configs/train_paper.yaml
```

Resume training:

```bash
python train.py --config configs/train_paper.yaml --resume experiments/<run_id>/checkpoints/last.pt
```

Evaluate:

```bash
python eval.py --config configs/eval_paper.yaml --checkpoint experiments/<run_id>/checkpoints/best.pt
```

Plot:

```bash
python plot_results.py --run experiments/<run_id> --checkpoint experiments/<run_id>/checkpoints/best.pt
```

Reproduce paper tables:

```bash
python reproduce_paper.py --runs experiments/<run_id> experiments/<ablation_run_id>
```

## Step 13: What To Report From A Run

For a serious result, report:

- config used
- checkpoint used
- train/val/test split seed
- raw accuracy
- sparse accuracy
- raw density
- sparse density
- data reduction
- beta mean
- plasticity factor means
- regime entropy
- transition rate
- memory statistics
- ablation comparison

The files to inspect first are:

```text
experiments/<run_id>/summary.json
experiments/<run_id>/eval/test_summary.json
paper_outputs/ablation_table.csv
paper_outputs/density_table.csv
```

## Step 14: Current Limitation

The synthetic tests prove the code path works without the real dataset. Real paper claims should only be made after running the full N-Caltech101 training, evaluation, figures, and ablations from saved checkpoints.
