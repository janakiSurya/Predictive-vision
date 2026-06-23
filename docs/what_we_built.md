# What We Built

This document explains the project from the beginning: the core idea, what was missing in the first version, what changed during the hard review, and what the current `predictive_vision_v3` codebase does.

## Core Idea

The project is a brain-inspired event-vision system. Instead of treating all event-camera activity as equally useful, it tries to keep only structured surprise:

- remove unsupported sensor noise early
- predict what event activity should happen next
- subtract prediction from observation
- keep meaningful residuals
- use plasticity and memory to decide what should be learned
- use regime context to decide what should be routed forward

The neuroscience inspiration is predictive coding: the system should not spend all of its compute replaying the world. It should suppress what is already expected and highlight what reality did that the model did not predict.

## Where We Started

The first codebase was a compact MVP. It had a single `model.py`, one quick run script, one benchmark script, one plotting script, a small dependency file, and a README.

That MVP already contained the outline of the idea:

- noise gate
- hybrid predictor
- residual/error extractor
- plasticity controller
- regime classifier
- sparse output

But the hard review found that several claims were not yet supported strongly enough by the code.

## What The Hard Review Found

The review found important gaps:

- Polarity channels were summed, so opposite polarity events could cancel each other out.
- The learned predictor used symmetric 3D convolution, which could leak future time bins into earlier predictions.
- The noise gate counted signed event values, so coherent negative events could be removed incorrectly.
- The regime gate was effectively inactive because its confidence threshold was too low.
- Plasticity was not really the paper's plasticity model yet.
- There were no stable, adaptive, or episodic memory stores.
- The regime classifier was residual-only and did not really understand context.
- Evaluation was not reproducible enough for paper claims.
- Checkpoints, configs, saved metrics, and paper-output collection were missing.

The conclusion was: the idea was strong, but the repo needed to move from demo code to paper-grade research code.

## What Changed

The project was upgraded into a modular, paper-grade implementation.

The old flat structure was replaced with a package:

```text
predictive_vision/
  config.py
  data.py
  model.py
  metrics.py
  training.py
  evaluation.py
  checkpoints.py
  plotting.py
  reproduction.py
```

The top-level scripts are now thin command-line wrappers:

- `train.py`
- `eval.py`
- `plot_results.py`
- `reproduce_paper.py`

The old MVP scripts and stale generated figures were removed. Figures are now produced only inside experiment folders.

## Current Architecture: predictive_vision_v3

The current architecture version is:

```text
predictive_vision_v3
```

It keeps the public event input format:

```text
[batch, polarity_channel, time, height, width]
```

The two event polarity channels are preserved throughout the model.

## Main Model Components

### Noise Gate

The noise gate removes unsupported events using local spatiotemporal support. It checks event activity with `abs(x) > 0`, not signed event values, so positive and negative events are both handled correctly.

This approximates early sensory filtering, similar to the idea that the visual system suppresses isolated unreliable signals before higher processing.

### Hybrid Predictor

The predictor combines two parts:

- a causal physics-style prior based on temporal persistence and local spatial propagation
- a learned causal predictor

The learned predictor uses causal convolution, so future time bins do not leak backward into earlier predictions.

### Complementary Memory Stores

The v3 model includes three memory stores:

- stable memory: slow-moving summary of long-term regularities
- adaptive memory: faster summary of recent patterns
- episodic buffer: compact storage for high-surprise low-compressibility events

These are practical PyTorch approximations of complementary learning systems. They are not a full biological hippocampus/neocortex model, but they give the architecture real multi-timescale memory behavior.

### Plasticity Controller

Plasticity is now broken into named factors:

- `gamma_conf`: confidence in the observation
- `gamma_pers`: temporal persistence of the residual
- `gamma_ctx`: contextual confidence from the regime classifier
- `gamma_comp`: compressibility/coherence of the residual
- `beta`: final update pressure

This makes plasticity inspectable. Instead of saying "the model gates learning," the code now exposes what is contributing to that gate.

### Regime Classifier

The regime classifier uses context-aware features from:

- cleaned events
- residual activity
- plasticity factors
- memory mismatch
- temporal change
- local coherence/compressibility

It outputs regime probabilities, confidence, regime IDs, pseudo-labels, transition score, and context gate.

### Regime Pseudo-Labels

The code creates self-supervised pseudo-labels for four regimes:

- stable or predictable
- coherent persistent motion
- anomaly or transition
- noisy or unstructured residual

This gives the regime classifier a training signal without requiring manually labeled regime data.

### Sparse Routing

The sparse output is now regime-conditioned. It preserves more residual activity for anomaly-like states and suppresses more residuals for stable or noisy states.

This is closer to the paper's claim that the system does not just compress data. It routes what is likely to be meaningful.

## Training And Evaluation System

Training is now config-driven:

```bash
python train.py --config configs/train_paper.yaml
```

Each experiment run saves:

- resolved config
- deterministic split indices
- metrics
- summary
- last checkpoint
- best checkpoint
- optional epoch checkpoints

Evaluation is checkpoint-only:

```bash
python eval.py --config configs/eval_paper.yaml --checkpoint experiments/<run_id>/checkpoints/best.pt
```

This means paper claims can be tied to saved checkpoints and saved summaries instead of temporary console output.

## Ablations

The repo includes ablation configs for the major neuroscience-heavy pieces:

- no memory
- no compressibility
- no regime objective
- no contextual routing
- no plasticity

These make it possible to test whether the brain-inspired parts actually matter.

## Paper Reproduction

The command:

```bash
python reproduce_paper.py --runs experiments/<run_id> experiments/<ablation_run_id>
```

collects run summaries into:

- `paper_outputs/paper_results.json`
- `paper_outputs/density_table.csv`
- `paper_outputs/ablation_table.csv`

This gives the project a path from trained checkpoints to paper-facing tables.

## Tests

The project includes synthetic tests that do not require N-Caltech101:

```bash
pytest
```

The tests check important behavior:

- polarity preservation
- negative event support
- causal prediction
- beta behavior
- compressibility behavior
- memory update behavior
- regime pseudo-labels
- routing behavior
- checkpoint save/load
- split reproducibility

## What Is Still Future Work

The project is now much closer to paper-grade, but some ideas remain approximations:

- memory stores are compact summaries, not full spatial memory maps
- compressibility is a coherence/variance proxy, not true Kolmogorov complexity
- regime labels are pseudo-labels, not manually annotated behavioral regimes
- the physics prior is still simple compared with full optical flow
- final paper claims should come from trained checkpoints, ablations, and repeated runs

The important shift is that the repo now has the machinery to test those claims honestly.
