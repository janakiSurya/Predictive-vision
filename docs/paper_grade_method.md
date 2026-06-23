# Paper-Grade Method Notes

This project implements a practical PyTorch approximation of a brain-inspired predictive vision system. It is not a full biological spiking simulator.

## Directly Implemented

- Predictive coding residuals: the model predicts cleaned event input and emits thresholded residuals.
- Retinal-style support gating: early events are filtered by local spatiotemporal support.
- Complementary learning stores: stable, adaptive, and episodic memories are represented as checkpointed non-gradient buffers.
- Neuromodulatory-style plasticity: update pressure is gated by confidence, persistence, context, and compressibility factors.
- Contextual gating: a GRU estimates regime probabilities and sparse routing depends on inferred regime.

## Engineering Approximations

- Memory stores are compact channel summaries, not full hippocampal/neocortical models.
- Regimes are trained from deterministic pseudo-labels instead of hand-labeled behavioral contexts.
- Compressibility is approximated with local coherence versus local variance, not Kolmogorov complexity.
- Episodic memory stores compact summaries of surprising batches, not raw event histories.

## Future Work

- Replace compact memory summaries with spatially structured memory maps.
- Add stronger optical-flow or event-flow physics priors.
- Learn regime labels with clustering or contrastive temporal objectives.
- Add multi-seed statistical reporting and hardware latency/energy measurements.
