# Calibration Probe Integration Plan

## Objectives
- Capture intermediate activations during `forward.py` runs to support calibration probing.
- Keep the hook workflow optional and memory-aware so default runs stay unchanged.
- Produce a quick estimate of memory requirements before scaling to the full MMLU set.

## Immediate Tasks
1. Audit `forward.py` to map where the Hugging Face model forward call occurs and what tensors are already moved between devices, so hooks can be attached safely.
2. Decide which module(s) to probe (initially a representative middle transformer block) and design a simple configuration interface for specifying probe targets.
3. Implement hook registration and teardown utilities that capture activations per batch without breaking existing caching logic.
4. Store captured activations in CPU memory (with optional dtype casting) and provide a lightweight buffer API for downstream analysis.
5. Run a limited evaluation batch to verify shapes/contents and compute approximate memory usage per question to inform larger calibration runs.

## Open Questions
- Do we want to support multiple simultaneous probe layers, or keep a single configurable layer for now?
- Should activation storage be implemented as in-memory buffers, on-disk serialization, or both?
- How should we parameterize hook activation (CLI flag, config file, or environment variable)?
