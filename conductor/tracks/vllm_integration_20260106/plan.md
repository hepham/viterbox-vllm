# Implementation Plan: vLLM Integration Enhancement

## Phase 1: Stabilization
Focus on fixing known issues and ensuring correctness.

- [x] Task: Audit vLLM internal API usage for compatibility with vLLM 0.10+ (commit 3ec486e)
- [~] Task: Fix CFG implementation to allow per-request tuning (not just env var)
- [ ] Task: Resolve batched request truncation edge cases
- [ ] Task: Implement proper error handling with actionable messages
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Stabilization' (Protocol in workflow.md)

## Phase 2: Multilingual Quality
Address quality gaps in multilingual support.

- [ ] Task: Implement Alignment Stream Analyzer to reduce errors/repetitions
- [ ] Task: Add learned speech positional encodings (or document workaround)
- [ ] Task: Implement Russian text stress processing
- [ ] Task: Test all 23 supported languages
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Multilingual Quality' (Protocol in workflow.md)

## Phase 3: Performance Optimization
Maximize throughput and efficiency.

- [ ] Task: Enable CUDA graphs without correctness issues
- [ ] Task: Profile and optimize S3Gen waveform generation (current bottleneck)
- [ ] Task: Add configurable diffusion steps (quality vs speed tradeoff)
- [ ] Task: Benchmark on reference hardware (RTX 3090, RTX 3060ti)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Performance Optimization' (Protocol in workflow.md)

## Phase 4: Documentation & Release
Prepare for wider adoption.

- [ ] Task: Update README with complete API documentation
- [ ] Task: Add migration guide from original Chatterbox
- [ ] Task: Document architecture with diagrams
- [ ] Task: Create example scripts for common use cases
- [ ] Task: Stabilize public API for 1.0.0 release
- [ ] Task: Conductor - User Manual Verification 'Phase 4: Documentation & Release' (Protocol in workflow.md)

## Dependencies
- Phase 2 can run in parallel with Phase 1
- Phase 3 requires Phase 1 completion
- Phase 4 requires Phases 1-3 completion
