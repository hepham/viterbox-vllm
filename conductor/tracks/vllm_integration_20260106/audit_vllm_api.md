# vLLM Internal API Usage Audit

## Date: 2026-01-06
## vLLM Version: 0.10.0 (pinned in pyproject.toml)

## Summary

The codebase uses vLLM 0.10.0 with custom multimodal processing for the T3 speech synthesis model. This audit identifies internal API usage and compatibility concerns.

---

## Imports Analysis (t3.py)

### Core vLLM Imports

| Import | Status | Notes |
|--------|--------|-------|
| `vllm.config.VllmConfig` | ✅ Stable | Core config class |
| `vllm.config.ModelConfig` | ✅ Stable | Model configuration |
| `vllm.model_executor.layers.logits_processor.LogitsProcessor` | ⚠️ Internal | Used for custom logits processing |
| `vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead` | ⚠️ Internal | Tensor-parallel embedding |
| `vllm.model_executor.models.interfaces.MultiModalEmbeddings` | ✅ Stable | Public multimodal interface |
| `vllm.model_executor.models.interfaces.SupportsMultiModal` | ✅ Stable | Public interface |
| `vllm.model_executor.models.interfaces_base.VllmModelForTextGeneration` | ⚠️ Internal | Base class for generation models |
| `vllm.model_executor.models.llama.LlamaModel` | ✅ Stable | Standard model implementation |
| `vllm.model_executor.sampling_metadata.SamplingMetadata` | ⚠️ Internal | Used in compute_logits |
| `vllm.multimodal.MULTIMODAL_REGISTRY` | ✅ Stable | Registry for multimodal processors |
| `vllm.multimodal.inputs.*` | ✅ Stable | MultiModalKwargs, etc. |
| `vllm.multimodal.parse.*` | ✅ Stable | Parsing utilities |
| `vllm.multimodal.processing.*` | ✅ Stable | Processing pipeline |
| `vllm.multimodal.profiling.BaseDummyInputsBuilder` | ✅ Stable | Profiling support |
| `vllm.sequence.IntermediateTensors` | ⚠️ Internal | Used in forward pass |

### tts.py Imports

| Import | Status | Notes |
|--------|--------|-------|
| `vllm.LLM` | ✅ Stable | Main public API |
| `vllm.SamplingParams` | ✅ Stable | Public API for generation params |

---

## Compatibility Concerns

### 1. CFG_SCALE Environment Variable (Line 296)
```python
self.cfg_scale = float(os.environ.get("CHATTERBOX_CFG_SCALE", "0.5"))
```
**Issue:** CFG scale is read once at model initialization from environment variable. This does not allow per-request tuning.
**Impact:** Medium - Users cannot adjust CFG per request
**Fix Needed:** Task 2 addresses this

### 2. Hidden Size Hack (Line 263)
```python
vllm_config.model_config.hf_config.hidden_size = 1024
```
**Issue:** Modifies internal vLLM config to work around CFG dual-embedding approach
**Risk:** Could break if vLLM changes config handling
**Mitigation:** Document this hack; monitor vLLM releases

### 3. SPEECH_TOKEN_OFFSET Hack (Line 49)
```python
SPEECH_TOKEN_OFFSET = 2500
```
**Issue:** Uses token offset to distinguish prefill vs decode tokens
**Risk:** Could conflict with models with vocabulary > 2500
**Status:** Acceptable for current use case (speech tokens < 2500)

### 4. MultiModalKwargs Construction (Lines 224-232)
```python
new_mm_kwargs = MultiModalKwargs.from_items([
    MultiModalKwargsItem.from_elems(...)
])
```
**Status:** Uses public API correctly

### 5. PlaceholderRange Hack (Lines 246-248)
```python
"conditionals": [PlaceholderRange(offset=0, length=len(final_prompt_ids), is_embed=None)]
```
**Issue:** Uses placeholder to inject embeddings across entire prompt
**Risk:** Medium - depends on vLLM multimodal placeholder behavior
**Status:** Works in 0.10.0, monitor for changes

---

## vLLM 0.10.x Breaking Changes (from web research)

Based on vLLM release notes:
1. V0 engine cleanup - various backends removed (not affecting us)
2. PyTorch 2.8.0 upgrade - no direct impact
3. API breaking changes around prompt_token_ids fallback - we don't use this pattern
4. LoRA extra vocab size deprecation - not applicable

**Conclusion:** No immediate breaking changes affect this codebase in 0.10.0

---

## Recommendations

1. **Pin vLLM version** ✅ Already done (vllm==0.10.0)
2. **Add version compatibility tests** - Create tests that verify key APIs work
3. **Document internal API usage** - This file serves as documentation
4. **Monitor vLLM releases** - Check 0.10.1, 0.10.2 release notes before upgrading
5. **Fix CFG per-request tuning** - Tracked as Task 2

---

## API Stability Assessment

| Component | Stability | Action |
|-----------|-----------|--------|
| LLM/SamplingParams | High | Safe to use |
| Multimodal Registry | Medium-High | Monitor changes |
| LogitsProcessor | Medium | Could change; wrap in abstraction |
| VllmModelForTextGeneration | Medium | Internal but stable pattern |
| SamplingMetadata | Low | Most likely to change; isolate usage |

---

## Test Coverage Needed

1. Basic generation with vLLM backend
2. Batched generation correctness
3. CFG scale application
4. Multimodal embedding injection
5. Long sequence handling (near max_model_len)
