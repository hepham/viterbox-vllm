# Speech Positional Embeddings in vLLM

## Background

The original Chatterbox T3 model uses learned speech positional embeddings
(`LearnedPositionEmbeddings`) to encode position information for speech tokens.
This helps the model understand where each speech token falls within the speech
sequence.

## Current vLLM Implementation Status

### What Works
- ✅ Text positional embeddings are applied correctly during prefill
- ✅ Start-of-speech token receives positional embedding (position 0)
- ✅ Conditioning embeddings are handled correctly

### Current Limitation
- ❌ Speech positional embeddings for decode tokens (position 1+) are NOT applied

### Why This Is Hard in vLLM

1. **KV-Cache Architecture**: vLLM processes one token at a time during decode,
   with previous tokens cached. The embeddings are only computed once per token.

2. **Position Tracking**: vLLM's `positions` tensor provides absolute positions
   within the full sequence (conditioning + text + speech), not speech-specific
   positions.

3. **Batching Complexity**: Multiple sequences are batched together, each at
   different speech positions. Tracking per-sequence speech positions requires
   careful coordination with vLLM's scheduler.

## Observed Impact

In practice, the missing speech positional embeddings have **minimal impact**
on output quality. This is because:

1. The T3 Llama backbone uses RoPE (Rotary Position Embeddings) which provides
   relative position information
2. The conditioning and text embeddings provide strong context for generation
3. The start-of-speech positional embedding anchors the speech sequence

Benchmark comparisons show comparable quality between vLLM and the original
HuggingFace implementation despite this difference.

## Future Implementation Path

To properly implement speech positional embeddings, we would need to:

1. Track per-sequence speech position counts in `T3VllmModel`
2. Look up speech positions for each decode token during `get_input_embeddings()`
3. Add the appropriate positional embedding from `precomputed_speech_pos_emb`
4. Handle sequence lifecycle (new sequences, completed sequences, aborted)

This could be done by:
```python
# In T3VllmModel.__init__:
self._speech_positions: dict[int, int] = {}  # seq_id -> current speech position

# In get_input_embeddings() for decode tokens:
# Note: Requires access to seq_id, which may need SamplingMetadata passed through
speech_pos = self._speech_positions.get(seq_id, 1)
embeds = self.speech_emb(input_ids - SPEECH_TOKEN_OFFSET)
embeds = embeds + self.precomputed_speech_pos_emb[speech_pos:speech_pos+len(embeds)]
self._speech_positions[seq_id] = speech_pos + len(embeds)
```

The challenge is that `get_input_embeddings()` doesn't currently receive
sequence ID information from vLLM's multimodal pipeline.

## Workaround

For now, the system works without explicit speech positional embeddings on
decode tokens. If quality issues are observed on long speech outputs, consider:

1. Using the `repetition_penalty` parameter (default 2.0) to reduce repetition
2. Enabling the alignment analyzer which monitors for repetition/hallucination
3. Limiting max speech length via `CHATTERBOX_MAX_FRAMES_FACTOR`

## Configuration

The current behavior can be observed in:
- `chatterbox-vllm/src/chatterbox_vllm/models/t3/t3.py`

The precomputed embeddings are available but only applied to start-of-speech:
- `self.precomputed_speech_pos_emb` - computed in `load_weights()`
- Applied at lines 534, 585, 609 for start-of-speech token only
