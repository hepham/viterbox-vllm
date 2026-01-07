"""
Alignment Stream Analyzer for vLLM-based Chatterbox TTS.

This is a simplified version that doesn't require attention weights (which are
unavailable in vLLM's optimized attention kernels). Instead, it uses token-based
heuristics to detect and mitigate:
1. False starts - suppress EOS for initial frames
2. Long tails - force EOS when speech exceeds max length relative to text
3. Token repetition - force EOS on consecutive repeated tokens
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from vllm.model_executor.sampling_metadata import SamplingMetadata


logger = logging.getLogger(__name__)


@dataclass
class AlignmentAnalysisResultVllm:
    """Result of alignment analysis for a single sequence."""
    false_start: bool
    long_tail: bool
    repetition: bool
    token_repetition: bool
    complete: bool
    position: int


@dataclass
class SequenceState:
    """Per-sequence state for tracking generation progress."""
    text_len: int
    max_frames: int
    frames_generated: int
    started: bool
    complete: bool


class AlignmentStreamAnalyzerVllm:
    """
    Token-based alignment analyzer for vLLM.
    
    Unlike the HuggingFace version, this doesn't use attention weights.
    Instead it relies on:
    - EOS suppression for initial frames (false start prevention)
    - Max speech length relative to text length (long tail prevention)  
    - Token repetition detection (repetition prevention)
    """
    
    def __init__(
        self,
        eos_idx: int = 0,
        max_frames_factor: float = 5.0,
        min_frames_before_eos: int = 15,
        min_speech_frames: int = 20,
        repetition_threshold: int = 2,
        enabled: bool = True,
    ):
        """
        Args:
            eos_idx: EOS token index in speech vocabulary (before SPEECH_TOKEN_OFFSET)
            max_frames_factor: Max speech frames = text_len * max_frames_factor
            min_frames_before_eos: Suppress EOS for this many initial frames
            min_speech_frames: Minimum speech frames regardless of text length
            repetition_threshold: Force EOS after this many consecutive same tokens
            enabled: Whether the analyzer is active
        """
        self.eos_idx = eos_idx
        self.max_frames_factor = max_frames_factor
        self.min_frames_before_eos = min_frames_before_eos
        self.min_speech_frames = min_speech_frames
        self.repetition_threshold = repetition_threshold
        self.enabled = enabled
        
        self._sequence_states: Dict[int, SequenceState] = {}
        self._sequence_tokens: Dict[int, list] = {}
    
    def reset(self):
        """Clear all sequence state."""
        self._sequence_states.clear()
        self._sequence_tokens.clear()
    
    def _get_or_init_state(
        self, 
        seq_id: int, 
        text_len: int,
        existing_speech_frames: int = 0
    ) -> SequenceState:
        """Get or initialize state for a sequence."""
        if seq_id not in self._sequence_states:
            max_frames = max(
                self.min_speech_frames,
                int(text_len * self.max_frames_factor)
            )
            self._sequence_states[seq_id] = SequenceState(
                text_len=text_len,
                max_frames=max_frames,
                frames_generated=existing_speech_frames,
                started=False,
                complete=False,
            )
            self._sequence_tokens[seq_id] = []
        return self._sequence_states[seq_id]
    
    def _check_token_repetition(self, seq_id: int, last_token: Optional[int] = None) -> bool:
        """Check if the last N tokens are all identical."""
        tokens = self._sequence_tokens.get(seq_id, [])
        
        if last_token is not None:
            tokens = tokens + [last_token]
        
        if len(tokens) < self.repetition_threshold:
            return False
        
        recent = tokens[-self.repetition_threshold:]
        return len(set(recent)) == 1
    
    def _record_token(self, seq_id: int, token_id: int):
        """Record a generated token for repetition tracking."""
        if seq_id not in self._sequence_tokens:
            self._sequence_tokens[seq_id] = []
        self._sequence_tokens[seq_id].append(token_id)
        if len(self._sequence_tokens[seq_id]) > 16:
            self._sequence_tokens[seq_id] = self._sequence_tokens[seq_id][-16:]
    
    def step(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        text_lengths: Optional[Dict[int, int]] = None,
    ) -> torch.Tensor:
        """
        Apply alignment heuristics to modify logits.
        
        Args:
            logits: Post-CFG logits [num_seqs, vocab_size]
            sampling_metadata: vLLM sampling metadata
            text_lengths: Optional dict mapping seq_id -> text token count
            
        Returns:
            Modified logits tensor
        """
        if not self.enabled:
            return logits
        
        logits = logits.clone()
        
        for seq_idx, seq_group in enumerate(sampling_metadata.seq_groups):
            seq_ids = seq_group.seq_ids
            
            for local_idx, seq_id in enumerate(seq_ids):
                if seq_idx + local_idx >= logits.shape[0]:
                    continue
                    
                logit_idx = seq_idx + local_idx
                seq_logits = logits[logit_idx]
                
                text_len = 50
                if text_lengths and seq_id in text_lengths:
                    text_len = text_lengths[seq_id]
                
                state = self._get_or_init_state(seq_id, text_len)
                state.frames_generated += 1
                
                false_start = False
                long_tail = False
                token_repetition = False
                
                if state.frames_generated < self.min_frames_before_eos:
                    false_start = True
                    seq_logits[self.eos_idx] = -2**15
                else:
                    state.started = True
                
                if state.frames_generated > state.max_frames:
                    long_tail = True
                    state.complete = True
                    logger.warning(
                        f"Long tail detected for seq {seq_id}: "
                        f"{state.frames_generated} frames > max {state.max_frames}"
                    )
                    seq_logits.fill_(-2**15)
                    seq_logits[self.eos_idx] = 2**15
                
                if not long_tail and self._check_token_repetition(seq_id):
                    recent_tokens = self._sequence_tokens.get(seq_id, [])[-self.repetition_threshold:]
                    if recent_tokens:
                        token_repetition = True
                        state.complete = True
                        logger.warning(
                            f"Token repetition detected for seq {seq_id}: "
                            f"token {recent_tokens[-1]} repeated {self.repetition_threshold}x"
                        )
                        seq_logits.fill_(-2**15)
                        seq_logits[self.eos_idx] = 2**15
                
                logits[logit_idx] = seq_logits
        
        return logits
    
    def record_sampled_token(self, seq_id: int, token_id: int):
        """
        Record a sampled token after sampling completes.
        Call this from outside after vLLM samples the token.
        """
        self._record_token(seq_id, token_id)
    
    def cleanup_sequence(self, seq_id: int):
        """Remove state for a completed/aborted sequence."""
        self._sequence_states.pop(seq_id, None)
        self._sequence_tokens.pop(seq_id, None)
