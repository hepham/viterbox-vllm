"""
Example demo for Viterbox - Vietnamese TTS model
Uses chatterbox library (faster branch) with dolly-vn/viterbox model
Following the exact same approach as viterbox-tts original
"""
import os
import torch
import numpy as np
import torchaudio as ta
from pathlib import Path
from huggingface_hub import snapshot_download

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, Conditionals, punc_norm
from chatterbox.models.t3 import T3
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.t3.modules.cond_enc import T3Cond
from chatterbox.models.s3gen import S3Gen
from chatterbox.models.s3tokenizer import drop_invalid_tokens
from chatterbox.models.voice_encoder import VoiceEncoder
from chatterbox.models.tokenizers import MTLTokenizer
from safetensors.torch import load_file as load_safetensors
import torch.nn.functional as F


VITERBOX_REPO_ID = "dolly-vn/viterbox"

# Add Vietnamese to supported languages for Viterbox
VITERBOX_SUPPORTED_LANGUAGES = {
    "vi": "Vietnamese",
    "ar": "Arabic",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "ms": "Malay",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "sw": "Swahili",
    "tr": "Turkish",
    "zh": "Chinese",
}


class ViterboxTTS(ChatterboxMultilingualTTS):
    """Viterbox TTS - Vietnamese fine-tuned model based on Chatterbox"""

    @classmethod
    def get_supported_languages(cls):
        """Return dictionary of supported language codes and names including Vietnamese."""
        return VITERBOX_SUPPORTED_LANGUAGES.copy()

    def generate(
        self,
        text,
        language_id,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens=1000,
        max_cache_len=1500,
        repetition_penalty=2.0,  # Viterbox uses 2.0, not 1.2
        min_p=0.05,
        top_p=1.0,
        t3_params={},
    ):
        # Override validation to use Viterbox supported languages (includes Vietnamese)
        if language_id and language_id.lower() not in VITERBOX_SUPPORTED_LANGUAGES:
            supported_langs = ", ".join(VITERBOX_SUPPORTED_LANGUAGES.keys())
            raise ValueError(
                f"Unsupported language_id '{language_id}'. "
                f"Supported languages: {supported_langs}"
            )
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1, dtype=_cond.speaker_emb.dtype),
            ).to(device=self.device)

        # Norm and tokenize text - use Viterbox format: "[vi] text" with space
        text = punc_norm(text)
        
        # Add language prefix with space (like viterbox-tts does)
        if language_id:
            text = f"[{language_id.lower()}] {text}"
        
        # Tokenize without lowercase (viterbox-tts doesn't lowercase)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=None, lowercase=False, nfkd_normalize=False).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                cfg_weight=cfg_weight,
                max_cache_len=max_cache_len,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                **t3_params,
            )
            speech_tokens = speech_tokens[0]

            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            # Remove last token to avoid click artifacts (like viterbox-tts does)
            if len(speech_tokens) > 1:
                speech_tokens = speech_tokens[:-1]
                
            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            
            # Apply fade-in to prevent click artifacts at the start (like viterbox-tts does)
            fade_in_samples = int(0.005 * self.sr)  # 5ms fade-in
            if len(wav) > fade_in_samples:
                fade_curve = np.linspace(0.0, 1.0, fade_in_samples)
                wav[:fade_in_samples] = wav[:fade_in_samples] * fade_curve
            
            # Apply fade-out to prevent click artifacts at the end
            fade_out_samples = int(0.01 * self.sr)  # 10ms fade-out
            if len(wav) > fade_out_samples:
                fade_curve = np.linspace(1.0, 0.0, fade_out_samples)
                wav[-fade_out_samples:] = wav[-fade_out_samples:] * fade_curve
            
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    @classmethod
    def from_pretrained(cls, device: torch.device) -> 'ViterboxTTS':
        ckpt_dir = Path(
            snapshot_download(
                repo_id=VITERBOX_REPO_ID,
                repo_type="model",
                revision="main",
                allow_patterns=[
                    "ve.pt",
                    "t3_ml24ls_v2.safetensors",
                    "s3gen.pt",
                    "tokenizer_vi_expanded.json",
                    "mtl_tokenizer.json",
                    "conds.pt",
                    "config.json",
                ],
                token=os.getenv("HF_TOKEN"),
            )
        )
        return cls.from_local(ckpt_dir, device)

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ViterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Load Voice Encoder
        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True, map_location=device))
        ve.to(device).eval()

        # Create T3Config with Viterbox vocab size (2549)
        t3_config = T3Config.multilingual()
        t3_config.text_tokens_dict_size = 2549  # Viterbox uses expanded Vietnamese vocab
        
        t3 = T3(t3_config)
        t3_state = load_safetensors(ckpt_dir / "t3_ml24ls_v2.safetensors", device=device)
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        # Load S3Gen
        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True, map_location=device))
        s3gen.to(device).eval()

        # Load Vietnamese tokenizer
        tokenizer = MTLTokenizer(str(ckpt_dir / "tokenizer_vi_expanded.json"))

        # Load default conditioning
        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=device).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)


if __name__ == "__main__":
    # Auto-detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print("Loading Viterbox model...")

    model = ViterboxTTS.from_pretrained(device=device)
    print("Model loaded!")

    # Use Vietnamese voice sample for voice cloning
    AUDIO_PROMPT = "mau.wav"

    # Vietnamese TTS with Vietnamese voice
    text_vi = "Xin chào! Đây là mô hình chuyển văn bản thành giọng nói tiếng Việt Viterbox."
    print(f"Generating Vietnamese: '{text_vi}'")
    wav = model.generate(
        text_vi, 
        language_id="vi", 
        audio_prompt_path=AUDIO_PROMPT,
        cfg_weight=0.5,
        exaggeration=0.5,
        repetition_penalty=2.0,
    )
    ta.save("viterbox_vietnamese.wav", wav, model.sr)
    print("Saved: viterbox_vietnamese.wav")

    # With expressive settings
    text_expressive = "Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố. Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc. Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên. Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"
    print(f"Generating expressive: '{text_expressive}'")
    wav = model.generate(
        text_expressive,
        language_id="vi",
        audio_prompt_path=AUDIO_PROMPT,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        repetition_penalty=2.0,
    )
    ta.save("viterbox_expressive.wav", wav, model.sr)
    print("Saved: viterbox_expressive.wav")

    # English for comparison
    text_en = "Hello! This is the Viterbox text-to-speech model speaking English."
    print(f"Generating English: '{text_en}'")
    wav = model.generate(
        text_en,
        language_id="en",
        audio_prompt_path=AUDIO_PROMPT,
        cfg_weight=0.5,
        exaggeration=0.5,
        repetition_penalty=2.0,
    )
    ta.save("viterbox_english.wav", wav, model.sr)
    print("Saved: viterbox_english.wav")

    print("\nDone!")
