"""Profile inference time for each component"""
import time
import torch
import torchaudio as ta
import numpy as np
from pathlib import Path

TEXT = """Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố. Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc. Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên. Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"""

AUDIO_PROMPT = "mau.wav"

def profile_viterbox():
    from example_viterbox import ViterboxTTS, punc_norm, VITERBOX_SUPPORTED_LANGUAGES
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    from chatterbox.models.s3tokenizer import drop_invalid_tokens
    import torch.nn.functional as F
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Text length: {len(TEXT)} chars")
    print("="*60)
    
    # Load model
    print("Loading model...")
    model = ViterboxTTS.from_pretrained(device=device)
    print("Model loaded!\n")
    
    # Warmup
    print("Warmup run...")
    _ = model.generate("Xin chào", language_id="vi", audio_prompt_path=AUDIO_PROMPT)
    torch.cuda.synchronize()
    print("Warmup done!\n")
    
    # Profile each step
    print("="*60)
    print("PROFILING EACH STEP")
    print("="*60)
    
    times = {}
    
    # 1. Prepare conditionals (voice encoder + audio processing)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.prepare_conditionals(AUDIO_PROMPT, exaggeration=0.5)
    torch.cuda.synchronize()
    times["1. Prepare Conditionals (VoiceEncoder)"] = time.perf_counter() - t0
    
    # 2. Text processing & tokenization
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    text = punc_norm(TEXT)
    text = f"[vi] {text}"
    text_tokens = model.tokenizer.text_to_tokens(text, language_id=None, lowercase=False, nfkd_normalize=False).to(device)
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
    sot = model.t3.hp.start_text_token
    eot = model.t3.hp.stop_text_token
    text_tokens = F.pad(text_tokens, (1, 0), value=sot)
    text_tokens = F.pad(text_tokens, (0, 1), value=eot)
    torch.cuda.synchronize()
    times["2. Text Tokenization"] = time.perf_counter() - t0
    print(f"   Text tokens: {text_tokens.shape[1]}")
    
    # 3. T3 inference (autoregressive - THE BOTTLENECK)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        speech_tokens = model.t3.inference(
            t3_cond=model.conds.t3,
            text_tokens=text_tokens,
            max_new_tokens=1000,
            temperature=0.8,
            cfg_weight=0.5,
            max_cache_len=1500,
            repetition_penalty=2.0,
            min_p=0.05,
            top_p=1.0,
        )
    torch.cuda.synchronize()
    times["3. T3 Inference (Autoregressive)"] = time.perf_counter() - t0
    
    speech_tokens = speech_tokens[0]
    speech_tokens = drop_invalid_tokens(speech_tokens)
    if len(speech_tokens) > 1:
        speech_tokens = speech_tokens[:-1]
    speech_tokens = speech_tokens.to(device)
    print(f"   Speech tokens: {len(speech_tokens)}")
    
    # 4. S3Gen inference (vocoder)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        wav, _ = model.s3gen.inference(
            speech_tokens=speech_tokens,
            ref_dict=model.conds.gen,
        )
    torch.cuda.synchronize()
    times["4. S3Gen (Vocoder)"] = time.perf_counter() - t0
    
    wav = wav.squeeze(0).detach().cpu().numpy()
    
    # 5. Post-processing (fade in/out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fade_in_samples = int(0.005 * model.sr)
    if len(wav) > fade_in_samples:
        fade_curve = np.linspace(0.0, 1.0, fade_in_samples)
        wav[:fade_in_samples] = wav[:fade_in_samples] * fade_curve
    fade_out_samples = int(0.01 * model.sr)
    if len(wav) > fade_out_samples:
        fade_curve = np.linspace(1.0, 0.0, fade_out_samples)
        wav[-fade_out_samples:] = wav[-fade_out_samples:] * fade_curve
    times["5. Fade In/Out"] = time.perf_counter() - t0
    
    # 6. Watermarking
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    watermarked_wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
    torch.cuda.synchronize()
    times["6. Watermarking"] = time.perf_counter() - t0
    
    # Results
    total = sum(times.values())
    audio_duration = len(watermarked_wav) / model.sr
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Step':<40} {'Time':>10} {'%':>8}")
    print("-"*60)
    for step, t in times.items():
        pct = (t / total) * 100
        print(f"{step:<40} {t:>9.3f}s {pct:>7.1f}%")
    print("-"*60)
    print(f"{'TOTAL':<40} {total:>9.3f}s {100:>7.1f}%")
    print(f"{'Audio duration':<40} {audio_duration:>9.2f}s")
    print(f"{'RTF (Real-Time Factor)':<40} {total/audio_duration:>9.3f}")
    print(f"{'Speed':<40} {audio_duration/total:>9.2f}x realtime")
    
    # Save audio
    ta.save("profile_output.wav", torch.from_numpy(watermarked_wav).unsqueeze(0), model.sr)
    print("\nSaved: profile_output.wav")


if __name__ == "__main__":
    profile_viterbox()
