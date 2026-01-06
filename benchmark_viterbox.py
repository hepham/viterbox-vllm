"""Benchmark: Viterbox-TTS (original) vs Chatterbox + Viterbox model"""
import time
import torch
import torchaudio as ta

# Test text
TEXT_VI = "Xin chào! Đây là mô hình chuyển văn bản thành giọng nói tiếng Việt."
AUDIO_PROMPT = "mau.wav"
NUM_RUNS = 5

# Longer text to show T3 speed difference more clearly
TEXT_LONG = """Hà Nội mang vẻ đẹp trầm lắng và cổ kính, nơi quá khứ và hiện tại hòa quyện trong từng con phố, Buổi sáng, thành phố thức dậy cùng mùi hoa sữa thoang thoảng và tiếng rao quen thuộc, Hồ Gươm lặng yên soi bóng những hàng cây xanh, còn những con phố cổ lưu giữ nhịp sống chậm rãi, bình yên, Hà Nội không ồn ào nhưng đủ sâu lắng để khiến ai từng ghé qua đều nhớ mãi"""

def benchmark_chatterbox_viterbox(text, label=""):
    """Benchmark chatterbox with Viterbox model"""
    from example_viterbox import ViterboxTTS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Chatterbox+Viterbox] Loading model on {device}...")
    
    model = ViterboxTTS.from_pretrained(device=device)
    print(f"[Chatterbox+Viterbox] Model loaded!")
    
    # Warmup
    print("[Chatterbox+Viterbox] Warmup...")
    _ = model.generate(TEXT_VI, language_id="vi", audio_prompt_path=AUDIO_PROMPT)
    
    # Benchmark
    print(f"[Chatterbox+Viterbox] Benchmarking {label}...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.time()
        wav = model.generate(text, language_id="vi", audio_prompt_path=AUDIO_PROMPT)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    audio_duration = wav.shape[1] / model.sr
    rtf = avg_time / audio_duration
    
    # Save audio for quality check
    ta.save("output_chatterbox_viterbox.wav", wav.cpu(), model.sr)
    print(f"[Chatterbox+Viterbox] Saved to output_chatterbox_viterbox.wav")
    
    del model
    torch.cuda.empty_cache()
    
    return avg_time, audio_duration, rtf


def benchmark_viterbox_original(text, label=""):
    """Benchmark original viterbox-tts"""
    import sys
    sys.path.insert(0, "temp_viterbox")
    from viterbox import Viterbox as ViterboxTTS
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Viterbox-TTS] Loading model on {device}...")
    
    model = ViterboxTTS.from_pretrained(device=device)
    print(f"[Viterbox-TTS] Model loaded!")
    
    # Warmup
    print("[Viterbox-TTS] Warmup...")
    _ = model.generate(TEXT_VI, language="vi", audio_prompt=AUDIO_PROMPT, split_sentences=False)
    
    # Benchmark
    print(f"[Viterbox-TTS] Benchmarking {label}...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.time()
        wav = model.generate(text, language="vi", audio_prompt=AUDIO_PROMPT, split_sentences=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    audio_duration = wav.shape[1] / model.sr
    rtf = avg_time / audio_duration
    
    # Save audio for quality check
    ta.save("output_viterbox_original.wav", wav.cpu(), model.sr)
    print(f"[Viterbox-TTS] Saved to output_viterbox_original.wav")
    
    del model
    torch.cuda.empty_cache()
    
    return avg_time, audio_duration, rtf


if __name__ == "__main__":
    print("="*70)
    print("BENCHMARK: Chatterbox+Viterbox vs Viterbox-TTS Original")
    print("="*70)
    print(f"Runs per test: {NUM_RUNS}")
    print("="*70)
    
    results = {}
    
    # Long text only
    print("\n>>> LONG TEXT <<<")
    print(f"Text: {TEXT_LONG[:60]}...")
    cb_long = benchmark_chatterbox_viterbox(TEXT_LONG, "long text")
    
    # Clear GPU before running viterbox original
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    vb_long = benchmark_viterbox_original(TEXT_LONG, "long text")
    results["long"] = (cb_long, vb_long)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"{'Test':<12} {'Implementation':<25} {'Time':<10} {'Audio':<10} {'RTF':<8} {'Realtime'}")
    print("-"*70)
    
    for test_name, (cb, vb) in results.items():
        cb_time, cb_dur, cb_rtf = cb
        vb_time, vb_dur, vb_rtf = vb
        print(f"{test_name:<12} {'Chatterbox+Viterbox':<25} {cb_time:.2f}s{'':<5} {cb_dur:.2f}s{'':<5} {cb_rtf:.3f}{'':<3} {1/cb_rtf:.1f}x")
        print(f"{'':<12} {'Viterbox-TTS (original)':<25} {vb_time:.2f}s{'':<5} {vb_dur:.2f}s{'':<5} {vb_rtf:.3f}{'':<3} {1/vb_rtf:.1f}x")
        speedup = vb_time / cb_time
        print(f"{'':<12} {'>>> Speedup:':<25} {speedup:.2f}x faster")
        print("-"*70)
